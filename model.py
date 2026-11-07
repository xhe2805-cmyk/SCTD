from contextlib import nullcontext

import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.bert.modeling_bert import BertModel, BertPreTrainedModel

from global_configs import ACOUSTIC_DIM, VISUAL_DIM
from modules.transformer import TransformerEncoder


VARIANT_ID = "RNPD-SCTD-26"
VARIANT_PAPER = "Selective Cross-View Teacher Distillation for Multimodal Sentiment Analysis"


class GaussianIB(nn.Module):
    def __init__(self, dim=50, hidden=160):
        super().__init__()
        self.encode = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.mu = nn.Linear(hidden, dim)
        self.logvar = nn.Linear(hidden, dim)

    def forward(self, x, stochastic=True):
        representation = self.encode(x)
        mean = self.mu(representation)
        log_variance = self.logvar(representation).clamp(-10.0, 5.0)
        if stochastic:
            noise = torch.randn_like(mean)
            sample = mean + torch.exp(0.5 * log_variance) * noise
        else:
            sample = mean
        prior_kl = 0.5 * (
            mean.square()
            + log_variance.exp()
            - log_variance
            - 1.0
        )
        return sample, prior_kl, mean, log_variance


def _attention_pool(z, mu, lv, mask, scorer):
    valid = mask.bool()
    logits = scorer(mu).squeeze(-1)
    logits = logits.masked_fill(~valid, -1e4)
    weights = logits.softmax(dim=-1)
    weights = weights * valid.to(dtype=z.dtype)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    expanded = weights.unsqueeze(-1)
    pooled_sample = (z * expanded).sum(dim=1)
    pooled_mean = (mu * expanded).sum(dim=1)
    pooled_variance = (
        lv.exp() * expanded.square()
    ).sum(dim=1).clamp_min(1e-6)
    return pooled_sample, pooled_mean, pooled_variance


def _renyi2(z):
    if z.shape[0] < 2:
        return z.new_zeros(())
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        autocast_context = torch.amp.autocast(z.device.type, enabled=False)
    else:
        autocast_context = nullcontext()
    with autocast_context:
        x = z.float().reshape(z.shape[0], -1)
        squared = x.square().sum(dim=1, keepdim=True)
        distances = (
            squared
            + squared.transpose(0, 1)
            - 2 * (x @ x.transpose(0, 1))
        ).clamp_min(0.0)
        count = x.shape[0]
        bandwidth = (
            distances.detach().sum() / (count * (count - 1))
        ).clamp_min(1e-4)
        kernel = torch.exp(-distances / bandwidth)
        return -torch.log2(kernel.square().mean().clamp_min(1e-9))


class PartialViewPosteriorFusion(nn.Module):
    def __init__(self, dim=50):
        super().__init__()
        self.dim = dim
        self.ibs = nn.ModuleList(
            [GaussianIB(dim, hidden=160) for _ in range(3)]
        )
        self.scorers = nn.ModuleList(
            [nn.Linear(dim, 1) for _ in range(3)]
        )
        self.modal_embeddings = nn.Parameter(
            torch.randn(1, 3, dim) * 0.02
        )
        self.cls = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.input_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=5,
            dropout=0.1,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, dim),
        )
        self.fusion_norm = nn.LayerNorm(dim)
        self.fused_ib = GaussianIB(dim, hidden=160)
        self.pred = nn.Linear(dim, 1)
        self._partial_step = 0
        self.configure_variant()

    def configure_variant(
        self,
        *,
        view_consistency=True,
        partial_supervision=True,
        teacher_reliability=True,
        posterior_weight=0.02,
        partial_weight=0.15,
        teacher_temperature=0.75,
        partial_views=1,
        kl_weight=0.001,
        renyi_weight=0.0001,
        deterministic_eval=True,
        dynamic_teacher=True,
        teacher_margin=0.08,
        full_student_weight=0.25,
        teacher_uncertainty_penalty=0.0,
        teacher_warmup_epochs=2,
    ):
        if partial_views not in (1, 3):
            raise ValueError("partial_views must be 1 or 3")
        self.use_consistency = bool(view_consistency)
        self.use_partial_supervision = bool(partial_supervision)
        self.use_reliability = bool(teacher_reliability)
        self.posterior_weight = float(posterior_weight)
        self.partial_weight = float(partial_weight)
        self.teacher_temperature = max(float(teacher_temperature), 0.05)
        self.partial_views = int(partial_views)
        self.kl_weight = float(kl_weight)
        self.renyi_weight = float(renyi_weight)
        self.deterministic_eval = bool(deterministic_eval)
        self.use_dynamic_teacher = bool(dynamic_teacher)
        self.teacher_margin = max(0.0, float(teacher_margin))
        self.full_student_weight = max(0.0, float(full_student_weight))
        self.teacher_uncertainty_penalty = max(
            0.0,
            float(teacher_uncertainty_penalty),
        )
        self.teacher_warmup_epochs = max(0, int(teacher_warmup_epochs))
        self.current_epoch = 0

    def set_epoch(self, epoch):
        self.current_epoch = int(epoch)

    def aggregate(self, pooled, active=None):
        batch_size = pooled.size(0)
        if active is None:
            active = torch.ones(
                batch_size,
                3,
                device=pooled.device,
                dtype=torch.bool,
            )
        token = self.input_norm(pooled + self.modal_embeddings)
        token = token * active.unsqueeze(-1).to(dtype=token.dtype)
        cls_token = self.cls.expand(batch_size, -1, -1)
        tokens = torch.cat((cls_token, token), dim=1)
        cls_not_masked = torch.zeros(
            batch_size,
            1,
            dtype=torch.bool,
            device=pooled.device,
        )
        key_mask = torch.cat((cls_not_masked, ~active), dim=1)
        attention, _ = self.attn(
            tokens,
            tokens,
            tokens,
            key_padding_mask=key_mask,
            need_weights=False,
        )
        updated_tokens = tokens + attention
        summary = updated_tokens[:, 0]
        refined = summary + self.ffn(summary)
        return self.fusion_norm(refined)

    @staticmethod
    def teacher_student_gaussian_kl(t_mu, t_lv, s_mu, s_lv):
        teacher_mean = t_mu.detach()
        teacher_logvar = t_lv.detach()
        teacher_variance = teacher_logvar.clamp(-10.0, 5.0).exp()
        student_variance = s_lv.clamp(-10.0, 5.0).exp()
        location_gap = (teacher_mean - s_mu).square()
        divergence = 0.5 * (
            s_lv
            - teacher_logvar
            + (teacher_variance + location_gap) / student_variance
            - 1.0
        )
        return divergence.mean(dim=-1).clamp_min(0.0)

    @staticmethod
    def content_masks(mask):
        text_mask = mask.bool()
        positions = torch.arange(
            mask.shape[1],
            device=mask.device,
        ).unsqueeze(0)
        length = text_mask.long().sum(dim=-1, keepdim=True)
        content = (
            (positions > 0)
            & (positions < length - 1)
        )
        fallback = (
            (length <= 2)
            & (positions == 0)
        )
        audio_visual_mask = content | fallback
        return text_mask, audio_visual_mask, audio_visual_mask

    def modality_posteriors(self, language, acoustic, visual, mask, stochastic):
        modality_inputs = (language, acoustic, visual)
        posteriors = [
            bottleneck(features, stochastic)
            for bottleneck, features in zip(self.ibs, modality_inputs)
        ]
        masks = self.content_masks(mask)
        pooled = [
            _attention_pool(z, mean, logvar, valid, scorer)[0]
            for (z, _, mean, logvar), valid, scorer in zip(
                posteriors,
                masks,
                self.scorers,
            )
        ]
        evidence = torch.stack(pooled, dim=1)
        return posteriors, masks, pooled, evidence

    @staticmethod
    def checked_active_mask(active_mask, evidence, labels):
        if active_mask is None:
            return None
        if labels is not None:
            raise ValueError(
                "active_mask is for evaluation only; training is unchanged"
            )
        expected_shape = (evidence.shape[0], 3)
        if tuple(active_mask.shape) != expected_shape:
            raise ValueError(
                f"active_mask must be [batch,3], got {tuple(active_mask.shape)}"
            )
        converted = active_mask.to(
            device=evidence.device,
            dtype=torch.bool,
        )
        if not converted.any(dim=1).all():
            raise ValueError("Each sample must retain at least one modality")
        return converted

    def fused_posterior(self, evidence, active, stochastic):
        fused_features = self.aggregate(evidence, active)
        sample, prior_kl, mean, log_variance = self.fused_ib(
            fused_features,
            stochastic,
        )
        prediction = self.pred(sample)
        return prediction, sample, prior_kl, mean, log_variance

    @staticmethod
    def primary_task_loss(prediction, target):
        values = prediction.reshape(-1)
        mae_loss = F.l1_loss(values, target)
        mse_loss = F.mse_loss(values, target)
        return mae_loss + 0.1 * mse_loss

    @staticmethod
    def unimodal_prior_loss(posteriors, masks):
        per_modality = [
            (prior_kl * valid.unsqueeze(-1)).sum((1, 2))
            / (
                valid.to(dtype=prior_kl.dtype).sum(dim=1)
                * prior_kl.size(-1)
            ).clamp_min(1.0)
            for (_, prior_kl, _, _), valid in zip(posteriors, masks)
        ]
        return torch.stack(per_modality, dim=1).sum(dim=1).mean()

    def regularized_task_loss(
        self,
        prediction,
        target,
        posteriors,
        masks,
        pooled,
        fused_prior_kl,
        fused_mean,
    ):
        task = self.primary_task_loss(prediction, target)
        unimodal_prior = self.unimodal_prior_loss(posteriors, masks)
        fused_prior = fused_prior_kl.mean()
        loss = task + self.kl_weight * (
            unimodal_prior + fused_prior
        )
        if self.renyi_weight > 0:
            representations = [*pooled, fused_mean]
            entropy_terms = [
                _renyi2(representation)
                for representation in representations
            ]
            entropy = sum(entropy_terms)
            loss = loss + self.renyi_weight * entropy
        return loss

    def distillation_flags(self):
        distillation = (
            self.use_consistency
            and self.posterior_weight > 0
        )
        partial_supervision = (
            self.use_partial_supervision
            and self.partial_weight > 0
        )
        return distillation, partial_supervision

    def dynamic_selection_active(self, distillation):
        return (
            self.use_dynamic_teacher
            and distillation
            and self.current_epoch > self.teacher_warmup_epochs
        )

    def partial_view_ids(self, dynamic_now):
        if dynamic_now or self.partial_views == 3:
            return (0, 1, 2)
        return (self._partial_step % 3,)

    @staticmethod
    def available_views(evidence, missing_index):
        mask = torch.ones(
            evidence.size(0),
            3,
            dtype=torch.bool,
            device=evidence.device,
        )
        mask[:, missing_index] = False
        return mask

    def encode_partial_views(
        self,
        evidence,
        missing_indices,
        stochastic,
        target,
        supervise,
    ):
        partial_posteriors = []
        auxiliary_losses = []
        for missing_index in missing_indices:
            available = self.available_views(evidence, missing_index)
            prediction, _, _, mean, logvar = self.fused_posterior(
                evidence,
                available,
                stochastic,
            )
            prediction = prediction.reshape(-1)
            partial_posteriors.append(
                (mean, logvar, prediction)
            )
            if supervise:
                auxiliary_losses.append(
                    F.smooth_l1_loss(prediction, target)
                )
        return partial_posteriors, auxiliary_losses

    @staticmethod
    def stacked_view_posteriors(full_mean, full_logvar, partial_views):
        means = torch.stack(
            [full_mean] + [view[0] for view in partial_views],
            dim=1,
        )
        logvars = torch.stack(
            [full_logvar] + [view[1] for view in partial_views],
            dim=1,
        )
        return means, logvars

    def candidate_errors_and_scores(self, means, logvars, target):
        with torch.no_grad():
            mean_predictions = self.pred(
                means.detach()
            ).squeeze(-1)
            errors = (
                mean_predictions - target[:, None]
            ).abs()
            posterior_std = logvars.detach().clamp(
                -10.0,
                5.0,
            ).exp().mean(dim=-1).sqrt()
            scores = (
                errors + self.teacher_uncertainty_penalty * posterior_std
            )
        return errors, scores

    def choose_teacher(self, scores):
        with torch.no_grad():
            best_partial_score, best_partial_index = scores[:, 1:].min(
                dim=1
            )
            choose_partial = (
                best_partial_score + self.teacher_margin < scores[:, 0]
            )
            teacher = torch.where(
                choose_partial,
                best_partial_index + 1,
                torch.zeros_like(best_partial_index),
            )
        return teacher

    def per_student_weights(self, errors, teacher_index):
        with torch.no_grad():
            teacher_error = errors.gather(
                1,
                teacher_index[:, None],
            )
            if self.use_reliability:
                reliability = torch.exp(
                    -teacher_error / self.teacher_temperature
                )
            else:
                reliability = torch.ones_like(teacher_error)
            need = torch.sigmoid(
                (errors - teacher_error) / self.teacher_temperature
            )
            role = torch.ones_like(errors)
            role.scatter_(
                1,
                teacher_index[:, None],
                0.0,
            )
            role[:, 0] *= self.full_student_weight
            weights = (
                reliability * need * role
            ).clamp(0.0, 1.0)
        return weights

    @staticmethod
    def selected_teacher_posterior(means, logvars, teacher_index):
        latent_dim = means.shape[-1]
        gather_index = teacher_index[:, None, None].expand(
            -1,
            1,
            latent_dim,
        )
        teacher_mean = means.gather(
            1,
            gather_index,
        )
        teacher_logvar = logvars.gather(
            1,
            gather_index,
        )
        return teacher_mean, teacher_logvar

    def adaptive_distillation(
        self,
        full_mean,
        full_logvar,
        partial_views,
        target,
    ):
        means, logvars = self.stacked_view_posteriors(
            full_mean,
            full_logvar,
            partial_views,
        )
        errors, scores = self.candidate_errors_and_scores(
            means,
            logvars,
            target,
        )
        teacher = self.choose_teacher(scores)
        weights = self.per_student_weights(
            errors,
            teacher,
        )
        teacher_mean, teacher_logvar = self.selected_teacher_posterior(
            means,
            logvars,
            teacher,
        )
        divergence = self.teacher_student_gaussian_kl(
            teacher_mean,
            teacher_logvar,
            means,
            logvars,
        )
        return (
            weights * divergence
        ).sum(dim=1).mean() / 3.0

    def fixed_teacher_distillation(
        self,
        full_mean,
        full_logvar,
        partial_views,
        target,
    ):
        with torch.no_grad():
            full_error = (
                self.pred(full_mean.detach()).reshape(-1) - target
            ).abs()
            if self.use_reliability:
                reliability = torch.exp(
                    -full_error / self.teacher_temperature
                )
            else:
                reliability = torch.ones_like(full_error)
        losses = []
        for student_mean, student_logvar, _ in partial_views:
            with torch.no_grad():
                student_error = (
                    self.pred(student_mean.detach()).reshape(-1) - target
                ).abs()
                difference = student_error - full_error
                need = torch.sigmoid(
                    difference / self.teacher_temperature
                )
                weight = (reliability * need).clamp(0.0, 1.0)
            divergence = self.teacher_student_gaussian_kl(
                full_mean,
                full_logvar,
                student_mean,
                student_logvar,
            )
            losses.append(
                (weight * divergence).mean()
            )
        return torch.stack(losses).mean()

    def training_view_loss(
        self,
        evidence,
        target,
        full_mean,
        full_logvar,
        stochastic,
    ):
        distill_on, auxiliary_on = self.distillation_flags()
        if not self.training or not (distill_on or auxiliary_on):
            return evidence.new_zeros(())
        dynamic_now = self.dynamic_selection_active(distill_on)
        missing_indices = self.partial_view_ids(dynamic_now)
        self._partial_step += 1
        partial_views, auxiliary_losses = self.encode_partial_views(
            evidence,
            missing_indices,
            stochastic,
            target,
            auxiliary_on,
        )
        additional_loss = evidence.new_zeros(())
        if auxiliary_losses:
            additional_loss = additional_loss + self.partial_weight * (
                torch.stack(auxiliary_losses).mean()
            )
        if distill_on:
            if dynamic_now:
                transfer = self.adaptive_distillation(
                    full_mean,
                    full_logvar,
                    partial_views,
                    target,
                )
            else:
                transfer = self.fixed_teacher_distillation(
                    full_mean,
                    full_logvar,
                    partial_views,
                    target,
                )
            additional_loss = additional_loss + (
                self.posterior_weight * transfer
            )
        return additional_loss

    def forward(
        self,
        l,
        a,
        v,
        mask,
        labels=None,
        return_features=False,
        active_mask=None,
    ):
        stochastic = self.training or not self.deterministic_eval
        posteriors, masks, pooled, evidence = self.modality_posteriors(
            l,
            a,
            v,
            mask,
            stochastic,
        )
        active_mask = self.checked_active_mask(
            active_mask,
            evidence,
            labels,
        )
        prediction, _, prior_kl, mean, logvar = self.fused_posterior(
            evidence,
            active_mask,
            stochastic,
        )
        if labels is None:
            if return_features:
                return prediction, mean
            return prediction, prediction.new_zeros(())
        target = labels.reshape(-1)
        loss = self.regularized_task_loss(
            prediction,
            target,
            posteriors,
            masks,
            pooled,
            prior_kl,
            mean,
        )
        loss = loss + self.training_view_loss(
            evidence,
            target,
            mean,
            logvar,
            stochastic,
        )
        return prediction, loss


class SCTDModel(BertPreTrainedModel):
    def __init__(self, config, multimodal_config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.d_l = 50
        self.proj_l = nn.Conv1d(
            config.hidden_size,
            self.d_l,
            3,
            padding=1,
            bias=False,
        )
        self.proj_a = nn.Conv1d(
            ACOUSTIC_DIM,
            self.d_l,
            3,
            padding=1,
            bias=False,
        )
        self.proj_v = nn.Conv1d(
            VISUAL_DIM,
            self.d_l,
            3,
            padding=1,
            bias=False,
        )
        self.transa = self.get_network(layers=3)
        self.transv = self.get_network(layers=3)
        self.fusion = PartialViewPosteriorFusion(dim=self.d_l)
        self.init_weights()

    @property
    def latent_width(self):
        return self.d_l

    def get_network(self, layers=3):
        return TransformerEncoder(
            embed_dim=self.d_l,
            num_heads=5,
            layers=layers,
            attn_dropout=0.5,
            relu_dropout=0.3,
            res_dropout=0.3,
            embed_dropout=0.2,
            attn_mask=False,
        )

    def text_features(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        position_ids,
        head_mask,
        inputs_embeds,
        output_attentions,
        output_hidden_states,
    ):
        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )[0]
        return self.proj_l(
            output.transpose(1, 2)
        ).transpose(1, 2)

    def acoustic_features(self, acoustic):
        projected = self.proj_a(
            acoustic.transpose(1, 2)
        ).permute(2, 0, 1)
        return self.transa(projected).permute(1, 0, 2)

    def visual_features(self, visual):
        projected = self.proj_v(
            visual.transpose(1, 2)
        ).permute(2, 0, 1)
        return self.transv(projected).permute(1, 0, 2)

    def forward(
        self,
        input_ids,
        visual,
        acoustic,
        label_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_loss=False,
        return_features=False,
        active_mask=None,
    ):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        language = self.text_features(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            output_attentions,
            output_hidden_states,
        )
        acoustic_features = self.acoustic_features(acoustic)
        visual_features = self.visual_features(visual)
        if return_features and return_loss:
            raise ValueError("return_features is for inference only")
        result = self.fusion(
            language,
            acoustic_features,
            visual_features,
            attention_mask,
            label_ids if return_loss else None,
            return_features=return_features,
            active_mask=active_mask,
        )
        if return_features:
            return result
        prediction, loss = result
        return (prediction, loss) if return_loss else prediction

    def test(
        self,
        input_ids,
        visual,
        acoustic,
        attention_mask=None,
        token_type_ids=None,
        return_features=False,
        active_mask=None,
        **kwargs,
    ):
        return self(
            input_ids,
            visual,
            acoustic,
            None,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_loss=False,
            return_features=return_features,
            active_mask=active_mask,
        )


fusion = PartialViewPosteriorFusion
