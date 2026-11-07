import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model import GaussianIB, PartialViewPosteriorFusion


def test_gaussian_bottleneck():
    torch.manual_seed(41)
    bottleneck = GaussianIB(dim=50, hidden=160)
    features = torch.randn(4, 9, 50)
    sampled, divergence, mean, log_variance = bottleneck(features)
    assert sampled.shape == features.shape
    assert divergence.shape == features.shape
    assert torch.allclose(bottleneck(features, stochastic=False)[0], mean)
    assert torch.isfinite(log_variance).all()


def test_full_and_partial_predictions():
    torch.manual_seed(41)
    fusion = PartialViewPosteriorFusion(dim=50)
    fusion.eval()
    language, audio, vision = [torch.randn(4, 9, 50) for _ in range(3)]
    token_mask = torch.ones(4, 9, dtype=torch.long)
    full_prediction, _ = fusion(language, audio, vision, token_mask)
    partial_mask = torch.ones(4, 3, dtype=torch.bool)
    partial_mask[:, 1] = False
    partial_prediction, _ = fusion(
        language,
        audio,
        vision,
        token_mask,
        active_mask=partial_mask,
    )
    assert full_prediction.shape == (4, 1)
    assert partial_prediction.shape == (4, 1)
    assert torch.isfinite(full_prediction).all()
    assert torch.isfinite(partial_prediction).all()


def test_margin_gated_teacher_selection():
    fusion = PartialViewPosteriorFusion(dim=50)
    scores = torch.tensor(
        [
            [0.20, 0.10, 0.19, 0.40],
            [0.20, 0.16, 0.17, 0.40],
            [0.30, 0.40, 0.10, 0.20],
        ]
    )
    indices = fusion.choose_teacher(scores)
    assert indices.tolist() == [1, 0, 2]


def test_training_backward():
    torch.manual_seed(41)
    fusion = PartialViewPosteriorFusion(dim=50)
    fusion.train()
    fusion.set_epoch(3)
    language, audio, vision = [torch.randn(4, 9, 50) for _ in range(3)]
    token_mask = torch.ones(4, 9, dtype=torch.long)
    targets = torch.tensor([2.0, -1.0, 0.0, 1.0])
    predictions, objective = fusion(
        language,
        audio,
        vision,
        token_mask,
        labels=targets,
    )
    assert predictions.shape == (4, 1)
    assert torch.isfinite(objective)
    objective.backward()
    gradients = [p.grad for p in fusion.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)


if __name__ == '__main__':
    test_gaussian_bottleneck()
    test_full_and_partial_predictions()
    test_margin_gated_teacher_selection()
    test_training_backward()
    print('All SCTD component checks passed.')
