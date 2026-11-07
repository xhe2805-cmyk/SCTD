import argparse
from contextlib import contextmanager
import importlib
import json
from pathlib import Path
import pickle
import random
import tempfile
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from transformers import BertConfig, BertTokenizer
import global_configs

ROOT = Path(__file__).resolve().parent
METHOD = 'RNPD-SCTD-26'

def parse_args(argv=None):
    p = argparse.ArgumentParser(description='RNPD-SCTD-26: selective cross-view posterior teacher distillation.')
    p.add_argument('--dataset', choices=['mosi','mosei'], default='mosi')
    numeric_defaults = {
        'max_seq_length': 50,
        'train_batch_size': 32,
        'dev_batch_size': 128,
        'test_batch_size': 128,
        'n_epochs': 48,
        'gradient_accumulation_step': 1,
        'seed': 0,
    }
    for key, default in numeric_defaults.items():
        p.add_argument('--'+key, '--'+key.replace('_','-'), dest=key, type=int, default=default)
    p.add_argument('--learning_rate','--learning-rate', type=float, default=1e-5)
    p.add_argument('--beta_shift', type=float, default=1.)
    p.add_argument('--dropout_prob', type=float, default=.5)
    p.add_argument('--warmup_proportion', type=float, default=.1, help='Legacy compatibility argument')
    p.add_argument('--model', default='bert-base-uncased')
    p.add_argument('--bert_path','--bert-path', default='')
    p.add_argument('--data_root','--data-root', default='datasets')
    p.add_argument('--device', default='auto')
    p.add_argument('--gpu', type=int, default=None)
    p.add_argument('--save_dir','--save-dir', default='checkpoints')
    p.add_argument('--checkpoint', default='')
    p.add_argument('--eval_only','--eval-only', action='store_true')
    p.add_argument('--include_zero','--include-zero', action='store_true')
    p.add_argument('--disable_view_consistency', action='store_true',help='Disable partial-view posterior distillation')
    p.add_argument('--disable_partial_supervision', action='store_true',help='Disable partial-view supervised task loss')
    p.add_argument('--disable_teacher_reliability', action='store_true',help='Use uniform student distillation trust')
    p.add_argument('--posterior_weight',type=float,default=.02,help='Reliability-gated posterior KL weight')
    p.add_argument('--partial_weight',type=float,default=.15,help='Partial-view sentiment supervision weight')
    p.add_argument('--teacher_temperature',type=float,default=.75,help='Teacher reliability temperature')
    p.add_argument('--partial_views',type=int,choices=[1,3],default=1,help='Number of missing views per training batch')
    p.add_argument('--kl_weight',type=float,default=.001)
    p.add_argument('--renyi_weight',type=float,default=.0001)
    p.add_argument('--disable_dynamic_teacher', action='store_true',
                   help='Ablation: use original full-view teacher, not dynamic view search')
    p.add_argument('--teacher_margin',type=float,default=.08,
                   help='Minimum score advantage required to replace full teacher')
    p.add_argument('--full_student_weight',type=float,default=.25,
                   help='Relative KL weight for the full view when a partial view teaches it')
    p.add_argument('--teacher_uncertainty_penalty',type=float,default=0.,
                   help='Optional penalty on posterior std during teacher selection')
    p.add_argument('--teacher_warmup_epochs',type=int,default=2,
                   help='Epochs to keep the original fixed-teacher RNPD path')
    p.add_argument('--select_metric',choices=['mse','f1'],default='mse',
                   help='Validation-only checkpoint selection')
    p.add_argument('--stochastic_eval', action='store_true', help='Draw posterior samples during evaluation')
    p.add_argument('--no_epoch_test', action='store_true', help='Skip intermediate TEST evaluation; run final TEST only')
    args=p.parse_args(argv)
    positive_integer_fields = (
        'max_seq_length',
        'train_batch_size',
        'dev_batch_size',
        'test_batch_size',
        'n_epochs',
        'gradient_accumulation_step',
    )
    for key in positive_integer_fields:
        if getattr(args,key) < (3 if key=='max_seq_length' else 1):
            p.error(key+' is too small')
    if args.teacher_temperature <= 0:
        p.error('--teacher_temperature must be positive')
    nonnegative_weights = (
        args.kl_weight,
        args.renyi_weight,
        args.posterior_weight,
        args.partial_weight,
        args.teacher_margin,
        args.full_student_weight,
        args.teacher_uncertainty_penalty,
    )
    if min(nonnegative_weights) < 0:
        p.error('Regularization weights and teacher controls must be nonnegative')
    if args.teacher_warmup_epochs < 0:
        p.error('--teacher_warmup_epochs must be nonnegative')
    if not 0 <= args.seed < 2**32 or args.learning_rate <= 0:
        p.error('Invalid seed or learning rate')
    return args

def root_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()

def choose_device(args):
    name = ('cpu' if args.gpu < 0 else f'cuda:{args.gpu}') if args.gpu is not None else args.device
    if name == 'auto':
        name = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(name)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA was requested but is unavailable; use --device cpu')
        if (device.index or 0) >= torch.cuda.device_count():
            raise ValueError('Requested GPU index exceeds visible devices')
        torch.cuda.set_device(device)
    return device

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

@contextmanager
def evaluation_rng(seed, device):
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(seed)
        if device.type == 'cuda':
            with torch.cuda.device(device):
                torch.cuda.manual_seed(seed)
        yield

def read_data(path):
    if not path.is_file():
        raise FileNotFoundError('Dataset not found: ' + str(path))
    with path.open('rb') as handle:
        data = pickle.load(handle, encoding='latin1')
    if not isinstance(data, dict):
        raise ValueError('Expected word-aligned sentiment dataset with train/dev/test splits')
    return data

def get_split(data, name):
    key = 'valid' if name == 'dev' and 'dev' not in data else name
    if key not in data or not len(data[key]):
        raise ValueError('Missing or empty split: ' + name)
    if isinstance(data[key], dict):
        raise ValueError('Use word-aligned tuples; MMSA dictionaries cannot just be renamed')
    return data[key]

def make_dataset(examples, tokenizer, length, expected_dims=None):
    ids_all, visuals, acoustics, masks, types, labels = [], [], [], [], [], []
    dims = expected_dims
    for index, example in enumerate(examples):
        try:
            (words, visual, acoustic), label, _ = example
        except (ValueError, TypeError) as exc:
            raise ValueError(f'Sample {index}: expected ((words, vision, audio), label, id)') from exc
        if isinstance(words, (str, bytes)):
            raise ValueError('Expected a word list, not a sentence string')
        visual, acoustic = np.asarray(visual, dtype=np.float32), np.asarray(acoustic, dtype=np.float32)
        if visual.ndim != 2 or acoustic.ndim != 2 or not (len(words) == len(visual) == len(acoustic)):
            raise ValueError(f'Sample {index}: require raw word-aligned feature rows. HME BERT-padded tuples are a different format')
        current_dims = (acoustic.shape[1], visual.shape[1])
        if min(current_dims) < 1:
            raise ValueError('Empty feature width')
        if dims is None:
            dims = current_dims
        if tuple(dims) != current_dims:
            raise ValueError(f'Feature dimensions {current_dims} do not match {dims}')
        visual = np.nan_to_num(visual, nan=0., posinf=0., neginf=0.)
        acoustic = np.nan_to_num(acoustic, nan=0., posinf=0., neginf=0.)
        tokens, inversions = [], []
        for word_index, word in enumerate(words):
            word = word.decode('utf-8') if isinstance(word, bytes) else str(word)
            pieces = tokenizer.tokenize(word) or [tokenizer.unk_token]
            tokens.extend(pieces)
            inversions.extend([word_index] * len(pieces))
        tokens, inversions = tokens[:length - 2], inversions[:length - 2]
        ids = tokenizer.convert_tokens_to_ids([tokenizer.cls_token] + tokens + [tokenizer.sep_token])
        ids_all.append(ids + [tokenizer.pad_token_id] * (length - len(ids)))
        masks.append([1] * len(ids) + [0] * (length - len(ids)))
        types.append([0] * length)
        v, a = np.zeros((length, dims[1]), np.float32), np.zeros((length, dims[0]), np.float32)
        v[1:1 + len(inversions)], a[1:1 + len(inversions)] = visual[inversions], acoustic[inversions]
        visuals.append(v)
        acoustics.append(a)
        value = np.asarray(label, dtype=np.float32).reshape(-1)
        if len(value) != 1 or not np.isfinite(value[0]):
            raise ValueError('Each regression target must be one finite scalar')
        labels.append(value[0])
    return TensorDataset(torch.tensor(ids_all, dtype=torch.long), torch.from_numpy(np.stack(visuals)),
                         torch.from_numpy(np.stack(acoustics)), torch.tensor(masks, dtype=torch.long),
                         torch.tensor(types, dtype=torch.long), torch.from_numpy(np.asarray(labels, dtype=np.float32))), tuple(dims)

def loader(dataset, batch_size, shuffle=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False, num_workers=0)

def metrics(pred, truth, include_zero=False):
    pred, truth = np.asarray(pred, dtype=np.float64), np.asarray(truth, dtype=np.float64)
    if not len(pred) or pred.shape != truth.shape or not np.isfinite(pred).all() or not np.isfinite(truth).all():
        raise ValueError('Predictions and targets must be finite, matching, nonempty vectors')
    selected = np.ones(len(truth), dtype=bool) if include_zero else truth != 0
    result = {'Acc_2': None, 'F1': None}
    if selected.any():
        result['Acc_2'] = float(accuracy_score(truth[selected] >= 0, pred[selected] >= 0))
        result['F1'] = float(f1_score(truth[selected] >= 0, pred[selected] >= 0, average='weighted', zero_division=0))
    result['Acc_7'] = float(np.mean(np.round(np.clip(pred, -3, 3)) == np.round(np.clip(truth, -3, 3))))
    result['MAE'] = float(np.mean(np.abs(pred - truth)))
    result['Corr'] = (float(np.corrcoef(pred, truth)[0, 1]) if len(pred) > 1 and pred.std() > 1e-12 and truth.std() > 1e-12 else None)
    result['MSE'] = float(np.mean((pred - truth)**2))
    return result

def format_metrics(result):
    return '  '.join(f'{name}=' + ('NA' if value is None else f'{value:.4f}') for name, value in result.items())

def format_epoch_line(epoch, split, result):
    result = result or {}
    fields = []
    for key in ('Acc_2', 'F1', 'Acc_7', 'MAE', 'Corr', 'MSE'):
        value = result.get(key)
        display = 'NA' if value is None else f'{value:.4f}'
        fields.append(f'{key}={display:>7}')
    objective = result.get('Objective')
    objective_display = '--' if objective is None else f'{objective:.4f}'
    fields.append(f'Objective={objective_display:>8}')
    return f'[RNPD-SCTD-26][Epoch {epoch}][{split:<13}] ' + '  '.join(fields)

def train_epoch(model, data_loader, optimizers, args, device, epoch):
    model.train()
    model.fusion.set_epoch(epoch)
    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)
    predictions, targets, total_loss, seen = [], [], 0., 0
    for step, batch in enumerate(tqdm(data_loader, desc=f'Epoch {epoch}/{args.n_epochs}', leave=True)):
        ids, v, a, mask, segments, y = [item.to(device) for item in batch]
        prediction, loss = model(ids, v, a, y, attention_mask=mask, token_type_ids=segments, return_loss=True)
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite training loss at batch {step}')
        start = step // args.gradient_accumulation_step * args.gradient_accumulation_step
        end = min(start + args.gradient_accumulation_step, len(data_loader))
        group_samples = min(end * data_loader.batch_size, len(data_loader.dataset)) - start * data_loader.batch_size
        (loss * len(y) / group_samples).backward()
        if step + 1 == end:
            for optimizer in optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        total_loss += float(loss.detach()) * len(y)
        seen += len(y)
        predictions.extend(prediction.detach().cpu().reshape(-1).tolist())
        targets.extend(y.detach().cpu().reshape(-1).tolist())
    result = metrics(predictions, targets, args.include_zero)
    result['Objective'] = total_loss / seen
    return result

def evaluate(model, data_loader, args, device):
    model.eval()
    predictions, targets = [], []
    with evaluation_rng((args.seed + 9876) % 2**32, device), torch.no_grad():
        for batch in data_loader:
            ids, v, a, mask, segments, y = [item.to(device) for item in batch]
            prediction = model.test(ids, v, a, attention_mask=mask, token_type_ids=segments)
            predictions.extend(prediction.detach().cpu().reshape(-1).tolist())
            targets.extend(y.detach().cpu().reshape(-1).tolist())
    return metrics(predictions, targets, args.include_zero)

def tokenizer_files(tokenizer):
    with tempfile.TemporaryDirectory(prefix='sctd-tokenizer-') as directory:
        tokenizer.save_pretrained(directory)
        return {p.name: p.read_bytes() for p in Path(directory).iterdir() if p.is_file()}

def load_checkpoint(path):
    if not path.is_file():
        raise FileNotFoundError('Checkpoint not found: ' + str(path))
    payload = torch.load(path, map_location='cpu')
    if payload.get('method') != METHOD:
        raise ValueError('Checkpoint belongs to a different variant; use an RNPD-SCTD-26 checkpoint')
    return payload

def configure_model(dims):
    global_configs.ACOUSTIC_DIM, global_configs.VISUAL_DIM = dims
    module = importlib.import_module('model')
    if getattr(module, 'VARIANT_ID', None) != METHOD:
        raise ValueError('main.py and model.py are from different SCTD variants')
    module.ACOUSTIC_DIM, module.VISUAL_DIM = dims
    return module.SCTDModel

def set_variant_options(model, settings):
    model.fusion.configure_variant(
        view_consistency=not getattr(settings,'disable_view_consistency',False),
        partial_supervision=not getattr(settings,'disable_partial_supervision',False),
        teacher_reliability=not getattr(settings,'disable_teacher_reliability',False),
        posterior_weight=getattr(settings,'posterior_weight',.02),
        partial_weight=getattr(settings,'partial_weight',.15),
        teacher_temperature=getattr(settings,'teacher_temperature',.75),
        partial_views=getattr(settings,'partial_views',1),
        kl_weight=getattr(settings,'kl_weight',.001),
        renyi_weight=getattr(settings,'renyi_weight',.0001),
        deterministic_eval=not getattr(settings,'stochastic_eval',False),
        dynamic_teacher=not getattr(settings,'disable_dynamic_teacher',False),
        teacher_margin=getattr(settings,'teacher_margin',.08),
        full_student_weight=getattr(settings,'full_student_weight',.25),
        teacher_uncertainty_penalty=getattr(settings,'teacher_uncertainty_penalty',0.),
        teacher_warmup_epochs=getattr(settings,'teacher_warmup_epochs',2),
    )

def restore(payload, device):
    SCTD = configure_model(tuple(payload['feature_dims']))
    old_config = payload['experiment_config']
    multi = SimpleNamespace(beta_shift=old_config['beta_shift'], dropout_prob=old_config['dropout_prob'])
    model = SCTD(BertConfig.from_dict(payload['bert_config']), multi)
    model.load_state_dict(payload['model_state_dict'], strict=True)
    saved_settings = payload['experiment_config']
    set_variant_options(model, SimpleNamespace(**payload['experiment_config']))
    return model.to(device)

def restore_tokenizer(payload):
    with tempfile.TemporaryDirectory(prefix='sctd-test-tokenizer-') as directory:
        for name, content in payload['tokenizer_files'].items():
            if Path(name).name != name:
                raise ValueError('Invalid embedded tokenizer file name')
            (Path(directory) / name).write_bytes(content)
        return BertTokenizer.from_pretrained(directory, local_files_only=True)

def create_optimizers(model, learning_rate):
    bert_parameters = list(model.bert.parameters())
    bert_ids = {
        id(parameter)
        for parameter in bert_parameters
    }
    other_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in bert_ids
    ]
    bert_optimizer = torch.optim.Adam(
        bert_parameters,
        lr=learning_rate,
    )
    remaining_optimizer = torch.optim.Adam(
        other_parameters,
        lr=2 * learning_rate,
    )
    return bert_optimizer, remaining_optimizer

def fit(model, train_loader, valid_loader, args, tokenizer, dims, device, checkpoint_path, test_loader):
    optimizers = create_optimizers(model, args.learning_rate)
    files = tokenizer_files(tokenizer)
    best = -float('inf') if args.select_metric == 'f1' else float('inf')
    best_epoch = None
    best_test_result = None
    for epoch in range(1, args.n_epochs + 1):
        train_result = train_epoch(model, train_loader, optimizers, args, device, epoch)
        valid_result = evaluate(model, valid_loader, args, device)
        test_result = None if args.no_epoch_test else evaluate(model, test_loader, args, device)
        selection_value=valid_result['F1'] if args.select_metric=='f1' else valid_result['MSE']
        if selection_value is None:
            raise ValueError('The requested validation selection metric is undefined')
        new_best=(selection_value>best if args.select_metric=='f1' else selection_value<best)
        if new_best:
            best = selection_value
            best_epoch = epoch
            best_test_result = dict(test_result) if test_result is not None else None
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {'method': METHOD, 'dataset': args.dataset, 'seed': args.seed,
                       'epoch': epoch, 'best_validation_mse': valid_result['MSE'], 'best_validation_selection_score': best,
                       'architecture_ablation': bool(args.disable_view_consistency or args.disable_partial_supervision),
                       'best_validation_f1': valid_result['F1'], 'selection_metric': 'VAL_'+args.select_metric.upper(), 'feature_dims': list(dims),
                       'best_validation_test_metrics': best_test_result,
                       'bert_config': model.config.to_dict(), 'experiment_config': vars(args).copy(),
                       'model_state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                       'tokenizer_files': files}
            temporary = checkpoint_path.with_suffix(checkpoint_path.suffix + '.tmp')
            torch.save(payload, temporary)
            temporary.replace(checkpoint_path)
        lines = [format_epoch_line(epoch, 'TRAIN', train_result),
                 format_epoch_line(epoch, 'VAL', valid_result),
                 format_epoch_line(epoch, 'TEST', test_result),
                 format_epoch_line(epoch, 'BEST TEST', best_test_result)
                 + (f'  BestEpoch={best_epoch}' if best_epoch is not None else '')]
        print('\n'.join(lines), flush=True)
        if new_best:
            print(f'New BEST VAL {args.select_metric.upper()}={best:.6f} -> {checkpoint_path}', flush=True)


def experiment_tag(args):
    tag = 'RNPD_SCTD26'
    flags = (
        (args.disable_view_consistency, 'NODISTILL'),
        (args.disable_partial_supervision, 'NOPARTIAL'),
        (args.disable_teacher_reliability, 'UNIFORMTRUST'),
        (args.stochastic_eval, 'STOCH'),
        (args.disable_dynamic_teacher, 'FIXEDTEACHER'),
    )
    for enabled, suffix in flags:
        if enabled:
            tag += '_' + suffix
    if args.posterior_weight != 0.02:
        tag += f'_P{args.posterior_weight:g}'
    if args.partial_views != 1:
        tag += f'_V{args.partial_views}'
    if args.select_metric == 'f1':
        tag += '_VALF1'
    return tag


def resolve_bert_source(args):
    local_default = ROOT / 'bert-base-uncased'
    if args.bert_path:
        local_path = root_path(args.bert_path)
        if local_path.is_dir():
            return str(local_path)
        return args.bert_path
    if (local_default / 'config.json').is_file():
        return str(local_default)
    return args.model


def check_checkpoint_options(checkpoint, args):
    stored = checkpoint['experiment_config']
    options = (
        'disable_view_consistency',
        'disable_partial_supervision',
        'disable_teacher_reliability',
        'stochastic_eval',
        'disable_dynamic_teacher',
    )
    for option in options:
        configured = bool(stored.get(option, False))
        requested = bool(getattr(args, option))
        if configured != requested:
            raise ValueError(
                f'Checkpoint ablation mode conflicts with CLI setting: {option}'
            )


def main(argv=None):
    args = parse_args(argv)
    device = choose_device(args)
    run_tag = experiment_tag(args)
    checkpoint_path = root_path(args.checkpoint or f'{args.save_dir}/{run_tag}_{args.dataset}_seed{args.seed}_best.pth')
    data_path = root_path(args.data_root) / (args.dataset + '.pkl')
    data = read_data(data_path)
    if args.eval_only:
        payload = load_checkpoint(checkpoint_path)
        check_checkpoint_options(payload, args)
        if payload['dataset'] != args.dataset:
            raise ValueError('Checkpoint dataset does not match --dataset')
        saved = SimpleNamespace(**payload['experiment_config'])
        setup_seed(saved.seed)
        tokenizer = restore_tokenizer(payload)
        test_data, _ = make_dataset(get_split(data, 'test'), tokenizer, saved.max_seq_length, tuple(payload['feature_dims']))
        model = restore(payload, device)
    else:
        setup_seed(args.seed)
        source = resolve_bert_source(args)
        tokenizer = BertTokenizer.from_pretrained(source)
        config = BertConfig.from_pretrained(source)
        if args.max_seq_length > config.max_position_embeddings:
            raise ValueError('max_seq_length exceeds BERT position capacity')
        train_data, dims = make_dataset(get_split(data, 'train'), tokenizer, args.max_seq_length)
        valid_data, _ = make_dataset(get_split(data, 'dev'), tokenizer, args.max_seq_length, dims)
        epoch_test_data, _ = make_dataset(get_split(data, 'test'), tokenizer, args.max_seq_length, dims)
        SCTD = configure_model(dims)
        multimodal_config = SimpleNamespace(beta_shift=args.beta_shift, dropout_prob=args.dropout_prob)
        model = SCTD.from_pretrained(source, config=config, multimodal_config=multimodal_config, _fast_init=False).to(device)
        set_variant_options(model, args)
        print(f'RNPD-SCTD-26 | {args.dataset} | seed={args.seed} | device={device} | audio/vision={dims}', flush=True)
        print(f'BERT: {source}\nDataset: {data_path}\nBest checkpoint: {checkpoint_path}', flush=True)
        print(f'Selection: VAL {args.select_metric.upper()}; {args.n_epochs} epochs; no added early stopping; learning rates={args.learning_rate:g}/{2*args.learning_rate:g}', flush=True)
        fit(model, loader(train_data, args.train_batch_size, True), loader(valid_data, args.dev_batch_size),
            args, tokenizer, dims, device, checkpoint_path, loader(epoch_test_data, args.test_batch_size))
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        payload = load_checkpoint(checkpoint_path)
        saved = SimpleNamespace(**payload['experiment_config'])
        model = restore(payload, device)
        test_data, _ = make_dataset(get_split(data, 'test'), tokenizer, saved.max_seq_length, tuple(dims))
    result = evaluate(model, loader(test_data, saved.test_batch_size), saved, device)
    print(f'Loaded BEST epoch={payload["epoch"]} {payload["selection_metric"]}={payload.get("best_validation_selection_score", payload["best_validation_mse"]):.6f}', flush=True)
    print('[RNPD-SCTD-26][TEST] ' + format_metrics(result), flush=True)
    print(
        'Final TEST: ' + json.dumps(
            result,
            sort_keys=True,
            allow_nan=False,
        ),
        flush=True,
    )
    return result


if __name__ == '__main__':
    main()
