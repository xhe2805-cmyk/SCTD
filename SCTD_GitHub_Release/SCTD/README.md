# SCTD: Selective Cross-View Teacher Distillation for Multimodal Sentiment Analysis

SCTD is a multimodal sentiment analysis method for linguistic, acoustic, and visual inputs. Instead of assuming that the full-modality view is always the most reliable teacher, SCTD selects a teacher among the full view and three leave-one-modality-out views **during training** and transfers a Gaussian posterior to the other views. The default inference path uses all three modalities and does not perform teacher selection.

This repository packages the previously supplied SCTD implementation in the same public-source layout as the [CVSG repository](https://github.com/xhe2805-cmyk/CVSG). The original model computations and checkpoint parameter names are retained. For publication, `main.py` and `model.py` have been **refactored for readability** (500 and 800 lines, respectively), with implementation-unrelated comments removed and the pretrained-model wrapper renamed to `SCTDModel`. Equivalence checks using identically initialized model weights confirmed matching forward predictions, losses, and gradients for the original and refactored fusion modules. This is not a claim of end-to-end retraining or benchmark reproduction.

## Overview

SCTD consists of three modules:

1. **Unimodal Gaussian Encoding:** BERT and acoustic/visual temporal encoders produce aligned sequences. Modality-specific Gaussian bottlenecks and masked attention pooling produce modality embeddings.
2. **Shared Attention Fusion:** a shared CLS attention module, residual feed-forward refinement, and a fused Gaussian bottleneck produce comparable posteriors for the full view and three one-modality-omitted views.
3. **Selective Cross-View Teacher Distillation:** after a full-view teacher warm-up, a margin-gated selector compares candidate posterior-mean prediction errors using **training labels only**. Reliability- and need-aware, role-weighted directional Gaussian KL transfers knowledge to the other views.

During training, SCTD also uses direct full-view and partial-view sentiment supervision, Gaussian prior regularization, and kernel entropy regularization. The teacher selector is disabled at inference.

## Project structure

```text
SCTD/
├── README.md                       # Documentation and reproduction commands
├── main.py                         # SCTD training, evaluation and checkpoints (500 lines)
├── model.py                        # SCTD model and learning objectives (800 lines)
├── global_configs.py               # Feature widths and device defaults
├── requirements.txt                # Python dependencies
├── .gitignore                      # Excludes datasets, BERT weights, outputs
├── modules/
│   ├── __init__.py
│   ├── transformer.py
│   ├── multihead_attention.py
│   └── position_embedding.py
├── scripts/
│   ├── check_dataset.py
│   └── run_five_seeds.py
├── tests/
│   └── test_components.py
├── datasets/
│   ├── README.md
│   ├── mosi.pkl                   # Download separately; not included
│   └── mosei.pkl                  # Download separately; not included
├── bert-base-uncased/              # Download separately; not included
├── checkpoints/                   # Created during training
└── results/                       # Created by multi-seed runner
```

## 1. Environment setup

Python **3.10 or 3.11** is recommended. Install the appropriate PyTorch/CUDA build from [PyTorch](https://pytorch.org/get-started/locally/) first, and install remaining dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

The implementation imports `transformers.models.bert.modeling_bert` classes; the requirements therefore use a compatible **Transformers 4.x** range rather than untested Transformers 5.x APIs. Package version ranges are guidance rather than an exact environment lockfile.

## 2. Download the CMU-MOSI and CMU-MOSEI datasets

The data loader expects **word-aligned** `.pkl` files, with training, validation, and test splits. A public download script for the preprocessing family is available at:

- [Word-aligned dataset download script](https://github.com/WasifurRahman/BERT_multimodal_transformer/blob/master/datasets/download_datasets.sh)
- [CMU-MOSI](https://multicomp.cs.cmu.edu/resources/cmu-mosi-dataset/)
- [CMU-MOSEI](https://multicomp.cs.cmu.edu/resources/cmu-mosei-dataset/)
- [CMU Multimodal Data SDK](https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK)

### Option A: Download the preprocessed pickle datasets

In a Unix-like terminal from the SCTD project root:

```bash
python -m pip install gdown
mkdir -p datasets
gdown "https://drive.google.com/uc?id=12HbavGOtoVCqicvSYWl3zImli5Jz0Nou" -O datasets/mosi.pkl
gdown "https://drive.google.com/uc?id=1VJhSc2TGrPU8zJSVTYwn5kfuG47VaNQ3" -O datasets/mosei.pkl
```

The Google Drive IDs are taken from the dataset download instructions used by the existing CVSG repository; access may change. Do not assume a raw SDK download is interchangeable with the preprocessed pickle files. On Windows PowerShell, use `New-Item -ItemType Directory -Force datasets` to create the directory.

### Expected sample structure

```python
{
    "train": [((words, visual_features, acoustic_features), sentiment_score, sample_id), ...],
    "dev":   [((words, visual_features, acoustic_features), sentiment_score, sample_id), ...],
    "test":  [((words, visual_features, acoustic_features), sentiment_score, sample_id), ...]
}
```

`valid` is accepted in place of `dev`. `words` must be a sequence of word tokens, with audio and visual rows aligned to the **original words** before BERT subword tokenization. Sentiment labels range from -3 to +3. The loader automatically infers audio and visual input dimensions from the dataset.

The `unaligned_50.pkl` / pre-extracted embedding format and MMSA split dictionaries cannot be renamed to `mosi.pkl` or `mosei.pkl`; they require preprocessing.

Validate after downloading:

```bash
python scripts/check_dataset.py datasets/mosi.pkl
python scripts/check_dataset.py datasets/mosei.pkl
```

Never unpickle files from untrusted sources.

## 3. Download pretrained BERT

SCTD uses **bert-base-uncased**. [Official Hugging Face model page](https://huggingface.co/google-bert/bert-base-uncased).

```bash
python -c "from transformers import BertTokenizer, BertModel; repo='google-bert/bert-base-uncased'; BertTokenizer.from_pretrained(repo).save_pretrained('bert-base-uncased'); BertModel.from_pretrained(repo).save_pretrained('bert-base-uncased')"
```

This produces the tokenizer/configuration/weights under `bert-base-uncased/`. The downloaded model files are excluded from the repository and source archive.

## 4. Training

Run from the **SCTD project root** after preparing datasets and pretrained BERT. **The commands below reproduce the manuscript-style 50-epoch protocol** while retaining the source implementation's default loss coefficients. This is a paper-protocol setting; it is not a claim that these exact commands have been rerun in this release environment.

### CMU-MOSI

```bash
python main.py --dataset mosi --gpu 0 --seed 0 --n_epochs 50 --bert_path ./bert-base-uncased --select_metric mse --no_epoch_test
```

### CMU-MOSEI

```bash
python main.py --dataset mosei --gpu 0 --seed 0 --n_epochs 50 --train_batch_size 128 --bert_path ./bert-base-uncased --select_metric mse --no_epoch_test
```

`--gpu 0` selects the first visible GPU. For CPU execution, use `--gpu -1` (significantly slower). The model uses `1e-5` for BERT and `2e-5` for the remaining parameters, with 50-dimensional latent representations.

The warm-up lasts 2 epochs. Then the selector compares the full view and all three leave-one-out views and replaces the full teacher only if a partial view improves the selection score by more than 0.08. The default selection score is **prediction absolute error only** (`--teacher_uncertainty_penalty 0`). Teacher reliability and student need jointly weight posterior KL, and `--full_student_weight 0.25` attenuates partial-to-full distillation. Default loss weights are `--posterior_weight 0.02`, `--partial_weight 0.15`, `--kl_weight 0.001`, and `--renyi_weight 0.0001`.

**Important test-set protocol:** `main.py` can print TEST after every epoch unless `--no_epoch_test` is specified. The reproducibility commands above **always set `--no_epoch_test`**, so checkpoint selection uses only **minimum validation MSE** and the held-out TEST result is printed only after restoring the selected checkpoint. Do not use per-epoch TEST metrics to tune hyperparameters.

## 5. Evaluate a saved checkpoint

```bash
python main.py --dataset mosi --gpu 0 --seed 0 --eval_only
python main.py --dataset mosei --gpu 0 --seed 0 --eval_only
```

A matching checkpoint must already be present under `checkpoints/`. Specify `--checkpoint /absolute/path/to/checkpoint.pth` if needed. The code reports Acc_2, support-weighted F1, Acc_7, MAE, Corr, and MSE. Acc_2 and F1 exclude exactly-zero ground-truth labels by default; `--include_zero` changes that evaluation convention. Acc_2/F1/Acc_7 in CLI output are proportions, e.g. `0.8551` means 85.51%.

## 6. Five independent random seeds

The manuscript summarizes five independent runs. Use a fixed, published seed list when reporting results; the **example list below is not proof of the seed IDs used in the manuscript's reported experiments**.

```bash
python scripts/run_five_seeds.py --dataset mosi --gpu 0 --seeds 0 54 58 101 2334 --n_epochs 50
python scripts/run_five_seeds.py --dataset mosei --gpu 0 --seeds 0 54 58 101 2334 --n_epochs 50 --train_batch_size 128
```

The runner saves per-seed logs under `results/`, uses validation MSE for checkpoint selection, and summarizes Acc_2, F1, Acc_7, Corr, and MAE (mean and sample standard deviation). Dataset and BERT files remain external.

## 7. Key ablation flags

```bash
# Fixed full-view teacher instead of dynamic selection
python main.py --dataset mosi --gpu 0 --seed 0 --n_epochs 50 --no_epoch_test --disable_dynamic_teacher

# Disable posterior distillation
python main.py --dataset mosi --gpu 0 --seed 0 --n_epochs 50 --no_epoch_test --disable_view_consistency

# Disable direct partial-view supervision
python main.py --dataset mosi --gpu 0 --seed 0 --n_epochs 50 --no_epoch_test --disable_partial_supervision

# Disable teacher reliability factor (keeps need and role control)
python main.py --dataset mosi --gpu 0 --seed 0 --n_epochs 50 --no_epoch_test --disable_teacher_reliability

# Try optional uncertainty-augmented teacher scoring (not SCTD default)
python main.py --dataset mosi --gpu 0 --seed 0 --n_epochs 50 --no_epoch_test --teacher_uncertainty_penalty 0.1
```

These commands illustrate supported switches in the refactored core code. Any nondefault experimental variants should be reported with their exact settings and freshly measured results.

## 8. Implementation provenance and limitations

- `main.py` and `model.py`: readability refactor of the previously supplied RNPD-SCTD-26 code. The numerical computations and source-compatible model state keys are retained; the model wrapper class is named `SCTDModel`. Both core files contain no legacy `DIB` identifier or narrative code comments. The internal checkpoint method identifier is retained for compatibility with the earlier release.
- `modules/transformer.py`, `modules/multihead_attention.py`, `modules/position_embedding.py`: recovered external Transformer dependencies previously saved with the user's code; these are not claimed to be byte-identical to the current CVSG GitHub modules.
- `global_configs.py`, `requirements.txt`, validation script, seed runner, tests, `.gitignore`, and this README: release packaging and support files.
- The model is derived from the DIB/RNPD implementation family; do not present it as an independent reimplementation of all upstream backbone components. See the [DIB repository](https://github.com/Taylor-HHT/DIB) and acknowledge upstream modules as appropriate.
- The archive contains **no dataset**, **no BERT weights**, and **no trained checkpoint**; public availability of a source package does not establish reproducibility of the reported scores. Verify the environment and compare multi-seed results before claiming exact numerical reproduction.
- Third-party modules may inherit upstream license and attribution requirements. Review [DIB](https://github.com/Taylor-HHT/DIB) and fairseq-originated component terms before assigning a license to the entire derived repository; no license is automatically asserted here.

## 9. Component smoke test

After installing requirements, check the Gaussian bottleneck, teacher selector, full/partial inference and backward pass without downloading any dataset or BERT weights:

```bash
python tests/test_components.py
```

This validates local tensor operations only. It does not establish that the manuscript's reported benchmark scores can be reproduced.

## 10. Results in the manuscript

| Dataset | Acc-2 (%) | F1 (%) | Acc-7 (%) | Corr | MAE |
|---|---:|---:|---:|---:|---:|
| CMU-MOSI | 85.51 | 85.49 | 46.91 | 0.781 | 0.735 |
| CMU-MOSEI | 86.24 | 86.25 | 54.81 | 0.772 | 0.542 |

These are **manuscript-reported numbers**, not benchmarks independently verified by this source-package preparation. Results depend on the exact datasets, preprocessing, initialization, seed choices, environment, and evaluation protocol.

## Citation

If you use the implementation, please cite the associated manuscript once a public bibliographic record is available. The current title is *SCTD: Selective Cross-View Teacher Distillation for Multimodal Sentiment Analysis*. No DOI is assumed.
