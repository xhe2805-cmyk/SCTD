# SCTD: Selective Cross-View Teacher Distillation for Multimodal Sentiment Analysis

SCTD is a multimodal sentiment analysis method for linguistic, acoustic, and visual inputs. Instead of assuming that the full-modality view is always the most reliable teacher, SCTD selects a teacher among the full view and three leave-one-modality-out views **during training** and transfers a Gaussian posterior to the other views. The default inference path uses all three modalities and does not perform teacher selection.

The repository provides model code, training and evaluation scripts, and instructions for preparing the datasets and pretrained language encoder.

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
├── main.py                         # SCTD training, evaluation and checkpoints
├── model.py                        # SCTD model and learning objectives
├── global_configs.py               # Feature widths and device defaults
├── requirements.txt                # Python dependencies
├── .gitignore                      # Excludes datasets, BERT weights, outputs
├── modules/
│   ├── __init__.py
│   ├── transformer.py
│   ├── multihead_attention.py
│   └── position_embedding.py
├── scripts/
│   └── check_dataset.py
├── datasets/
│   ├── README.md
│   ├── mosi.pkl                   # Download separately; not included
│   └── mosei.pkl                  # Download separately; not included
├── bert-base-uncased/              # Download separately; not included
├── checkpoints/                   # Created during training
└── results/                       # Optional training outputs
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