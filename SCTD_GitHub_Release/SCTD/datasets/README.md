# Data

Place the **word-aligned** CMU-MOSI and CMU-MOSEI datasets here:

```text
datasets/
├── mosi.pkl
└── mosei.pkl
```

Public dataset script: https://github.com/WasifurRahman/BERT_multimodal_transformer/blob/master/datasets/download_datasets.sh

The expected sample structure is `((words, visual_features, acoustic_features), sentiment_score, sample_id)`, grouped under `train`, `dev` (or `valid`) and `test`. Inputs must be word-aligned *before* BERT tokenization. These are not raw CMU SDK archives, MMSA dictionaries, or the `unaligned_50.pkl` representation.

Validate with:

```bash
python scripts/check_dataset.py datasets/mosi.pkl
python scripts/check_dataset.py datasets/mosei.pkl
```

Dataset files are not distributed here. Only unpickle data from trusted sources.
