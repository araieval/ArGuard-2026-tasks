# Task 1 / Track A — Baselines

Five baselines are provided, in increasing order of strength:

| Script | Family | Subtasks |
|---|---|---|
| `majority_baseline.py`       | Trivial (most-frequent-class)   | A1, A2 |
| `random_baseline.py`         | Trivial (prior-weighted random) | A1, A2 |
| `train_text.py`              | Text-only (BERT-family)         | A1, A2 |
| `train_image.py`             | Image-only (ViT / BEiT)         | A1, A2 |
| `train_multimodal.py`        | Late-fusion (text + image)      | A1, A2 |

All five read JSONL splits produced by `data/download_data.py` and write
predictions in the **exact submission format** described in the task README
— so their outputs can be piped straight into the format checker and the
scorer.

> Throughout this README, paths are relative to the `task1/` directory.

---

## Conventions

### No subset filtering
Both subtasks train, validate, and predict on the **full** splits — there
is no train-time filtering by binary class. A2's label space is the
unified hateful + non-hateful fine-grained taxonomy, and a meme may
legitimately have zero positive labels under A2.

### `--target {dev,dev_test,test}`
Pick which split the trained model writes predictions for. Default is
`dev_test` (the development-phase leaderboard target). Use `dev` for a
local sanity check; use `test` once it is released for the final phase.

### Multi-label heads
- Binary cross-entropy with optional `pos_weight` derived from the training
  set (`pos_weight[k] = (N - n_k) / max(n_k, 1)`).
- Default sigmoid decision threshold is **0.5**; override with `--threshold`.

### Multi-GPU
The HF Trainer-based scripts (`train_text.py`, `train_image.py`) honour
`CUDA_VISIBLE_DEVICES` and the standard `accelerate launch` / `torchrun`
flow. `train_multimodal.py` is a single-GPU script for clarity.

### Reproducibility
All training scripts seed Python, NumPy, and PyTorch with `--seed` (default
`42`). Re-run with the same flags to get bit-identical results on the same
hardware.

---

## Suggested model lists

These are tested on the released data and give competitive starting points:

**Text-only**
- `aubmindlab/bert-base-arabertv02`
- `UBC-NLP/MARBERTv2`
- `CAMeL-Lab/bert-base-arabic-camelbert-mix`
- `xlm-roberta-base`

**Image-only**
- `google/vit-base-patch16-224`
- `microsoft/beit-base-patch16-224`
- `facebook/convnextv2-tiny-22k-224`

**Multimodal (text + image)** — any combination of the above.

For a stronger multimodal baseline (vision-language LoRA fine-tune), see the
ArHate5k research repository.

---

## End-to-end example

```bash
# Download data
python data/download_data.py

# Train text baseline for Subtask A1
python baselines/train_text.py --subtask a1 \
    --model aubmindlab/bert-base-arabertv02 \
    --data-dir data --target dev_test \
    --out predictions/text_a1.tsv --run-id arabert_text

# Validate format
python format_checker/format_checker.py --subtask a1 \
    --predictions predictions/text_a1.tsv

# Local scoring against dev (re-train on dev as target if you want metrics)
python baselines/train_text.py --subtask a1 \
    --model aubmindlab/bert-base-arabertv02 \
    --data-dir data --target dev \
    --out predictions/text_a1_dev.tsv --run-id arabert_text_dev
python scorer/scorer.py --subtask a1 \
    --gold data/splits/dev.jsonl \
    --predictions predictions/text_a1_dev.tsv
```

---

## Notebooks

[`notebooks/`](notebooks/) contains Jupyter notebooks that walk through
each baseline (data loading, training, evaluation, submission writing).

| Notebook | Topic |
|---|---|
| `01_explore_data.ipynb`        | dataset overview, label distributions, image preview |
| `02_text_baseline.ipynb`       | text-only BERT classifier walk-through (Subtask A1) |
| `03_image_baseline.ipynb`      | image-only ViT classifier walk-through (Subtask A1) |
| `04_multimodal_baseline.ipynb` | late-fusion multimodal classifier (Subtask A1) |
| `05_simple_baselines.ipynb`    | majority + random baselines, format checker, scorer |

To extend a notebook from Subtask A1 to A2, change `--subtask` to `a2`
and use a `.jsonl` output file. The training/prediction loops are
otherwise identical because the scripts dispatch on `TaskSpec` from
[`labels.py`](labels.py).

---

## Adding your own baseline

1. Drop a new `train_<family>.py` next to the existing scripts.
2. Reuse [`labels.py`](labels.py), [`io_utils.py`](io_utils.py), and
   [`metrics.py`](metrics.py) — the format-checker and scorer treat any
   file that conforms to the submission schema as valid.
3. Emit predictions with `io_utils.write_subtask_a1_tsv` (for A1) or
   `io_utils.write_multilabel_jsonl` (for A2).

That is everything — there is no model registry to update.
