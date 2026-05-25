# Task 1 — Track A: Multimodal Hateful Meme Detection (ArGuard 2026)

Starter kit for **Task 1 / Track A** of the [ArGuard 2026 shared task][website]:
detecting hateful content in Arabic memes.

[website]: https://araieval.gitlab.io/ArGuard2026/

__Table of contents:__
- [Subtasks](#subtasks)
- [Repository contents](#repository-contents)
- [Setup](#setup)
- [Data](#data)
- [Baselines](#baselines)
- [Reference numbers](#reference-numbers)
- [Format checker](#format-checker)
- [Scorer](#scorer)
- [Submission](#submission)
- [Notebooks](#notebooks)
- [Licensing](#licensing)
- [Citation and contact](#citation-and-contact)

---

## Subtasks

Given an Arabic meme (image + OCR-extracted text), Track A has **two subtasks**:

| Subtask | Type | Label space | Evaluated on |
|---|---|---|---|
| **A1** | Binary, single-label | `Hateful`, `Not Hateful` | all memes |
| **A2** | Multi-label, fine-grained | unified 10-class vocab<sup>†</sup> | all memes |

<sup>†</sup> A2 uses one unified label space across both groups: hateful sub-types (`Contempt`, `Dehumanization`, `Exclusion`, `Incitement`, `Inferiority`, `Mocking`, `Slurs`), non-hateful sub-types (`Humor`, `Sarcasm`), plus `Other` (shared between the two groups). The full annotation taxonomy additionally includes 5 hateful classes with zero training support (`Extremism`, `Historical`, `Insults`, `Stereotyping`, `Threat`) — the format checker accepts predictions from these, but they are not scored.

**Official metric (both subtasks): macro-F1.** Accuracy (A1) / subset-accuracy (A2), macro-precision, macro-recall, weighted F1 and per-class F1 are also reported.

Both subtasks are scored against the **full set of gold IDs** — no subset filtering.

---

## Repository contents

```
task1/
├── README.md                       # this file
├── requirements.txt
├── data/
│   ├── README.md
│   ├── download_data.py            # download from QCRI/ArGuard-Task1 (HF Hub)
│   └── sample_submissions/         # one example file per subtask
├── baselines/
│   ├── README.md
│   ├── labels.py                   # shared label vocabularies
│   ├── io_utils.py                 # JSONL / TSV helpers
│   ├── metrics.py                  # shared metric utilities
│   ├── majority_baseline.py        # most-frequent-class baseline
│   ├── random_baseline.py          # prior-weighted random baseline
│   ├── train_text.py               # text-only BERT classifier
│   ├── train_image.py              # image-only ViT/BEiT classifier
│   ├── train_multimodal.py         # text + image late-fusion classifier
│   └── notebooks/                  # Jupyter walkthroughs for each baseline
├── format_checker/
│   ├── README.md
│   └── format_checker.py
└── scorer/
    ├── README.md
    └── scorer.py
```

---

## Setup

```bash
git clone https://github.com/araieval/ArGuard-2026-tasks.git
cd ArGuard-2026-tasks/task1
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Pick the PyTorch wheel that matches your CUDA version, e.g.
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

If the dataset is gated/private on the Hub during the development phase, authenticate first:

```bash
huggingface-cli login        # or: export HF_TOKEN=<your token>
```

---

## Data

The official splits live at **[QCRI/ArGuard-Task1](https://huggingface.co/datasets/QCRI/ArGuard-Task1)** on the Hugging Face Hub. Download them with:

```bash
python data/download_data.py
```

This writes (under `task1/data/`):

```
data/
├── splits/
│   ├── train.jsonl       (3,500 records, full labels)
│   ├── dev.jsonl         (500 records,   full labels)
│   ├── dev_test.jsonl    (500 records,   no labels — leaderboard target)
│   └── test.jsonl        (released for the final-evaluation phase)
└── img/
    └── <id>              # one image file per meme
```

Each JSONL record:

```json
{
    "id": "f9a8…b1.jpg",
    "image_path": "img/f9a8…b1.jpg",
    "text": "OCR-extracted Arabic text…",
    "label": "Hateful" | "Not Hateful",
    "fine_grained_label": ["Mocking", "Incitement"],
    "annotations": []
}
```

See [`data/README.md`](data/README.md) for details and download options.

---

## Baselines

Five baselines are provided. All read the JSONL splits written by `download_data.py` and emit predictions in the **exact submission format** (see [Submission](#submission)).

```bash
# 1) Most-frequent-class baseline
python baselines/majority_baseline.py --subtask a1 \
    --train data/splits/train.jsonl --target data/splits/dev_test.jsonl \
    --out predictions/majority_a1.tsv --run-id majority

# 2) Prior-weighted random baseline
python baselines/random_baseline.py --subtask a1 \
    --train data/splits/train.jsonl --target data/splits/dev_test.jsonl \
    --out predictions/random_a1.tsv --run-id random --seed 42

# 3) Text-only BERT classifier
python baselines/train_text.py --subtask a1 \
    --model aubmindlab/bert-base-arabertv02 \
    --data-dir data --target dev_test \
    --out predictions/text_a1.tsv --run-id arabert_text

# 4) Image-only ViT classifier
python baselines/train_image.py --subtask a1 \
    --model google/vit-base-patch16-224 \
    --data-dir data --target dev_test \
    --out predictions/image_a1.tsv --run-id vit_image

# 5) Multimodal late-fusion (text + image, end-to-end fine-tune)
python baselines/train_multimodal.py --subtask a1 \
    --text-model aubmindlab/bert-base-arabertv02 \
    --image-model google/vit-base-patch16-224 \
    --data-dir data --target dev_test \
    --out predictions/mm_a1.tsv --run-id arabert_vit
```

Swap `--subtask a1` for `a2` (and use a `.jsonl` `--out`) for the fine-grained subtask. The training scripts train on **all** train/dev records for both subtasks — no subset filtering.

For a stronger VLM-based multimodal baseline (Qwen-VL LoRA fine-tune via `ms-swift`), see the research repository — out of scope for this starter kit. See [`baselines/README.md`](baselines/README.md) for hyperparameters and recommended model lists.

---

## Reference numbers

End-to-end smoke run on `dev.jsonl` (500 records, fully labelled). These are
intentionally short runs — **1 epoch, batch size 16, max sequence length 128,
seed 42** — meant to confirm the pipeline runs end-to-end and to give a lower
bound on what a participant should be able to beat with a few hours of tuning.
They are **not** strong baselines. Numbers are **macro-F1** (the official metric).

| Baseline | Backbone(s) | A1 (binary) | A2 (multi-label fine-grained) |
|---|---|---:|---:|
| Majority   | —                                                               | 0.3835 | 0.0403 |
| Random     | training-set priors (seed 42)                                   |  —     | 0.0992 |
| Text       | `aubmindlab/bert-base-arabertv02`                               | 0.6197 | 0.2476 |
| Image      | `google/vit-base-patch16-224`                                   | 0.6961 | 0.1945 |
| Multimodal | `aubmindlab/bert-base-arabertv02` + `google/vit-base-patch16-224` (late fusion) | 0.6645 | 0.2796 |

Reproducer: [`slurm/test_arguard_starter_kit.sh`](../slurm/test_arguard_starter_kit.sh)
in the research repo, or run the commands in [Baselines](#baselines) yourself
with `--target dev` and the same flags.

> Bump `--epochs`, tune `--lr` / `--threshold`, try a stronger backbone, or
> swap in a VLM via `ms-swift` to push these numbers up.

---

## Format checker

Before submitting, validate the file format:

```bash
python format_checker/format_checker.py --subtask a1 --predictions predictions/text_a1.tsv
python format_checker/format_checker.py --subtask a2 --predictions predictions/text_a2.jsonl
```

The checker validates header / schema, label vocabulary, ID uniqueness, and (when a `--gold` file is supplied) that the prediction IDs match the gold IDs. See [`format_checker/README.md`](format_checker/README.md).

---

## Scorer

```bash
# Subtask A1 — gold can be the dataset JSONL or a TSV in submission format
python scorer/scorer.py --subtask a1 \
    --gold data/splits/dev.jsonl \
    --predictions predictions/text_a1.tsv

# Subtask A2 — gold is the dataset JSONL (or any submission-shape jsonl)
python scorer/scorer.py --subtask a2 \
    --gold data/splits/dev.jsonl \
    --predictions predictions/text_a2.jsonl
```

The scorer prints a summary and writes a `metrics.json` next to the predictions file. See [`scorer/README.md`](scorer/README.md).

> During the **development phase**, you submit predictions on the `dev_test` split. Gold labels for `dev_test` are not released — the scorer described above can only be run locally against `dev.jsonl`. The official `dev_test` leaderboard runs the same scorer on the organisers' side.

---

## Submission

### File formats

**Subtask A1** — TSV with header `id<TAB>label<TAB>run_id`:

```
id	label	run_id
abc123.jpg	Hateful	my_team_run1
def456.jpg	Not Hateful	my_team_run1
```

**Subtask A2** — JSONL (multi-label, unified fine-grained taxonomy):

```
{"id": "abc123.jpg", "labels": ["Mocking", "Incitement"]}
{"id": "def456.jpg", "labels": ["Humor", "Sarcasm"]}
{"id": "ghi789.jpg", "labels": []}
```

Sample files in [`data/sample_submissions/`](data/sample_submissions/).

### Submission site

The official submission site (CodaBench) will be announced on the [task website][website].

### Guidelines

1. **Development phase** — build and tune your system on `train.jsonl` and `dev.jsonl`. Submit predictions on **`dev_test.jsonl`** to the leaderboard for progress tracking.
2. **Final-evaluation phase** — submit predictions on the blind **`test.jsonl`** for the official ranking.

For each phase:
- Each team should maintain a single submission account.
- The most recent valid submission before the deadline will be considered the final submission.
- Include your team name and a short method description with each submission.

---

## Notebooks

Friendly walk-throughs of the scripts above live in [`baselines/notebooks/`](baselines/notebooks/):

| Notebook | Topic |
|---|---|
| `01_explore_data.ipynb` | load splits, browse memes, plot label distributions |
| `02_text_baseline.ipynb` | text-only BERT classifier on Subtask A1 |
| `03_image_baseline.ipynb` | image-only ViT classifier on Subtask A1 |
| `04_multimodal_baseline.ipynb` | late-fusion multimodal classifier |
| `05_simple_baselines.ipynb` | majority + random baselines, format checker, scorer |

Launch with:

```bash
pip install jupyter
jupyter notebook baselines/notebooks/
```

---

## Licensing

The released dataset is distributed under **CC BY-NC 4.0** (non-commercial research use). See the dataset page on the Hub for details. The starter-kit code in this directory is released for research and evaluation use within the shared task.

---

## Citation and contact

- **Website:** https://araieval.gitlab.io/ArGuard2026/
- **Email:** arguard2026-organizers@googlegroups.com
