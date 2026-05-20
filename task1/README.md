# Task 1: Multimodal Hateful Meme Detection (ArGuard)

Starter kit for **Task 1** of the [ArGuard 2026 shared task][website]:
detecting hateful content in Arabic memes.

[website]: https://araieval.gitlab.io/ArGuard2026/

__Table of contents:__
- [Subtasks](#subtasks)
- [Repository contents](#repository-contents)
- [Setup](#setup)
- [Data](#data)
- [Baselines](#baselines)
- [Format checker](#format-checker)
- [Scorer](#scorer)
- [Submission](#submission)
- [Notebooks](#notebooks)
- [Licensing](#licensing)
- [Citation and contact](#citation-and-contact)

---

## Subtasks

Given an Arabic meme (image + OCR-extracted text), Task 1 has **three hierarchical subtasks**:

| Subtask | Type | Label space | Evaluated on |
|---|---|---|---|
| **1A** | Binary, single-label | `Hateful`, `Not Hateful` | all memes |
| **1B** | Multi-label, fine-grained hateful | 13-class taxonomy<sup>†</sup> | memes with gold binary = `Hateful` |
| **1C** | Multi-label, fine-grained non-hateful | `Humor`, `Sarcasm`, `Other` | memes with gold binary = `Not Hateful` |

<sup>†</sup> The full hateful taxonomy is: `Contempt`, `Dehumanization`, `Exclusion`, `Extremism`, `Historical`, `Incitement`, `Inferiority`, `Insults`, `Mocking`, `Other`, `Slurs`, `Stereotyping`, `Threat`. Eight of these (Contempt, Dehumanization, Exclusion, Incitement, Inferiority, Mocking, Other, Slurs) have non-zero training support — the format checker accepts predictions from all thirteen.

**Official metric (all subtasks): macro-F1.** Accuracy, macro-precision, macro-recall, weighted F1 and per-class F1 are also reported.

For Subtasks 1B and 1C the scorer **filters by the gold binary label**: predictions for memes outside the relevant binary class are ignored, and missing predictions for in-class memes are treated as the empty label set.

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
    "label": "Hateful" | "Not Hateful" | null,
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
python baselines/majority_baseline.py --subtask 1a \
    --train data/splits/train.jsonl --target data/splits/dev_test.jsonl \
    --out predictions/majority_1a.tsv --run-id majority

# 2) Prior-weighted random baseline
python baselines/random_baseline.py --subtask 1a \
    --train data/splits/train.jsonl --target data/splits/dev_test.jsonl \
    --out predictions/random_1a.tsv --run-id random --seed 42

# 3) Text-only BERT classifier
python baselines/train_text.py --subtask 1a \
    --model aubmindlab/bert-base-arabertv02 \
    --data-dir data --target dev_test \
    --out predictions/text_1a.tsv --run-id arabert_text

# 4) Image-only ViT classifier
python baselines/train_image.py --subtask 1a \
    --model google/vit-base-patch16-224 \
    --data-dir data --target dev_test \
    --out predictions/image_1a.tsv --run-id vit_image

# 5) Multimodal late-fusion (text + image, end-to-end fine-tune)
python baselines/train_multimodal.py --subtask 1a \
    --text-model aubmindlab/bert-base-arabertv02 \
    --image-model google/vit-base-patch16-224 \
    --data-dir data --target dev_test \
    --out predictions/mm_1a.tsv --run-id arabert_vit
```

Swap `--subtask 1a` for `1b` or `1c` (and use a `.jsonl` `--out`) for the fine-grained subtasks. The training scripts automatically filter the train and dev splits to the relevant binary subset for 1B/1C, but they **predict on the full target split** so the resulting submission can be scored directly.

For a stronger VLM-based multimodal baseline (Qwen-VL LoRA fine-tune via `ms-swift`), see the research repository — out of scope for this starter kit. See [`baselines/README.md`](baselines/README.md) for hyperparameters and recommended model lists.

---

## Format checker

Before submitting, validate the file format:

```bash
python format_checker/format_checker.py --subtask 1a --predictions predictions/text_1a.tsv
python format_checker/format_checker.py --subtask 1b --predictions predictions/text_1b.jsonl
python format_checker/format_checker.py --subtask 1c --predictions predictions/text_1c.jsonl
```

The checker validates header / schema, label vocabulary, ID uniqueness, and (when a `--gold` file is supplied) that the prediction IDs match the gold IDs. See [`format_checker/README.md`](format_checker/README.md).

---

## Scorer

```bash
# Subtask 1A — gold can be the dataset JSONL or a TSV in submission format
python scorer/scorer.py --subtask 1a \
    --gold data/splits/dev.jsonl \
    --predictions predictions/text_1a.tsv

# Subtask 1B / 1C — gold must be the dataset JSONL (so the scorer knows
# which records to filter to the relevant binary subset)
python scorer/scorer.py --subtask 1b \
    --gold data/splits/dev.jsonl \
    --predictions predictions/text_1b.jsonl
```

The scorer prints a summary and writes a `metrics.json` next to the predictions file. See [`scorer/README.md`](scorer/README.md).

> During the **development phase**, you submit predictions on the `dev_test` split. Gold labels for `dev_test` are not released — the scorer described above can only be run locally against `dev.jsonl`. The official `dev_test` leaderboard runs the same scorer on the organisers' side.

---

## Submission

### File formats

**Subtask 1A** — TSV with header `id<TAB>label<TAB>run_id`:

```
id	label	run_id
abc123.jpg	Hateful	my_team_run1
def456.jpg	Not Hateful	my_team_run1
```

**Subtask 1B** — JSONL (multi-label, hateful sub-types):

```
{"id": "abc123.jpg", "labels": ["Mocking", "Incitement"]}
{"id": "def456.jpg", "labels": []}
```

**Subtask 1C** — JSONL (multi-label, non-hateful sub-types):

```
{"id": "abc123.jpg", "labels": ["Humor"]}
{"id": "def456.jpg", "labels": ["Sarcasm", "Other"]}
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
| `02_text_baseline.ipynb` | text-only BERT classifier on subtasks 1A/1B/1C |
| `03_image_baseline.ipynb` | image-only ViT classifier |
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
- **Citation:** an overview paper will be released alongside the shared task. A provisional BibTeX entry is provided in [`../bibtex/bibliography.bib`](../bibtex/bibliography.bib).
