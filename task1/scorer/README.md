# Task 1 — Official Scorer

Computes the official metric (**macro-F1**) and a set of diagnostic
metrics for any of the three subtasks. The scorer **invokes the format
checker first** and refuses to score a malformed submission.

```bash
python scorer/scorer.py --subtask 1a \
    --gold data/splits/dev.jsonl \
    --predictions predictions/text_1a.tsv

python scorer/scorer.py --subtask 1b \
    --gold data/splits/dev.jsonl \
    --predictions predictions/text_1b.jsonl

python scorer/scorer.py --subtask 1c \
    --gold data/splits/dev.jsonl \
    --predictions predictions/text_1c.jsonl
```

## Scoring rules

- **Subtask 1A** — single-label binary. The scorer matches predictions
  to gold by `id`, requires the ID sets to be equal, and computes
  accuracy + macro/weighted P/R/F1 + per-class P/R/F1.
- **Subtask 1B** — multi-label. The scorer restricts evaluation to memes
  whose **gold binary label is `Hateful`**. Predictions for non-hateful
  gold IDs are ignored. Missing predictions for in-class IDs are treated
  as the empty label set.
- **Subtask 1C** — same as 1B but for gold binary `Not Hateful`.

## Gold-file formats accepted

Two shapes work for the gold file:

1. The dataset JSONL produced by `data/download_data.py` —
   `{"id":..., "label":..., "fine_grained_label":[...], ...}`. **This is
   required for Subtask 1B and 1C** because the scorer needs the binary
   field to perform the filtering.
2. A submission-shape file (Subtask 1A only) —
   TSV with header `id<TAB>label<TAB>run_id`.

## Output

By default the scorer writes `metrics.json` alongside the predictions
file (e.g. `predictions/text_1a.tsv.metrics.json`). Override with
`--out <path>`.

The JSON has the shape:

```json
{
    "subtask": "subtask_1a",
    "metrics": {
        "f1_macro": 0.83,
        "accuracy": 0.86,
        "precision_macro": 0.84,
        "recall_macro": 0.82,
        "f1_weighted": 0.86,
        "precision__Hateful": 0.78, "recall__Hateful": 0.74, "f1__Hateful": 0.76,
        "precision__Not_Hateful": 0.91, "recall__Not_Hateful": 0.93, "f1__Not_Hateful": 0.92
    }
}
```

For Subtask 1B/1C the metrics block adds multi-label fields
(`subset_accuracy`, `hamming_loss`, `precision_micro`, `recall_micro`,
`f1_micro`, per-class F1 + support).

## Disabling the format check

Use `--skip-format-check` to run the scorer without the format checker
(useful when iterating on the scorer itself).
