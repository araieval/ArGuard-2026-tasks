# Task A (Task 1) / Track A — Official Scorer

Computes the official metric (**macro-F1**) and a set of diagnostic
metrics for both subtasks. The scorer **invokes the format checker
first** and refuses to score a malformed submission.

```bash
python scorer/scorer.py --subtask a1 \
    --gold data/splits/dev.jsonl \
    --predictions predictions/text_a1.tsv

python scorer/scorer.py --subtask a2 \
    --gold data/splits/dev.jsonl \
    --predictions predictions/text_a2.jsonl
```

## Scoring rules

- **Subtask A1** — single-label binary. The scorer matches predictions
  to gold by `id`, requires the ID sets to be equal, and computes
  accuracy + macro/weighted P/R/F1 + per-class P/R/F1.
- **Subtask A2** — multi-label fine-grained over the unified
  hateful + non-hateful taxonomy. The scorer matches predictions to
  gold by `id` and requires the ID sets to be equal. Any label in a
  prediction that is not in the active vocabulary (i.e. zero-support
  taxonomy labels) is silently ignored.

Both subtasks are scored on **every gold record** — no subset filtering.

## Gold-file formats accepted

Two shapes work for the gold file:

1. The dataset JSONL produced by `data/download_data.py` —
   `{"id":..., "label":..., "fine_grained_label":[...], ...}`. **This is
   required for Subtask A2** because that is where fine-grained gold
   lives.
2. A submission-shape file (Subtask A1: TSV with header
   `id<TAB>label<TAB>run_id`; Subtask A2: JSONL with
   `{"id":..., "labels":[...]}`).

## Output

By default the scorer writes `metrics.json` alongside the predictions
file (e.g. `predictions/text_a1.tsv.metrics.json`). Override with
`--out <path>`.

The JSON has the shape:

```json
{
    "subtask": "subtask_a1",
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

For Subtask A2 the metrics block adds multi-label fields
(`subset_accuracy`, `hamming_loss`, `precision_micro`, `recall_micro`,
`f1_micro`, per-class F1 + support).

## Disabling the format check

Use `--skip-format-check` to run the scorer without the format checker
(useful when iterating on the scorer itself).
