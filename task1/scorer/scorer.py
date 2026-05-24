"""
Official scorer for ArGuard 2026 Task 1 / Track A.

The scorer:

1. Invokes the format checker on the submission file (and exits early if
   the submission is malformed).
2. Computes the official metric (macro-F1) plus diagnostic metrics
   (accuracy, macro-precision, macro-recall, weighted F1, per-class F1).
3. Writes a ``metrics.json`` next to the submission file (unless
   ``--out`` is provided) and prints a human-readable summary.

Both subtasks score predictions against the full set of gold IDs.

- **Subtask A1** — single-label binary classification. Predictions are
  matched by id; ID sets must match.
- **Subtask A2** — multi-label fine-grained categorisation over the
  unified hateful + non-hateful taxonomy. Predictions are matched by id;
  any label in a prediction that is not in the active A2 vocabulary
  (i.e. zero-support taxonomy labels) is ignored.

For local debugging, the gold for A1 can be a TSV (same format as a
submission file) or a JSONL with ``{"id":..., "label":...}``.
The gold for A2 can be a JSONL with ``{"id":..., "labels":[...]}`` —
i.e. the same JSONL format as a submission — or the dataset's split
JSONL with ``{"id":..., "label":..., "fine_grained_label":[...]}``
(produced by ``data/download_data.py``).

Usage
-----
    # Subtask A1
    python scorer/scorer.py --subtask a1 --gold gold_a1.tsv --pred preds_a1.tsv

    # Subtask A2 (gold can be the full dataset jsonl)
    python scorer/scorer.py --subtask a2 \
        --gold data/splits/dev.jsonl --pred preds_a2.jsonl

    # Subtask A2, write metrics to a custom path
    python scorer/scorer.py --subtask a2 \
        --gold data/splits/dev.jsonl --pred preds_a2.jsonl \
        --out reports/dev_a2.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

# Make sibling packages importable.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines.io_utils import (  # noqa: E402
    read_jsonl,
    read_multilabel_jsonl,
    read_subtask_a1_tsv,
)
from baselines.labels import (  # noqa: E402
    BINARY_LABEL2ID,
    BINARY_LABELS,
    FINE_GRAINED_LABEL2ID,
    FINE_GRAINED_LABELS,
    resolve_subtask,
)
from baselines.metrics import (  # noqa: E402
    compute_multilabel_metrics,
    compute_single_label_metrics,
)
from format_checker.format_checker import (  # noqa: E402
    FormatError,
    check_subtask_a1,
    check_subtask_a2,
)

log = logging.getLogger("scorer")


# --------------------------------------------------------------------------
# Gold readers — accept either the official submission shape OR the dataset
# JSONL produced by data/download_data.py.
# --------------------------------------------------------------------------
def read_subtask_a1_gold(path: Path) -> dict[str, str]:
    """Returns mapping id -> binary label."""
    if path.suffix.lower() == ".tsv":
        rows = read_subtask_a1_tsv(path)
        return {r["id"]: r["label"] for r in rows}
    out: dict[str, str] = {}
    for r in read_jsonl(path):
        if "label" not in r:
            raise ValueError(f"{path}: record missing 'label': {r.get('id')}")
        out[str(r["id"])] = r["label"]
    return out


def read_finegrained_gold(path: Path, key: str = "fine_grained_label") -> dict[str, list[str]]:
    """
    Returns mapping id -> list of fine-grained labels.

    Accepts:
    - dataset jsonl with ``{"id":..., "fine_grained_label":[...], "label":...}``,
    - submission-shape jsonl with ``{"id":..., "labels":[...]}``.

    The optional ``key`` selects which field to read for the dataset
    jsonl. The function autodetects between the two shapes.
    """
    out: dict[str, list[str]] = {}
    for r in read_jsonl(path):
        rid = str(r["id"])
        if key in r:
            out[rid] = list(r[key] or [])
        elif "labels" in r:
            out[rid] = list(r["labels"] or [])
        else:
            raise ValueError(
                f"{path}: record {rid!r} missing both {key!r} and 'labels'"
            )
    return out


# --------------------------------------------------------------------------
# Subtask A1
# --------------------------------------------------------------------------
def score_subtask_a1(pred_path: Path, gold_path: Path) -> dict:
    pred_rows = read_subtask_a1_tsv(pred_path)
    gold_map = read_subtask_a1_gold(gold_path)

    pred_ids = {r["id"] for r in pred_rows}
    gold_ids = set(gold_map)
    if pred_ids != gold_ids:
        missing = gold_ids - pred_ids
        extra = pred_ids - gold_ids
        raise ValueError(
            f"ID mismatch: missing={len(missing)} (first 5: {sorted(missing)[:5]}); "
            f"extra={len(extra)} (first 5: {sorted(extra)[:5]})"
        )

    y_true: list[int] = []
    y_pred: list[int] = []
    for r in pred_rows:
        y_true.append(BINARY_LABEL2ID[gold_map[r["id"]]])
        y_pred.append(BINARY_LABEL2ID[r["label"]])
    return compute_single_label_metrics(y_true, y_pred, class_names=list(BINARY_LABELS))


# --------------------------------------------------------------------------
# Subtask A2
# --------------------------------------------------------------------------
def _encode_multilabel(labels: list[str], label2id: dict[str, int], K: int) -> np.ndarray:
    v = np.zeros(K, dtype=int)
    for l in labels:
        if l in label2id:
            v[label2id[l]] = 1
    return v


def score_subtask_a2(pred_path: Path, gold_path: Path) -> dict:
    label2id = dict(FINE_GRAINED_LABEL2ID)
    classes = list(FINE_GRAINED_LABELS)

    pred_map = {r["id"]: r["labels"] for r in read_multilabel_jsonl(pred_path)}
    gold_fg = read_finegrained_gold(gold_path)

    pred_ids = set(pred_map)
    gold_ids = set(gold_fg)
    if pred_ids != gold_ids:
        missing = gold_ids - pred_ids
        extra = pred_ids - gold_ids
        raise ValueError(
            f"ID mismatch: missing={len(missing)} (first 5: {sorted(missing)[:5]}); "
            f"extra={len(extra)} (first 5: {sorted(extra)[:5]})"
        )

    K = len(classes)
    eval_ids = sorted(gold_ids)
    y_true = np.stack(
        [_encode_multilabel(gold_fg[rid], label2id, K) for rid in eval_ids]
    )
    y_pred = np.stack(
        [_encode_multilabel(pred_map[rid], label2id, K) for rid in eval_ids]
    )

    extra = {
        "n_records_evaluated": len(eval_ids),
        "active_vocab": classes,
    }
    m = compute_multilabel_metrics(y_true, y_pred, class_names=classes)
    return {**m, **extra}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _print_summary(subtask: str, metrics: dict) -> None:
    print()
    print(f"=== {subtask}: scoring summary ===")
    print(f"f1_macro          : {metrics['f1_macro']:.4f}   (official metric)")
    if "accuracy" in metrics:
        print(f"accuracy          : {metrics['accuracy']:.4f}")
    print(f"precision_macro   : {metrics['precision_macro']:.4f}")
    print(f"recall_macro      : {metrics['recall_macro']:.4f}")
    print(f"f1_weighted       : {metrics['f1_weighted']:.4f}")
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subtask", required=True,
                    choices=["a1", "a2", "A1", "A2", "subtask_a1", "subtask_a2"])
    ap.add_argument("--predictions", "--pred", required=True, type=Path)
    ap.add_argument("--gold", required=True, type=Path)
    ap.add_argument("--out", default=None, type=Path,
                    help="where to write the metrics.json (default: alongside predictions).")
    ap.add_argument("--skip-format-check", action="store_true",
                    help="do not run the format checker before scoring.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    subtask = resolve_subtask(args.subtask)
    pred_path = args.predictions
    gold_path = args.gold

    if not pred_path.exists():
        sys.stderr.write(f"ERROR predictions file not found: {pred_path}\n")
        return 2
    if not gold_path.exists():
        sys.stderr.write(f"ERROR gold file not found: {gold_path}\n")
        return 2

    if not args.skip_format_check:
        try:
            if subtask == "subtask_a1":
                check_subtask_a1(pred_path)
            else:
                check_subtask_a2(pred_path)
        except FormatError as e:
            sys.stderr.write(f"FORMAT ERROR: {e}\n")
            return 1

    try:
        if subtask == "subtask_a1":
            metrics = score_subtask_a1(pred_path, gold_path)
        else:
            metrics = score_subtask_a2(pred_path, gold_path)
    except ValueError as e:
        sys.stderr.write(f"SCORING ERROR: {e}\n")
        return 1

    out_path = args.out or pred_path.with_suffix(pred_path.suffix + ".metrics.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"subtask": subtask, "metrics": metrics}
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    _print_summary(subtask, metrics)
    print(f"Wrote metrics -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
