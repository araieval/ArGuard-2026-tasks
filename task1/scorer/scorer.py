"""
Official scorer for ArGuard Task 1.

The scorer:

1. Invokes the format checker on the submission file (and exits early if
   the submission is malformed).
2. Computes the official metric (macro-F1) plus diagnostic metrics
   (accuracy, macro-precision, macro-recall, weighted F1, per-class F1).
3. Writes a ``metrics.json`` next to the submission file (unless
   ``--out`` is provided) and prints a human-readable summary.

For Subtasks 1B and 1C, the scorer **filters the gold and the
predictions to the gold-positive subset** for the binary parent label.
That is:

- Subtask 1B is scored on memes whose **gold** binary label is
  ``Hateful``. Predictions for non-hateful memes are ignored, missing
  predictions for hateful memes are treated as the empty label set.
- Subtask 1C is the same for ``Not Hateful``.

For local debugging, the gold for 1A can be a TSV (same format as a
submission file) or a JSONL with ``{"id":..., "label":...}``.
The gold for 1B/1C can be a JSONL with ``{"id":..., "labels":[...]}`` —
i.e. the same JSONL format as a submission — or the dataset's split
JSONL with ``{"id":..., "label":..., "fine_grained_label":[...]}``
(produced by ``data/download_data.py``).

Usage
-----
    # Subtask 1A
    python scorer/scorer.py --subtask 1a --gold gold_1a.tsv --pred preds_1a.tsv

    # Subtask 1B (gold can be the full dataset jsonl)
    python scorer/scorer.py --subtask 1b \
        --gold data/splits/dev.jsonl --pred preds_1b.jsonl

    # Subtask 1C, write metrics to a custom path
    python scorer/scorer.py --subtask 1c \
        --gold data/splits/dev.jsonl --pred preds_1c.jsonl \
        --out reports/dev_1c.json
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
    read_subtask_1a_tsv,
)
from baselines.labels import (  # noqa: E402
    BINARY_LABEL2ID,
    BINARY_LABELS,
    HATEFUL_LABEL2ID,
    HATEFUL_SUBTYPES,
    NONHATEFUL_LABEL2ID,
    NONHATEFUL_SUBTYPES,
    resolve_subtask,
)
from baselines.metrics import (  # noqa: E402
    compute_multilabel_metrics,
    compute_single_label_metrics,
)
from format_checker.format_checker import (  # noqa: E402
    FormatError,
    check_multilabel,
    check_subtask_1a,
)

log = logging.getLogger("scorer")


# --------------------------------------------------------------------------
# Gold readers — accept either the official submission shape OR the dataset
# JSONL produced by data/download_data.py.
# --------------------------------------------------------------------------
def read_subtask_1a_gold(path: Path) -> dict[str, str]:
    """Returns mapping id -> binary label."""
    if path.suffix.lower() == ".tsv":
        rows = read_subtask_1a_tsv(path)
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


def read_binary_field_gold(path: Path) -> dict[str, str]:
    """Read the binary 'label' field from a dataset JSONL (no TSV path)."""
    out: dict[str, str] = {}
    for r in read_jsonl(path):
        out[str(r["id"])] = r.get("label")
    return out


# --------------------------------------------------------------------------
# Subtask 1A
# --------------------------------------------------------------------------
def score_subtask_1a(pred_path: Path, gold_path: Path) -> dict:
    pred_rows = read_subtask_1a_tsv(pred_path)
    gold_map = read_subtask_1a_gold(gold_path)

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
# Subtasks 1B / 1C
# --------------------------------------------------------------------------
def _encode_multilabel(labels: list[str], label2id: dict[str, int], K: int) -> np.ndarray:
    v = np.zeros(K, dtype=int)
    for l in labels:
        if l in label2id:
            v[label2id[l]] = 1
    return v


def score_finegrained(
    pred_path: Path,
    gold_path: Path,
    subtask: str,
) -> dict:
    if subtask == "subtask_1b":
        binary_filter = "Hateful"
        label2id = dict(HATEFUL_LABEL2ID)
        classes = list(HATEFUL_SUBTYPES)
    elif subtask == "subtask_1c":
        binary_filter = "Not Hateful"
        label2id = dict(NONHATEFUL_LABEL2ID)
        classes = list(NONHATEFUL_SUBTYPES)
    else:  # pragma: no cover - guarded by caller
        raise ValueError(f"score_finegrained: unsupported subtask {subtask!r}")

    pred_map = {r["id"]: r["labels"] for r in read_multilabel_jsonl(pred_path)}
    gold_fg = read_finegrained_gold(gold_path)
    gold_binary = read_binary_field_gold(gold_path)
    if not gold_binary:
        raise ValueError(
            f"{gold_path}: cannot determine binary labels. The gold file must "
            "be a dataset JSONL with a 'label' field (use the file produced "
            "by data/download_data.py)."
        )

    # Restrict to the gold-positive subset for this subtask
    eval_ids = sorted(rid for rid, b in gold_binary.items() if b == binary_filter)
    if not eval_ids:
        raise ValueError(
            f"{gold_path}: no records with binary label {binary_filter!r}"
        )

    K = len(classes)
    y_true = np.stack(
        [_encode_multilabel(gold_fg.get(rid, []), label2id, K) for rid in eval_ids]
    )
    y_pred = np.stack(
        [_encode_multilabel(pred_map.get(rid, []), label2id, K) for rid in eval_ids]
    )

    extra = {
        "n_records_evaluated": len(eval_ids),
        "n_records_with_prediction": int(sum(rid in pred_map for rid in eval_ids)),
        "n_records_missing_prediction": int(sum(rid not in pred_map for rid in eval_ids)),
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
                    choices=["1a", "1b", "1c", "subtask_1a", "subtask_1b", "subtask_1c"])
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
            if subtask == "subtask_1a":
                check_subtask_1a(pred_path)
            else:
                check_multilabel(pred_path, subtask)
        except FormatError as e:
            sys.stderr.write(f"FORMAT ERROR: {e}\n")
            return 1

    try:
        if subtask == "subtask_1a":
            metrics = score_subtask_1a(pred_path, gold_path)
        else:
            metrics = score_finegrained(pred_path, gold_path, subtask)
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
