"""
Submission format checker for ArGuard 2026 Task A (Task 1) / Track A.

Usage
-----
    # Subtask A1 — TSV
    python format_checker/format_checker.py \
        --subtask a1 --predictions preds_a1.tsv [--gold gold_a1.tsv]

    # Subtask A2 — JSONL
    python format_checker/format_checker.py \
        --subtask a2 --predictions preds_a2.jsonl [--gold gold_a2.jsonl]

What is checked
---------------
**Subtask A1** (TSV with header ``id<TAB>label<TAB>run_id``):
- File parses; the header is exactly ``id\\tlabel\\trun_id``.
- Each row has 3 columns.
- ``label`` ∈ {"Hateful", "Not Hateful"}.
- ``run_id`` is a single non-empty string and the same on every row.
- IDs are unique. (If ``--gold`` is given: prediction IDs == gold IDs.)

**Subtask A2** (JSONL with ``{"id": str, "labels": [str, ...]}``):
- Each line is valid JSON with required keys ``id`` and ``labels``.
- ``id`` is a non-empty string; IDs are unique.
- ``labels`` is a list of strings drawn from the unified A2 taxonomy
  (hateful sub-types + non-hateful sub-types + shared ``Other``).
  An empty list is allowed (meme has no fine-grained category).
- (If ``--gold`` is given: prediction IDs == gold IDs.)

The checker exits with code 0 if all checks pass, and code 1 otherwise.
A short error report is written to STDERR.
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

# Make the baselines/ package importable so we can reuse labels.py and io_utils.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baselines.io_utils import (  # noqa: E402
    read_multilabel_jsonl,
    read_subtask_a1_tsv,
)
from baselines.labels import (  # noqa: E402
    BINARY_LABELS,
    FINE_GRAINED_LABELS,
    FINE_GRAINED_TAXONOMY,
    resolve_subtask,
)

log = logging.getLogger("format_checker")


class FormatError(Exception):
    """Raised when the submission file fails any structural check."""


# --------------------------------------------------------------------------
# Subtask A1
# --------------------------------------------------------------------------
def check_subtask_a1(pred_path: Path, gold_path: Path | None = None) -> None:
    try:
        rows = read_subtask_a1_tsv(pred_path)
    except ValueError as e:
        raise FormatError(str(e))
    if not rows:
        raise FormatError(f"{pred_path}: no prediction rows")

    valid = set(BINARY_LABELS)
    seen_run_ids = {r["run_id"] for r in rows}
    if len(seen_run_ids) != 1:
        raise FormatError(
            f"{pred_path}: run_id must be the same on every row "
            f"(found {len(seen_run_ids)} distinct values)"
        )
    run_id = next(iter(seen_run_ids))
    if not run_id.strip():
        raise FormatError(f"{pred_path}: run_id is empty")

    bad_labels = [r for r in rows if r["label"] not in valid]
    if bad_labels:
        sample = ", ".join(f"{r['id']}={r['label']!r}" for r in bad_labels[:5])
        raise FormatError(
            f"{pred_path}: {len(bad_labels)} rows have an invalid label "
            f"(expected one of {sorted(valid)}). Examples: {sample}"
        )

    ids = [r["id"] for r in rows]
    counts = Counter(ids)
    dupes = [i for i, c in counts.items() if c > 1]
    if dupes:
        raise FormatError(
            f"{pred_path}: duplicate prediction IDs: {dupes[:5]} "
            f"({len(dupes)} total)"
        )
    if any(not i.strip() for i in ids):
        raise FormatError(f"{pred_path}: at least one row has an empty id")

    if gold_path is not None:
        gold_ids = {r["id"] for r in read_subtask_a1_tsv(gold_path)}
        _check_id_match(set(ids), gold_ids, pred_path, gold_path)

    log.info(
        "[subtask_a1] OK: %d predictions, run_id=%r, labels in %s",
        len(rows), run_id, sorted(valid),
    )


# --------------------------------------------------------------------------
# Subtask A2
# --------------------------------------------------------------------------
def check_subtask_a2(
    pred_path: Path,
    gold_path: Path | None = None,
) -> None:
    valid = set(FINE_GRAINED_TAXONOMY)   # accept full taxonomy (incl. zero-support)
    active = set(FINE_GRAINED_LABELS)    # active labels actually scored

    try:
        rows = read_multilabel_jsonl(pred_path)
    except ValueError as e:
        raise FormatError(str(e))
    if not rows:
        raise FormatError(f"{pred_path}: no prediction rows")

    ids = [r["id"] for r in rows]
    counts = Counter(ids)
    dupes = [i for i, c in counts.items() if c > 1]
    if dupes:
        raise FormatError(
            f"{pred_path}: duplicate prediction IDs: {dupes[:5]} "
            f"({len(dupes)} total)"
        )
    if any(not i.strip() for i in ids):
        raise FormatError(f"{pred_path}: at least one row has an empty id")

    bad: list[tuple[str, list[str]]] = []
    inactive_used: Counter = Counter()
    for r in rows:
        if not isinstance(r["labels"], list):
            raise FormatError(f"{pred_path}: id={r['id']!r}: 'labels' must be a list")
        bad_lbls = [l for l in r["labels"] if l not in valid]
        if bad_lbls:
            bad.append((r["id"], bad_lbls))
        for l in r["labels"]:
            if l in valid and l not in active:
                inactive_used[l] += 1
    if bad:
        sample = ", ".join(f"{i}={ls}" for i, ls in bad[:3])
        raise FormatError(
            f"{pred_path}: {len(bad)} records contain invalid labels "
            f"(allowed: {sorted(valid)}). Examples: {sample}"
        )
    if inactive_used:
        log.warning(
            "[subtask_a2] %d records use zero-support taxonomy labels: %s "
            "(accepted by the checker but ignored by the scorer)",
            sum(inactive_used.values()), dict(inactive_used),
        )

    if gold_path is not None:
        gold_ids = {r["id"] for r in read_multilabel_jsonl(gold_path)}
        _check_id_match(set(ids), gold_ids, pred_path, gold_path)

    log.info(
        "[subtask_a2] OK: %d predictions, active vocab=%d, full taxonomy=%d",
        len(rows), len(active), len(valid),
    )


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _check_id_match(
    pred_ids: set[str],
    gold_ids: set[str],
    pred_path: Path,
    gold_path: Path,
) -> None:
    missing = gold_ids - pred_ids
    extra = pred_ids - gold_ids
    if missing:
        raise FormatError(
            f"{pred_path}: missing predictions for {len(missing)} gold IDs "
            f"(first 5: {sorted(missing)[:5]})"
        )
    if extra:
        raise FormatError(
            f"{pred_path}: {len(extra)} prediction IDs are not in {gold_path} "
            f"(first 5: {sorted(extra)[:5]})"
        )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subtask", required=True,
                    choices=["a1", "a2", "A1", "A2", "subtask_a1", "subtask_a2"])
    ap.add_argument("--predictions", "--pred", required=True, type=Path,
                    help="path to the submission file")
    ap.add_argument("--gold", default=None, type=Path,
                    help="(optional) gold file — when given, the checker also "
                         "verifies that the prediction ID set equals the gold ID set.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    subtask = resolve_subtask(args.subtask)
    pred_path = args.predictions
    gold_path = args.gold

    if not pred_path.exists():
        sys.stderr.write(f"ERROR predictions file not found: {pred_path}\n")
        return 2

    try:
        if subtask == "subtask_a1":
            check_subtask_a1(pred_path, gold_path)
        else:
            check_subtask_a2(pred_path, gold_path)
    except FormatError as e:
        sys.stderr.write(f"FORMAT ERROR: {e}\n")
        return 1

    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
