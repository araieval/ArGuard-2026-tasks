"""
Majority-class baseline for all three Task-1 subtasks.

Subtask 1A : predict the majority binary class from the training set
             (= ``Not Hateful`` on the released data).
Subtask 1B : predict the single most frequent hateful sub-type from the
             training set on every hateful meme (multi-label output with
             one entry).
Subtask 1C : predict the single most frequent non-hateful sub-type from
             the training set on every non-hateful meme.

Usage
-----
    python baselines/majority_baseline.py \
        --subtask 1a \
        --train data/splits/train.jsonl \
        --target data/splits/dev_test.jsonl \
        --out predictions/majority_1a.tsv \
        --run-id majority

    python baselines/majority_baseline.py --subtask 1b \
        --train data/splits/train.jsonl --target data/splits/dev_test.jsonl \
        --out predictions/majority_1b.jsonl --run-id majority

    python baselines/majority_baseline.py --subtask 1c \
        --train data/splits/train.jsonl --target data/splits/dev_test.jsonl \
        --out predictions/majority_1c.jsonl --run-id majority
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from io_utils import (  # noqa: E402
    read_jsonl,
    write_multilabel_jsonl,
    write_subtask_1a_tsv,
)
from labels import (  # noqa: E402
    BINARY_LABELS,
    HATEFUL_SUBTYPES,
    NONHATEFUL_SUBTYPES,
    resolve_subtask,
)

log = logging.getLogger("majority")


def _majority_binary(train_records: list[dict]) -> str:
    counts = Counter(r.get("label") for r in train_records if r.get("label") in BINARY_LABELS)
    if not counts:
        raise ValueError("training data has no usable binary labels")
    return counts.most_common(1)[0][0]


def _majority_finegrained(train_records: list[dict], allowed: set[str]) -> str:
    counts: Counter = Counter()
    for r in train_records:
        for l in (r.get("fine_grained_label") or []):
            if l in allowed:
                counts[l] += 1
    if not counts:
        raise ValueError(f"training data has no usable fine-grained labels in {sorted(allowed)}")
    return counts.most_common(1)[0][0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subtask", required=True, choices=["1a", "1b", "1c"])
    ap.add_argument("--train", required=True, type=Path, help="training jsonl")
    ap.add_argument("--target", required=True, type=Path,
                    help="jsonl to predict on (e.g., dev.jsonl or dev_test.jsonl)")
    ap.add_argument("--out", required=True, type=Path, help="output predictions file")
    ap.add_argument("--run-id", default="majority", help="run identifier (Subtask 1A only)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    subtask = resolve_subtask(args.subtask)
    train = read_jsonl(args.train)
    target = read_jsonl(args.target)
    log.info("loaded train=%d  target=%d", len(train), len(target))

    if subtask == "subtask_1a":
        cls = _majority_binary(train)
        log.info("majority class = %r", cls)
        write_subtask_1a_tsv(((r["id"], cls) for r in target), args.out, run_id=args.run_id)
    elif subtask == "subtask_1b":
        cls = _majority_finegrained(train, set(HATEFUL_SUBTYPES))
        log.info("majority hateful subtype = %r", cls)
        write_multilabel_jsonl(((r["id"], [cls]) for r in target), args.out)
    else:  # subtask_1c
        cls = _majority_finegrained(train, set(NONHATEFUL_SUBTYPES))
        log.info("majority non-hateful subtype = %r", cls)
        write_multilabel_jsonl(((r["id"], [cls]) for r in target), args.out)

    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
