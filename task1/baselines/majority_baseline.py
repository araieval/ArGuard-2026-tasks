"""
Majority-class baseline for ArGuard 2026 Task 1 / Track A.

Subtask A1 : predict the majority binary class from the training set
             (= ``Not Hateful`` on the released data).
Subtask A2 : predict the single most frequent fine-grained sub-type
             from the training set on every meme (multi-label output
             with one entry).

Usage
-----
    python baselines/majority_baseline.py \
        --subtask a1 \
        --train data/splits/train.jsonl \
        --target data/splits/dev_test.jsonl \
        --out predictions/majority_a1.tsv \
        --run-id majority

    python baselines/majority_baseline.py --subtask a2 \
        --train data/splits/train.jsonl --target data/splits/dev_test.jsonl \
        --out predictions/majority_a2.jsonl --run-id majority
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
    write_subtask_a1_tsv,
)
from labels import (  # noqa: E402
    BINARY_LABELS,
    FINE_GRAINED_LABELS,
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
    ap.add_argument("--subtask", required=True,
                    choices=["a1", "a2", "A1", "A2", "subtask_a1", "subtask_a2"])
    ap.add_argument("--train", required=True, type=Path, help="training jsonl")
    ap.add_argument("--target", required=True, type=Path,
                    help="jsonl to predict on (e.g., dev.jsonl or dev_test.jsonl)")
    ap.add_argument("--out", required=True, type=Path, help="output predictions file")
    ap.add_argument("--run-id", default="majority", help="run identifier (Subtask A1 only)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    subtask = resolve_subtask(args.subtask)
    train = read_jsonl(args.train)
    target = read_jsonl(args.target)
    log.info("loaded train=%d  target=%d", len(train), len(target))

    if subtask == "subtask_a1":
        cls = _majority_binary(train)
        log.info("majority class = %r", cls)
        write_subtask_a1_tsv(((r["id"], cls) for r in target), args.out, run_id=args.run_id)
    else:  # subtask_a2
        cls = _majority_finegrained(train, set(FINE_GRAINED_LABELS))
        log.info("majority fine-grained subtype = %r", cls)
        write_multilabel_jsonl(((r["id"], [cls]) for r in target), args.out)

    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
