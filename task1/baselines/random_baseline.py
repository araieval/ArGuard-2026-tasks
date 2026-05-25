"""
Random baseline for ArGuard 2026 Task 1 / Track A.

Subtask A1 : Bernoulli draw with the training-set hateful prior, returning
             ``Hateful`` or ``Not Hateful``.
Subtask A2 : multi-label draw; each fine-grained sub-type is sampled
             independently with its marginal training-set frequency
             across all memes.

Usage
-----
    python baselines/random_baseline.py --subtask a1 \
        --train data/splits/train.jsonl --target data/splits/dev_test.jsonl \
        --out predictions/random_a1.tsv --run-id random --seed 42
"""
from __future__ import annotations

import argparse
import logging
import random
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

log = logging.getLogger("random")


def _binary_prior(train: list[dict]) -> float:
    counts = Counter(r.get("label") for r in train if r.get("label") in BINARY_LABELS)
    total = sum(counts.values())
    if total == 0:
        return 0.5
    return counts.get("Hateful", 0) / total


def _finegrained_priors(train: list[dict], allowed: tuple[str, ...]) -> dict[str, float]:
    if not train:
        return {a: 0.0 for a in allowed}
    counts = Counter()
    for r in train:
        for l in (r.get("fine_grained_label") or []):
            if l in allowed:
                counts[l] += 1
    n = len(train)
    return {a: counts.get(a, 0) / n for a in allowed}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subtask", required=True,
                    choices=["a1", "a2", "A1", "A2", "subtask_a1", "subtask_a2"])
    ap.add_argument("--train", required=True, type=Path)
    ap.add_argument("--target", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--run-id", default="random")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    rng = random.Random(args.seed)
    subtask = resolve_subtask(args.subtask)
    train = read_jsonl(args.train)
    target = read_jsonl(args.target)
    log.info("loaded train=%d  target=%d", len(train), len(target))

    if subtask == "subtask_a1":
        p_hateful = _binary_prior(train)
        log.info("p(Hateful) = %.4f", p_hateful)
        rows = (
            (r["id"], "Hateful" if rng.random() < p_hateful else "Not Hateful")
            for r in target
        )
        write_subtask_a1_tsv(rows, args.out, run_id=args.run_id)
    else:  # subtask_a2
        priors = _finegrained_priors(train, FINE_GRAINED_LABELS)
        log.info("fine-grained priors = %s", {k: round(v, 3) for k, v in priors.items()})
        rows = []
        for r in target:
            ls = [l for l, p in priors.items() if rng.random() < p]
            rows.append((r["id"], ls))
        write_multilabel_jsonl(rows, args.out)

    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
