"""
Small I/O helpers used by the baselines, the format checker, and the
scorer. No external dependencies beyond the stdlib so the format checker
and scorer remain lightweight enough to run on a fresh CodaBench worker.
"""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Iterable, Iterator


# --------------------------------------------------------------------------
# JSONL
# --------------------------------------------------------------------------
def read_jsonl(path: str | Path) -> list[dict]:
    """Read a JSONL file into a list of records."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    """Stream a JSONL file as a generator of records."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(records: Iterable[dict], path: str | Path) -> None:
    """Write an iterable of records to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------
# Subtask-A1 TSV (id<TAB>label<TAB>run_id, with header)
# --------------------------------------------------------------------------
SUBTASK_A1_HEADER: tuple[str, ...] = ("id", "label", "run_id")


def write_subtask_a1_tsv(
    predictions: Iterable[tuple[str, str]],
    path: str | Path,
    run_id: str,
) -> None:
    """
    Write a Subtask-A1 submission file.

    Parameters
    ----------
    predictions : iterable of (id, label) pairs.
    path        : destination .tsv path.
    run_id      : single run identifier repeated on every line.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(SUBTASK_A1_HEADER)
        for rid, label in predictions:
            w.writerow([rid, label, run_id])


def read_subtask_a1_tsv(path: str | Path) -> list[dict]:
    """
    Read a Subtask-A1 TSV file. Returns a list of dicts with keys
    ``id``, ``label``, ``run_id``. Raises ``ValueError`` on a bad header.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh, delimiter="\t")
        rows = list(reader)
    if not rows:
        raise ValueError(f"empty file: {path}")
    header = tuple(rows[0])
    if header != SUBTASK_A1_HEADER:
        raise ValueError(
            f"unexpected header in {path}: {header!r}. "
            f"Expected: {SUBTASK_A1_HEADER!r}"
        )
    out: list[dict] = []
    for i, row in enumerate(rows[1:], start=2):
        if len(row) != 3:
            raise ValueError(f"{path}:{i}: expected 3 columns, got {len(row)}: {row!r}")
        rid, label, run_id = row
        out.append({"id": rid, "label": label, "run_id": run_id})
    return out


# --------------------------------------------------------------------------
# Subtask A2 JSONL submissions ({"id":..., "labels":[...]})
# --------------------------------------------------------------------------
def write_multilabel_jsonl(
    predictions: Iterable[tuple[str, list[str]]],
    path: str | Path,
) -> None:
    """Write multi-label predictions as JSONL: ``{"id":..., "labels":[...]}``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rid, labels in predictions:
            fh.write(
                json.dumps({"id": rid, "labels": list(labels)}, ensure_ascii=False) + "\n"
            )


def read_multilabel_jsonl(path: str | Path) -> list[dict]:
    """
    Read a JSONL submission with ``{"id":..., "labels":[...]}`` records.
    Returns a list of dicts with the same keys.
    """
    path = Path(path)
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "id" not in obj or "labels" not in obj:
                raise ValueError(
                    f"{path}:{i}: missing required keys; got {sorted(obj.keys())!r}"
                )
            if not isinstance(obj["labels"], list):
                raise ValueError(f"{path}:{i}: 'labels' must be a list")
            out.append({"id": str(obj["id"]), "labels": list(obj["labels"])})
    return out
