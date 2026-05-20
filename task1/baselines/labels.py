"""
Label vocabularies for ArGuard Task 1 (Multimodal Hateful Meme Detection).

The vocabularies here are the authoritative source consumed by the
baselines, the format checker, and the scorer. They match the labels
shipped in the released dataset (QCRI/ArGuard-Task1 on the Hub).

Three subtasks:

* **Subtask 1A** — Binary single-label: ``Hateful`` / ``Not Hateful``.
* **Subtask 1B** — Multi-label fine-grained *hateful* sub-types
  (8 classes with non-zero training support; 5 additional taxonomy
  classes — Extremism, Historical, Insults, Stereotyping, Threat — have
  zero training support and are excluded from the active label vocab,
  but remain documented for completeness).
* **Subtask 1C** — Multi-label fine-grained *non-hateful* sub-types
  (Humor, Sarcasm, Other).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Mapping


# --- Subtask 1A: binary ---------------------------------------------------
BINARY_LABELS: Final[tuple[str, ...]] = ("Not Hateful", "Hateful")
BINARY_LABEL2ID: Final[Mapping[str, int]] = {k: i for i, k in enumerate(BINARY_LABELS)}
BINARY_ID2LABEL: Final[Mapping[int, str]] = {i: k for k, i in BINARY_LABEL2ID.items()}


# --- Subtask 1B: fine-grained HATEFUL subtypes ----------------------------
# Full taxonomy (13 classes), documented for completeness.
HATEFUL_SUBTYPES_TAXONOMY: Final[tuple[str, ...]] = (
    "Contempt", "Dehumanization", "Exclusion", "Extremism", "Historical",
    "Incitement", "Inferiority", "Insults", "Mocking", "Other", "Slurs",
    "Stereotyping", "Threat",
)
# Active vocabulary (8 classes — those with non-zero support in the dataset).
HATEFUL_SUBTYPES: Final[tuple[str, ...]] = (
    "Contempt",
    "Dehumanization",
    "Exclusion",
    "Incitement",
    "Inferiority",
    "Mocking",
    "Other",
    "Slurs",
)
HATEFUL_SUBTYPES_ZERO_SUPPORT: Final[tuple[str, ...]] = tuple(
    c for c in HATEFUL_SUBTYPES_TAXONOMY if c not in HATEFUL_SUBTYPES
)
HATEFUL_LABEL2ID: Final[Mapping[str, int]] = {k: i for i, k in enumerate(HATEFUL_SUBTYPES)}
HATEFUL_ID2LABEL: Final[Mapping[int, str]] = {i: k for k, i in HATEFUL_LABEL2ID.items()}


# --- Subtask 1C: fine-grained NON-HATEFUL subtypes ------------------------
NONHATEFUL_SUBTYPES: Final[tuple[str, ...]] = ("Humor", "Other", "Sarcasm")
NONHATEFUL_LABEL2ID: Final[Mapping[str, int]] = {k: i for i, k in enumerate(NONHATEFUL_SUBTYPES)}
NONHATEFUL_ID2LABEL: Final[Mapping[int, str]] = {i: k for k, i in NONHATEFUL_LABEL2ID.items()}


@dataclass(frozen=True)
class TaskSpec:
    """Metadata for a single subtask."""

    name: str                      # canonical name: subtask_1a / 1b / 1c
    problem_type: str              # "single_label_classification" | "multi_label_classification"
    classes: tuple[str, ...]
    label2id: Mapping[str, int]
    id2label: Mapping[int, str]
    filter_binary: str | None      # which binary label records to keep; None = keep all

    @property
    def num_labels(self) -> int:
        return len(self.classes)

    @property
    def is_multilabel(self) -> bool:
        return self.problem_type == "multi_label_classification"


TASK_SPECS: Final[dict[str, TaskSpec]] = {
    "subtask_1a": TaskSpec(
        name="subtask_1a",
        problem_type="single_label_classification",
        classes=BINARY_LABELS,
        label2id=BINARY_LABEL2ID,
        id2label=BINARY_ID2LABEL,
        filter_binary=None,
    ),
    "subtask_1b": TaskSpec(
        name="subtask_1b",
        problem_type="multi_label_classification",
        classes=HATEFUL_SUBTYPES,
        label2id=HATEFUL_LABEL2ID,
        id2label=HATEFUL_ID2LABEL,
        filter_binary="Hateful",
    ),
    "subtask_1c": TaskSpec(
        name="subtask_1c",
        problem_type="multi_label_classification",
        classes=NONHATEFUL_SUBTYPES,
        label2id=NONHATEFUL_LABEL2ID,
        id2label=NONHATEFUL_ID2LABEL,
        filter_binary="Not Hateful",
    ),
}


SUBTASK_ALIASES: Final[dict[str, str]] = {
    # short names
    "1a": "subtask_1a", "1A": "subtask_1a",
    "1b": "subtask_1b", "1B": "subtask_1b",
    "1c": "subtask_1c", "1C": "subtask_1c",
    # canonical names
    "subtask_1a": "subtask_1a",
    "subtask_1b": "subtask_1b",
    "subtask_1c": "subtask_1c",
}


def resolve_subtask(name: str) -> str:
    """Normalize any supported alias to the canonical subtask name."""
    if name in SUBTASK_ALIASES:
        return SUBTASK_ALIASES[name]
    raise ValueError(
        f"Unknown subtask '{name}'. Known: {sorted(set(SUBTASK_ALIASES))}"
    )


def get_task(subtask: str) -> TaskSpec:
    return TASK_SPECS[resolve_subtask(subtask)]
