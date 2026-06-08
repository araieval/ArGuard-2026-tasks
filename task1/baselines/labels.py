"""
Label vocabularies for ArGuard 2026 Task 1 / Track A
(Multimodal Hateful Meme Detection).

Two subtasks:

* **Subtask A1** — Binary single-label: ``Hateful`` / ``Not Hateful``.
* **Subtask A2** — Multi-label fine-grained categorisation over **all**
  memes. The label space unifies hateful and non-hateful subtypes:
    - Hateful side (active in the released data):
      Contempt, Dehumanization, Exclusion, Incitement, Inferiority,
      Mocking, Slurs.
    - Non-hateful side: Humor, Sarcasm.
    - Shared across both: Other.
  The full annotation taxonomy additionally includes 5 hateful classes
  with zero training support (Extremism, Historical, Insults,
  Stereotyping, Threat); these are documented for completeness but are
  excluded from the active A2 vocabulary.

The vocabularies here are the authoritative source consumed by the
baselines, the format checker, and the scorer. They match the labels
shipped in the released dataset (QCRI/ArGuard-Task1 on the Hub).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Mapping


# --- Subtask A1: binary ---------------------------------------------------
BINARY_LABELS: Final[tuple[str, ...]] = ("Not Hateful", "Hateful")
BINARY_LABEL2ID: Final[Mapping[str, int]] = {k: i for i, k in enumerate(BINARY_LABELS)}
BINARY_ID2LABEL: Final[Mapping[int, str]] = {i: k for k, i in BINARY_LABEL2ID.items()}


# --- A2 hateful taxonomy --------------------------------------------------
# Full annotation taxonomy on the hateful side (13 classes); 5 have zero
# training support and are excluded from the active A2 vocabulary.
HATEFUL_SUBTYPES_TAXONOMY: Final[tuple[str, ...]] = (
    "Contempt", "Dehumanization", "Exclusion", "Extremism", "Historical",
    "Incitement", "Inferiority", "Insults", "Mocking", "Other", "Slurs",
    "Stereotyping", "Threat",
)
HATEFUL_SUBTYPES_ACTIVE: Final[tuple[str, ...]] = (
    "Contempt", "Dehumanization", "Exclusion", "Incitement", "Inferiority",
    "Mocking", "Other", "Slurs",
)
HATEFUL_SUBTYPES_ZERO_SUPPORT: Final[tuple[str, ...]] = tuple(
    c for c in HATEFUL_SUBTYPES_TAXONOMY if c not in HATEFUL_SUBTYPES_ACTIVE
)

# --- A2 non-hateful taxonomy ---------------------------------------------
NONHATEFUL_SUBTYPES: Final[tuple[str, ...]] = ("Humor", "Other", "Sarcasm")


# --- Subtask A2: unified multi-label vocab -------------------------------
# Active label space across both hateful and non-hateful memes, sorted
# alphabetically for stable indexing. ``Other`` is shared between the two
# annotation groups.
FINE_GRAINED_LABELS: Final[tuple[str, ...]] = tuple(sorted(
    set(HATEFUL_SUBTYPES_ACTIVE) | set(NONHATEFUL_SUBTYPES)
))
FINE_GRAINED_LABEL2ID: Final[Mapping[str, int]] = {
    k: i for i, k in enumerate(FINE_GRAINED_LABELS)
}
FINE_GRAINED_ID2LABEL: Final[Mapping[int, str]] = {
    i: k for k, i in FINE_GRAINED_LABEL2ID.items()
}

# Full A2 taxonomy (active labels + zero-support hateful labels). The
# format checker accepts predictions from this set but the scorer ignores
# any labels that are not in the active vocabulary.
FINE_GRAINED_TAXONOMY: Final[tuple[str, ...]] = tuple(sorted(
    set(FINE_GRAINED_LABELS) | set(HATEFUL_SUBTYPES_TAXONOMY)
))


@dataclass(frozen=True)
class TaskSpec:
    """Metadata for a single subtask."""

    name: str                      # canonical name: subtask_a1 / subtask_a2
    problem_type: str              # "single_label_classification" | "multi_label_classification"
    classes: tuple[str, ...]
    label2id: Mapping[str, int]
    id2label: Mapping[int, str]

    @property
    def num_labels(self) -> int:
        return len(self.classes)

    @property
    def is_multilabel(self) -> bool:
        return self.problem_type == "multi_label_classification"


TASK_SPECS: Final[dict[str, TaskSpec]] = {
    "subtask_a1": TaskSpec(
        name="subtask_a1",
        problem_type="single_label_classification",
        classes=BINARY_LABELS,
        label2id=BINARY_LABEL2ID,
        id2label=BINARY_ID2LABEL,
    ),
    "subtask_a2": TaskSpec(
        name="subtask_a2",
        problem_type="multi_label_classification",
        classes=FINE_GRAINED_LABELS,
        label2id=FINE_GRAINED_LABEL2ID,
        id2label=FINE_GRAINED_ID2LABEL,
    ),
}


SUBTASK_ALIASES: Final[dict[str, str]] = {
    "a1": "subtask_a1", "A1": "subtask_a1",
    "a2": "subtask_a2", "A2": "subtask_a2",
    "subtask_a1": "subtask_a1",
    "subtask_a2": "subtask_a2",
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
