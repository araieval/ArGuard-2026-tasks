"""
Metrics module shared by baselines and the scorer.

* ``compute_single_label_metrics`` — Subtask A1 (binary).
* ``compute_multilabel_metrics``    — Subtask A2 (multi-label fine-grained).

Both functions return a flat ``dict[str, float | int]`` so the scorer
can dump it straight to JSON. The official ranking metric for every
subtask is **macro-F1** (``f1_macro``).

The sklearn metric calls below intentionally avoid the ``zero_division=``
kwarg (added in sklearn 0.22) so this module also works on older
installations — including the legacy py37 image CodaBench uses for
server-side scoring. Sklearn's default behaviour for a class with no
positive support is to return 0 and emit ``UndefinedMetricWarning``,
which is numerically identical to ``zero_division=0`` but noisier; we
silence the warning instead.
"""
from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
)

warnings.filterwarnings("ignore", category=UserWarning)
try:
    from sklearn.exceptions import UndefinedMetricWarning
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
except Exception:
    pass


def _slug(name: str) -> str:
    return (
        str(name)
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
    )


# --------------------------------------------------------------------------
# Single-label (Subtask A1)
# --------------------------------------------------------------------------
def compute_single_label_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str] | None = None,
) -> dict[str, float]:
    """Accuracy + macro / weighted P/R/F1 (+ per-class P/R/F1)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro")),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro")),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted")),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
    }
    labels = (
        list(range(len(class_names)))
        if class_names is not None
        else sorted(set(np.concatenate([y_true, y_pred]).tolist()))
    )
    if class_names is None:
        class_names = [str(i) for i in labels]
    p = precision_score(y_true, y_pred, average=None, labels=labels)
    r = recall_score(y_true, y_pred, average=None, labels=labels)
    f = f1_score(y_true, y_pred, average=None, labels=labels)
    for name, pp, rr, ff in zip(class_names, p, r, f):
        s = _slug(name)
        metrics[f"precision__{s}"] = float(pp)
        metrics[f"recall__{s}"] = float(rr)
        metrics[f"f1__{s}"] = float(ff)
    return metrics


# --------------------------------------------------------------------------
# Multi-label (Subtask A2)
# --------------------------------------------------------------------------
def compute_multilabel_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str] | None = None,
) -> dict[str, float]:
    """
    Multi-label macro/micro/weighted P/R/F1 + per-label P/R/F1 + support.

    Parameters
    ----------
    y_true : (N, K) binary {0,1}
    y_pred : (N, K) binary {0,1}
    class_names : optional names for the K classes.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    K = y_true.shape[1]
    if class_names is None:
        class_names = [f"class_{i}" for i in range(K)]

    metrics: dict[str, float] = {
        "subset_accuracy": float(accuracy_score(y_true, y_pred)),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "precision_micro": float(precision_score(y_true, y_pred, average="micro")),
        "recall_micro": float(recall_score(y_true, y_pred, average="micro")),
        "f1_micro": float(f1_score(y_true, y_pred, average="micro")),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro")),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro")),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted")),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
    }
    per_p = precision_score(y_true, y_pred, average=None)
    per_r = recall_score(y_true, y_pred, average=None)
    per_f = f1_score(y_true, y_pred, average=None)
    support = y_true.sum(axis=0)
    for i, name in enumerate(class_names):
        s = _slug(name)
        metrics[f"precision__{s}"] = float(per_p[i])
        metrics[f"recall__{s}"] = float(per_r[i])
        metrics[f"f1__{s}"] = float(per_f[i])
        metrics[f"support__{s}"] = int(support[i])
    return metrics
