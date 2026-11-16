"""Classification metrics with a stable class order and explicit undefined AUCs."""

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def compute_metrics(
    targets: np.ndarray, probabilities: np.ndarray, classes: Sequence[str]
) -> dict[str, Any]:
    """Compute one-vs-rest AUC; return None when a class lacks positives or negatives."""
    targets = np.asarray(targets)
    probabilities = np.asarray(probabilities)
    num_classes = len(classes)
    if num_classes < 3 or len(set(classes)) != num_classes:
        raise ValueError("At least three unique class names are required")
    if targets.ndim != 1 or targets.size == 0 or targets.dtype.kind not in "iu":
        raise ValueError("Targets must be a nonempty one-dimensional integer array")
    if probabilities.shape != (len(targets), num_classes):
        raise ValueError("Probability shape must be (number of targets, number of classes)")
    if (targets < 0).any() or (targets >= num_classes).any():
        raise ValueError("Targets contain an invalid class index")
    if (
        not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or (probabilities > 1).any()
        or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5)
    ):
        raise ValueError("Probabilities must be finite, in [0, 1], and sum to one per row")

    predictions = probabilities.argmax(axis=1)
    labels = np.arange(num_classes)
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, predictions, labels=labels, zero_division=0
    )
    top3 = np.argsort(-probabilities, axis=1, kind="stable")[:, :3]
    per_class = {}
    for index, name in enumerate(classes):
        binary_targets = targets == index
        auc = (
            float(roc_auc_score(binary_targets, probabilities[:, index]))
            if 0 < binary_targets.sum() < len(targets)
            else None
        )
        per_class[name] = {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
            "roc_auc_ovr": auc,
        }
    return {
        "num_samples": len(targets),
        "classes": list(classes),
        "accuracy": float(accuracy_score(targets, predictions)),
        "top3_accuracy": float(np.mean((top3 == targets[:, None]).any(axis=1))),
        "macro_f1": float(
            f1_score(targets, predictions, labels=labels, average="macro", zero_division=0)
        ),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(targets, predictions, labels=labels).tolist(),
    }
