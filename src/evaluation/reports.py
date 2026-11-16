"""Headless report figures for slides and error analysis."""

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image


def save_metric_plots(metrics: dict[str, Any], output_dir: Path) -> None:
    labels = [name.replace("_", " ").title() for name in metrics["classes"]]
    counts = np.asarray(metrics["confusion_matrix"])
    totals = counts.sum(axis=1, keepdims=True)
    normalized = np.divide(counts, totals, out=np.zeros_like(counts, dtype=float), where=totals > 0)
    for matrix, suffix, fmt in [(counts, "counts", "d"), (normalized, "normalized", ".2f")]:
        fig, ax = plt.subplots(figsize=(12, 10), layout="constrained")
        sns.heatmap(
            matrix,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
            vmin=0,
            vmax=1 if suffix == "normalized" else None,
        )
        ax.set(xlabel="Predicted class", ylabel="True class", title=f"Confusion matrix ({suffix})")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        plt.setp(ax.get_yticklabels(), rotation=0)
        fig.savefig(output_dir / f"confusion_matrix_{suffix}.png", dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6), layout="constrained")
    positions = np.arange(len(labels))
    for offset, metric in zip((-0.25, 0, 0.25), ("precision", "recall", "f1"), strict=True):
        values = [metrics["per_class"][name][metric] for name in metrics["classes"]]
        ax.bar(positions + offset, values, width=0.25, label=metric.title())
    ax.set(xticks=positions, xticklabels=labels, ylim=(0, 1.05), title="Per-class performance")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.legend()
    fig.savefig(output_dir / "per_class_metrics.png", dpi=180)
    plt.close(fig)


def save_gradcam_plot(image: Image.Image, heatmap: np.ndarray, path: Path, title: str) -> None:
    """The image and heatmap must use exactly the same resize/center-crop geometry."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), layout="constrained")
    for ax in axes:
        ax.imshow(image)
        ax.axis("off")
    axes[0].set_title("Model input crop")
    axes[1].imshow(heatmap, cmap="jet", alpha=0.45, vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM: predicted class")
    fig.suptitle(title)
    fig.savefig(path, dpi=180)
    plt.close(fig)
