from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image


def display_name(class_name: str) -> str:
    return class_name.replace("_", " ").title()


def save_metric_plots(metrics: dict[str, Any], output_dir: Path) -> None:
    labels = [display_name(name) for name in metrics["classes"]]
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


def save_gradcam_plot(
    image: Image.Image,
    heatmaps: Sequence[tuple[str, np.ndarray]],
    path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(
        1, 1 + len(heatmaps), figsize=(4.5 * (1 + len(heatmaps)), 4.5), layout="constrained"
    )
    for ax in axes:
        ax.imshow(image)
        ax.axis("off")
    axes[0].set_title("Model input crop")
    for ax, (heatmap_title, heatmap) in zip(axes[1:], heatmaps, strict=True):
        ax.imshow(heatmap, cmap="jet", alpha=0.45, vmin=0, vmax=1)
        ax.set_title(heatmap_title)
    fig.suptitle(title)
    fig.savefig(path, dpi=180)
    plt.close(fig)
