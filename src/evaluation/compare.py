import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

CURVES = (
    ("val_accuracy", "Validation accuracy"),
    ("val_macro_f1", "Validation macro-F1"),
    ("val_loss", "Validation loss"),
)


@dataclass
class Run:
    experiment: str
    label: str
    history: list[dict[str, Any]]
    summary: dict[str, Any]
    validation_report: dict[str, Any] | None


def _file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def find_validation_report(reports_dir: Path, best_checkpoint: Path) -> dict[str, Any] | None:
    """Newest validation report of this experiment that was made from the current best.pt."""
    if not best_checkpoint.is_file():
        return None
    checkpoint_hash = _file_sha256(best_checkpoint)
    candidates = sorted(
        reports_dir.glob("validation*/metrics.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        report = json.loads(path.read_text(encoding="utf-8"))
        run = report.get("run", {})
        if run.get("split") == "validation" and run.get("checkpoint_sha256") == checkpoint_hash:
            return report
    return None


def load_run(
    experiment: str,
    label: str,
    checkpoint_dir: str | Path = "checkpoints",
    reports_dir: str | Path = "reports",
) -> Run:
    run_dir = Path(checkpoint_dir) / experiment
    history_path, summary_path = run_dir / "history.json", run_dir / "summary.json"
    if not history_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(
            f"No finished run for '{experiment}' in {run_dir}. "
            f"Train it first: uv run scripts/train.py --config configs/{experiment}.yaml"
        )
    history = json.loads(history_path.read_text(encoding="utf-8"))
    if not history:
        raise ValueError(f"{history_path} is empty")
    return Run(
        experiment=experiment,
        label=label,
        history=history,
        summary=json.loads(summary_path.read_text(encoding="utf-8")),
        validation_report=find_validation_report(
            Path(reports_dir) / experiment, run_dir / "best.pt"
        ),
    )


def plot_validation_curves(runs: list[Run], path: Path) -> None:
    fig, axes = plt.subplots(1, len(CURVES), figsize=(6 * len(CURVES), 4.5), layout="constrained")
    for ax, (key, title) in zip(axes, CURVES, strict=True):
        for run in runs:
            epochs = [row["epoch"] for row in run.history]
            values = [row[key] for row in run.history]
            (line,) = ax.plot(epochs, values, label=run.label, linewidth=2)
            best = next(
                (row for row in run.history if row["epoch"] == run.summary["best_epoch"]), None
            )
            if best is not None:
                ax.scatter(
                    best["epoch"], best[key], color=line.get_color(), s=60, zorder=3, marker="*"
                )
        ax.set(title=title, xlabel="Epoch")
        ax.grid(alpha=0.3)
        if key != "val_loss":
            ax.set_ylim(0, 1)
    axes[0].legend(loc="lower right")
    fig.suptitle("ResNet-18 on Food-11: training from scratch vs ImageNet pretraining (★ best)")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _format_minutes(seconds: float) -> str:
    return f"{seconds / 60:.1f} min"


def results_table(runs: list[Run]) -> str:
    lines = [
        "| Model | Hardware | Training time | Epochs (best) | Val accuracy | Val top-3 "
        "| Val macro-F1 |",
        "|-------|----------|---------------|---------------|--------------|-----------"
        "|--------------|",
    ]
    for run in runs:
        summary = run.summary
        hardware = summary.get("hardware", {})
        best = summary.get("best_val_metrics", {})
        top3 = f"{run.validation_report['top3_accuracy']:.3f}" if run.validation_report else "—"
        lines.append(
            f"| {run.label} | {hardware.get('gpu', hardware.get('device', '—'))} "
            f"| {_format_minutes(summary['training_seconds'])} "
            f"| {summary['epochs_completed']} ({summary['best_epoch']}) "
            f"| {best.get('accuracy', float('nan')):.3f} | {top3} "
            f"| {summary['best_val_macro_f1']:.3f} |"
        )
    return "\n".join(lines)


def compare_runs(
    experiments: dict[str, str],
    output_dir: str | Path = Path("reports") / "comparison",
    checkpoint_dir: str | Path = "checkpoints",
    reports_dir: str | Path = "reports",
) -> dict[str, Any]:
    if len(experiments) < 2:
        raise ValueError("Compare at least two experiments")
    runs = [
        load_run(experiment, label, checkpoint_dir, reports_dir)
        for experiment, label in experiments.items()
    ]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    curves_path = output_dir / "val_curves.png"
    plot_validation_curves(runs, curves_path)
    table = results_table(runs)
    table_path = output_dir / "results.md"
    table_path.write_text(table + "\n", encoding="utf-8")
    return {"curves": curves_path, "table_path": table_path, "table": table, "runs": runs}
