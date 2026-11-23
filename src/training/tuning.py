import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.training.checkpoint import LAST_CHECKPOINT
from src.training.config import TrainConfig, load_config

TUNING_CHECKPOINT_DIR = Path("checkpoints") / "tuning"
TUNING_LOG_DIR = Path("runs") / "tuning"


@dataclass
class Sweep:
    name: str
    base_config: Path
    overrides: dict[str, Any]
    trials: dict[str, dict[str, Any]]


@dataclass
class TrialResult:
    trial: str
    changes: dict[str, Any]
    summary: dict[str, Any]


def load_sweep(path: str | Path) -> Sweep:
    with open(path, encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    unknown = sorted(set(data) - {"name", "base_config", "overrides", "trials"})
    if unknown:
        raise ValueError(f"Unknown keys in {path}: {unknown}")
    trials = data.get("trials") or {}
    if not data.get("name") or not data.get("base_config") or not trials:
        raise ValueError(f"{path} must define name, base_config and at least one trial")
    sweep = Sweep(
        name=data["name"],
        base_config=Path(data["base_config"]),
        overrides=data.get("overrides") or {},
        trials={name: changes or {} for name, changes in trials.items()},
    )
    trial_configs(sweep)  # Fail early on typos or invalid values in any trial.
    return sweep


def trial_configs(sweep: Sweep) -> dict[str, TrainConfig]:
    return {
        trial: load_config(
            sweep.base_config,
            overrides={
                **sweep.overrides,
                **changes,
                "experiment": f"{sweep.name}_{trial}",
                "checkpoint_dir": str(TUNING_CHECKPOINT_DIR),
                "log_dir": str(TUNING_LOG_DIR),
            },
        )
        for trial, changes in sweep.trials.items()
    }


def run_sweep(
    sweep: Sweep,
    fit: Callable[..., dict[str, Any]],
    *,
    device: str,
    only: list[str] | None = None,
    limit_batches: int | None = None,
) -> list[TrialResult]:
    """Run every trial; finished trials are reused and interrupted ones are resumed."""
    configs = trial_configs(sweep)
    unknown = sorted(set(only or []) - configs.keys())
    if unknown:
        raise ValueError(f"Unknown trials: {unknown}")
    results = []
    for trial, config in configs.items():
        if only and trial not in only:
            continue
        run_dir = config.run_checkpoint_dir
        summary_path = run_dir / "summary.json"
        if summary_path.is_file():
            print(f"[{trial}] already finished, reusing {summary_path}")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        else:
            last = run_dir / LAST_CHECKPOINT
            resume = str(last) if last.is_file() else None
            print(f"\n=== Trial {trial}: {sweep.trials[trial] or 'baseline'} ===")
            summary = fit(config, device=device, resume=resume, limit_batches=limit_batches)
        results.append(TrialResult(trial, sweep.trials[trial], summary))
    return results


def results_table(results: list[TrialResult]) -> str:
    rows = sorted(results, key=lambda result: result.summary["best_val_macro_f1"], reverse=True)
    lines = [
        "| Trial | Changes | Val macro-F1 | Val accuracy | Best epoch | Time |",
        "|-------|---------|--------------|--------------|------------|------|",
    ]
    for result in rows:
        summary = result.summary
        changes = ", ".join(f"{key}={value}" for key, value in result.changes.items()) or "—"
        lines.append(
            f"| {result.trial} | {changes} | {summary['best_val_macro_f1']:.3f} "
            f"| {summary['best_val_metrics'].get('accuracy', float('nan')):.3f} "
            f"| {summary['best_epoch']}/{summary['epochs_completed']} "
            f"| {summary['training_seconds'] / 60:.1f} min |"
        )
    return "\n".join(lines)


def best_trial(results: list[TrialResult]) -> TrialResult:
    return max(results, key=lambda result: result.summary["best_val_macro_f1"])


def best_config(sweep: Sweep, result: TrialResult) -> dict[str, Any]:
    """The base config with the winning trial's changes, keeping the base training budget."""
    with open(sweep.base_config, encoding="utf-8") as stream:
        base = yaml.safe_load(stream) or {}
    return {**base, **result.changes}
