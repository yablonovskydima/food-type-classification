"""Run a hyper-parameter sweep of short trainings and report the best validation macro-F1."""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from src.evaluation.compare import load_run, plot_validation_curves  # noqa: E402
from src.training.trainer import check_processed_data, fit, select_device  # noqa: E402
from src.training.tuning import (  # noqa: E402
    TUNING_CHECKPOINT_DIR,
    best_config,
    best_trial,
    load_sweep,
    results_table,
    run_sweep,
    trial_configs,
)

DEFAULT_SWEEP = Path("configs") / "tuning" / "scratch.yaml"
REPORTS_DIR = Path("reports") / "tuning"


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a hyper-parameter sweep")
    parser.add_argument("--sweep", type=Path, default=DEFAULT_SWEEP)
    parser.add_argument("--only", nargs="+", metavar="TRIAL", help="Run only these trials")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--limit-batches", type=int, default=None, help="Smoke test")
    return parser.parse_args(args)


def run(options: argparse.Namespace) -> None:
    sweep = load_sweep(options.sweep)
    configs = trial_configs(sweep)
    check_processed_data(next(iter(configs.values())).data_dir)
    device = select_device(options.allow_cpu)

    results = run_sweep(
        sweep, fit, device=device, only=options.only, limit_batches=options.limit_batches
    )

    output_dir = REPORTS_DIR / sweep.name
    output_dir.mkdir(parents=True, exist_ok=True)
    table = results_table(results)
    (output_dir / "results.md").write_text(table + "\n", encoding="utf-8")
    runs = [
        load_run(configs[result.trial].experiment, result.trial, TUNING_CHECKPOINT_DIR, REPORTS_DIR)
        for result in results
    ]
    plot_validation_curves(
        runs, output_dir / "val_curves.png", title=f"Hyper-parameter sweep '{sweep.name}'"
    )
    best = best_trial(results)
    best_path = output_dir / "best_config.yaml"
    best_path.write_text(
        yaml.safe_dump(best_config(sweep, best), sort_keys=False), encoding="utf-8"
    )

    print(f"\n{table}\n")
    print(f"Best trial: {best.trial} ({best.changes or 'baseline'})")
    print(f"Results:    {output_dir / 'results.md'}")
    print(f"Curves:     {output_dir / 'val_curves.png'}")
    print(f"Best config (full training budget): {best_path}")


def main(args: list[str] | None = None) -> int:
    options = parse_args(args)
    if options.sweep.exists():
        options.sweep = options.sweep.resolve()
    previous_cwd = Path.cwd()
    os.chdir(ROOT)
    try:
        run(options)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Tuning failed: {error}", file=sys.stderr)
        return 1
    finally:
        os.chdir(previous_cwd)
    return 0


if __name__ == "__main__":
    sys.exit(main())
