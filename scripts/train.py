"""Train a model from a YAML config and evaluate it on the validation split."""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.pipeline import evaluate_checkpoint  # noqa: E402
from src.training.checkpoint import BEST_CHECKPOINT, LAST_CHECKPOINT  # noqa: E402
from src.training.config import TrainConfig, load_config  # noqa: E402
from src.training.trainer import check_processed_data, fit, select_device  # noqa: E402

DEFAULT_CONFIG = Path("configs") / "resnet18_scratch.yaml"


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Food-11 model and evaluate it")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--resume", action="store_true", help="Continue the run from last.pt")
    parser.add_argument(
        "--allow-cpu", action="store_true", help="Train without a GPU (very slow, ~days)"
    )
    parser.add_argument("--no-eval", action="store_true", help="Skip validation reports")
    parser.add_argument("--num-workers", type=int, default=None, help="Override num_workers")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument(
        "--limit-batches", type=int, default=None, help="Only N batches per epoch (smoke test)"
    )
    return parser.parse_args(args)


def report_dir(config: TrainConfig) -> Path:
    base = Path("reports") / config.experiment
    target = base / "validation"
    if target.exists() and any(target.iterdir()):
        target = base / f"validation_{datetime.now():%Y%m%d_%H%M%S}"
    return target


def run(options: argparse.Namespace) -> None:
    config = load_config(
        options.config,
        overrides={"num_workers": options.num_workers, "epochs": options.epochs},
    )
    check_processed_data(config.data_dir)
    device = select_device(options.allow_cpu)

    run_dir = config.run_checkpoint_dir
    resume = None
    if options.resume:
        resume = run_dir / LAST_CHECKPOINT
        if not resume.is_file():
            raise FileNotFoundError(f"Nothing to resume: {resume} does not exist")
    elif (run_dir / LAST_CHECKPOINT).exists():
        raise FileExistsError(
            f"{run_dir} already contains a run. Continue it with "
            f"`uv run scripts/train.py --config {options.config} --resume`, "
            "or change 'experiment' in the config to start a new one."
        )

    summary = fit(
        config,
        device=device,
        resume=str(resume) if resume else None,
        limit_batches=options.limit_batches,
    )

    metrics = None
    if not options.no_eval:
        output_dir = report_dir(config)
        print(f"Evaluating {run_dir / BEST_CHECKPOINT} on the validation split...")
        metrics = evaluate_checkpoint(
            run_dir / BEST_CHECKPOINT,
            config.data_dir,
            output_dir,
            split="validation",
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            device=device,
        )

    hardware = summary["hardware"]
    best = summary["best_val_metrics"]
    print("\n=== Results ===")
    print(f"Experiment:      {config.experiment}")
    print(f"Hardware:        {hardware.get('gpu', hardware['cpu'])} (torch {hardware['torch']})")
    print(
        f"Training time:   {summary['training_seconds'] / 60:.1f} min, "
        f"{summary['epochs_completed']} epochs"
        f"{' (stopped early)' if summary['stopped_early'] else ''}"
    )
    print(
        f"Best epoch:      {summary['best_epoch']} | val accuracy {best.get('accuracy', 0):.4f} "
        f"| val macro-F1 {summary['best_val_macro_f1']:.4f}"
    )
    if metrics is not None:
        print(f"Val top-3 acc:   {metrics['top3_accuracy']:.4f}")
        print(f"Reports:         {output_dir}")
    print(f"Best model:      {run_dir / BEST_CHECKPOINT}")
    print(f"Run summary:     {run_dir / 'summary.json'}")
    print(f"Curves:          uv run tensorboard --logdir {config.run_log_dir}")


def main(args: list[str] | None = None) -> int:
    options = parse_args(args)
    if options.config.exists():
        options.config = options.config.resolve()
    previous_cwd = Path.cwd()
    os.chdir(ROOT)
    try:
        run(options)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Training pipeline failed: {error}", file=sys.stderr)
        return 1
    finally:
        os.chdir(previous_cwd)
    return 0


if __name__ == "__main__":
    sys.exit(main())
