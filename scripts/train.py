"""Train a Food-11 model and evaluate it on the validation split in one command.

Steps:
    1. Check that data/processed/food11 exists and a CUDA GPU is available.
    2. Train with the given YAML config (validation after every epoch, early stopping,
       TensorBoard logs in runs/<experiment>/).
    3. Evaluate the best checkpoint on the validation split: metrics, confusion matrices
       and Grad-CAM in reports/<experiment>/validation/.
    4. Print the results to copy into the README (best epoch, metrics, time, hardware).

The held-out test split is never used here; evaluate the final model on it separately.

Usage:
    uv run scripts/train.py                                            # main experiment
    uv run scripts/train.py --config configs/resnet18_pretrained.yaml  # pretrained reference
    uv run scripts/train.py --resume                                   # continue after a stop
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.evaluation.pipeline import evaluate_checkpoint  # noqa: E402
from src.training.checkpoint import BEST_CHECKPOINT, LAST_CHECKPOINT  # noqa: E402
from src.training.config import TrainConfig, load_config  # noqa: E402
from src.training.trainer import fit  # noqa: E402

DEFAULT_CONFIG = Path("configs") / "resnet18_scratch.yaml"
SPLITS = ("training", "validation")


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


def check_data(config: TrainConfig) -> None:
    data_dir = Path(config.data_dir)
    missing = [split for split in SPLITS if not (data_dir / split).is_dir()]
    if missing:
        raise FileNotFoundError(
            f"Processed dataset not found in {data_dir} (missing: {', '.join(missing)}). "
            "Run `uv run scripts/download_data.py` and `uv run scripts/prepare_data.py` first."
        )


def choose_device(allow_cpu: bool) -> str:
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        return "cuda"
    if not allow_cpu:
        raise RuntimeError(
            "No CUDA GPU visible to PyTorch. Check the NVIDIA driver and that the CUDA build "
            "is installed (`uv sync`), or pass --allow-cpu to train on the CPU anyway."
        )
    print("Warning: training on the CPU, this will take a very long time.")
    return "cpu"


def report_dir(config: TrainConfig) -> Path:
    # Evaluation refuses non-empty folders, so a re-run gets a timestamped folder.
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
    check_data(config)
    device = choose_device(options.allow_cpu)

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
    if options.config.exists():  # A config given relative to the current folder.
        options.config = options.config.resolve()
    previous_cwd = Path.cwd()
    os.chdir(ROOT)  # Paths in configs are relative to the repository root.
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
