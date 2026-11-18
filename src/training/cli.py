import argparse
import sys
from pathlib import Path

from src.training.config import load_config
from src.training.trainer import fit


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Food-11 ResNet-18 from a YAML config.")
    parser.add_argument("--config", type=Path, required=True, help="Experiment YAML config")
    parser.add_argument("--resume", type=Path, default=None, help="last.pt of an interrupted run")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--num-workers", type=int, default=None, help="Override num_workers")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument(
        "--limit-batches",
        type=int,
        default=None,
        help="Use only the first N batches per epoch (smoke tests)",
    )
    return parser.parse_args(args)


def main(args: list[str] | None = None) -> int:
    options = parse_args(args)
    try:
        config = load_config(
            options.config,
            overrides={"num_workers": options.num_workers, "epochs": options.epochs},
        )
        summary = fit(
            config,
            device=options.device,
            resume=options.resume,
            limit_batches=options.limit_batches,
        )
    except (OSError, ValueError) as error:
        print(f"Training failed: {error}", file=sys.stderr)
        return 1
    print(
        f"Best validation macro-F1 {summary['best_val_macro_f1']:.4f} at epoch "
        f"{summary['best_epoch']} ({summary['epochs_completed']} epochs"
        f"{', stopped early' if summary['stopped_early'] else ''}) "
        f"in {summary['training_seconds'] / 60:.1f} min on "
        f"{summary['hardware'].get('gpu', summary['hardware']['device'])}"
    )
    print(f"Best model: {summary['best_checkpoint']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
