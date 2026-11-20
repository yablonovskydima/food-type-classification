"""Evaluate a trained model on the validation split, or once on the test split."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.pipeline import evaluate_checkpoint  # noqa: E402


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained Food-11 checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="New or empty folder")
    parser.add_argument(
        "--split",
        choices=("validation", "evaluation"),
        default="validation",
        help="'evaluation' is the test set: use it once per final model",
    )
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "processed" / "food11")
    parser.add_argument(
        "--image-size", type=int, default=None, help="Only for checkpoints without metadata"
    )
    parser.add_argument(
        "--imagenet-stats",
        action="store_const",
        const=True,
        default=None,
        dest="use_imagenet_stats",
        help="Only for checkpoints without metadata",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--gradcam-examples", type=int, default=8, help="0 disables Grad-CAM")
    return parser.parse_args(args)


def main(args: list[str] | None = None) -> int:
    options = parse_args(args)
    try:
        metrics = evaluate_checkpoint(**vars(options))
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Evaluation failed: {error}", file=sys.stderr)
        return 1
    print(
        f"{metrics['num_samples']} images | accuracy={metrics['accuracy']:.4f} | "
        f"top-3={metrics['top3_accuracy']:.4f} | macro-F1={metrics['macro_f1']:.4f}"
    )
    print(f"Reports: {options.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
