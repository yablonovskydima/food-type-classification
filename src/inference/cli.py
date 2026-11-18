import argparse
import json
import sys
from pathlib import Path
from typing import Any

from src.inference.checkpoints import DEFAULT_CHECKPOINT
from src.inference.predictor import FoodPredictor


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference on food images using trained Food-11 models."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=Path, help="Path to a single image file")
    group.add_argument("--dir", type=Path, help="Path to a directory containing images")

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=f"Trained checkpoint (default: {DEFAULT_CHECKPOINT})",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top predictions to display (default: 3)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda"],
        help="Device to run inference on (default: auto-detect)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Print output in JSON format",
    )

    return parser.parse_args(args)


def format_text_output(filename: str, predictions: list[dict[str, Any]]) -> str:
    lines = [f"\nPredictions for {filename}:"]
    for i, pred in enumerate(predictions, start=1):
        pct = pred["probability"] * 100
        category = pred["class"].replace("_", " ").title()
        lines.append(f"  {i}. {pct:5.1f}%  -  {category}")
    return "\n".join(lines)


def main(cli_args: list[str] | None = None) -> int:
    args = parse_args(cli_args)
    try:
        # Input size and normalisation come from the checkpoint metadata.
        predictor = FoodPredictor(checkpoint_path=args.checkpoint, device=args.device)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Error: could not load the model: {error}", file=sys.stderr)
        return 1

    if args.image is not None:
        if not args.image.is_file():
            print(f"Error: image file not found: {args.image}", file=sys.stderr)
            return 1
        results = predictor.predict_image(args.image, top_k=args.top_k)
        if args.json_output:
            print(json.dumps({args.image.name: results}, indent=2))
        else:
            print(format_text_output(args.image.name, results))

    elif args.dir is not None:
        if not args.dir.is_dir():
            print(f"Error: directory not found: {args.dir}", file=sys.stderr)
            return 1
        dir_results = predictor.predict_directory(args.dir, top_k=args.top_k)
        if not dir_results:
            print(f"No valid images found in directory: {args.dir}", file=sys.stderr)
            return 0

        if args.json_output:
            print(json.dumps(dir_results, indent=2))
        else:
            for fname, results in dir_results.items():
                print(format_text_output(fname, results))

    return 0


if __name__ == "__main__":
    sys.exit(main())
