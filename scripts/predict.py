"""Predict the food category of photos with a trained model (.pt or exported .onnx)."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_MODEL = ROOT / "checkpoints" / "resnet18_scratch" / "best.pt"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict food categories for photos")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="A single photo")
    source.add_argument("--dir", type=Path, help="A folder of photos")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="best.pt or model.onnx")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None, help=".pt only")
    parser.add_argument("--json", action="store_true", dest="json_output")
    return parser.parse_args(args)


def load_predictor(model: Path, device: str | None):
    if model.suffix == ".onnx":
        from src.inference.onnx_predictor import OnnxFoodPredictor

        return OnnxFoodPredictor(model)
    from src.inference.predictor import FoodPredictor

    return FoodPredictor(checkpoint_path=model, device=device)


def image_paths(options: argparse.Namespace) -> list[Path]:
    if options.image is not None:
        if not options.image.is_file():
            raise FileNotFoundError(f"Image not found: {options.image}")
        return [options.image]
    if not options.dir.is_dir():
        raise NotADirectoryError(f"Folder not found: {options.dir}")
    return sorted(path for path in options.dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


def format_text_output(filename: str, predictions: list[dict]) -> str:
    lines = [f"\n{filename}:"]
    for rank, prediction in enumerate(predictions, start=1):
        category = prediction["class"].replace("_", " ").title()
        lines.append(f"  {rank}. {prediction['probability'] * 100:5.1f}%  {category}")
    return "\n".join(lines)


def main(args: list[str] | None = None) -> int:
    options = parse_args(args)
    try:
        paths = image_paths(options)
        predictor = load_predictor(options.model, options.device)
        results = {path.name: predictor.predict_image(path, top_k=options.top_k) for path in paths}
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Prediction failed: {error}", file=sys.stderr)
        return 1
    if options.json_output:
        print(json.dumps(results, indent=2))
    else:
        for name, predictions in results.items():
            print(format_text_output(name, predictions))
    return 0


if __name__ == "__main__":
    sys.exit(main())
