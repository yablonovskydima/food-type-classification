"""Export a final Food-11 checkpoint with reproducible validation and CPU timings."""

import argparse
import sys
from pathlib import Path

from src.inference.onnx_export import export_checkpoint


def main(args: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export and verify a Food-11 checkpoint as ONNX")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="New directory for model/report"
    )
    parser.add_argument("--image-size", type=int, help="Required for checkpoints without metadata")
    parser.add_argument("--normalization", choices=("food11", "imagenet"))
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    options = parser.parse_args(args)
    try:
        report = export_checkpoint(**vars(options))
    except (OSError, ValueError, RuntimeError, AssertionError) as error:
        print(f"Export failed: {error}", file=sys.stderr)
        return 1
    benchmark = report["benchmark"]
    print(f"Saved {options.output_dir / 'model.onnx'}; numerical parity passed")
    print(
        f"CPU mean: PyTorch {benchmark['pytorch']['mean_ms']:.2f} ms | "
        f"ONNX Runtime {benchmark['onnxruntime']['mean_ms']:.2f} ms | "
        f"speedup {benchmark['speedup']:.2f}x"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
