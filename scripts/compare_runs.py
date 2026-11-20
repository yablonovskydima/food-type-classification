"""Plot validation curves of several runs on one chart and write the results table."""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.compare import compare_runs  # noqa: E402

DEFAULT_RUNS = {
    "resnet18_scratch": "ResNet-18 from scratch",
    "resnet18_pretrained": "ResNet-18 ImageNet-pretrained",
}


def parse_run(value: str) -> tuple[str, str]:
    experiment, _, label = value.partition("=")
    if not experiment:
        raise argparse.ArgumentTypeError("Use --run <experiment>[=<label>]")
    return experiment, label or experiment


def main(args: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare training runs on one chart")
    parser.add_argument(
        "--run",
        type=parse_run,
        action="append",
        dest="runs",
        help="experiment[=label]; repeat for each run (default: scratch vs pretrained)",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Default: reports/comparison")
    options = parser.parse_args(args)

    previous_cwd = Path.cwd()
    output_dir = (
        options.output_dir.resolve()
        if options.output_dir is not None
        else ROOT / "reports" / "comparison"
    )
    os.chdir(ROOT)  # checkpoints/ and reports/ are relative to the repository root.
    try:
        result = compare_runs(dict(options.runs or DEFAULT_RUNS.items()), output_dir)
    except (OSError, ValueError) as error:
        print(f"Comparison failed: {error}", file=sys.stderr)
        return 1
    finally:
        os.chdir(previous_cwd)

    print(result["table"])
    print(f"\nCurves: {result['curves']}")
    print(f"Table:  {result['table_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
