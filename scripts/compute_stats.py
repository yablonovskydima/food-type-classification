import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.stats import compute_mean_std  # noqa: E402

DEFAULT_DIR = ROOT / "data" / "processed" / "food11" / "training"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute RGB mean/std of an image folder")
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR, help="Image folder")
    args = parser.parse_args()

    mean, std = compute_mean_std(args.dir)
    print(f"FOOD11_MEAN = {mean}")
    print(f"FOOD11_STD = {std}")


if __name__ == "__main__":
    main()
