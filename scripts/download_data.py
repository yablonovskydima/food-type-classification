"""Download the Food-11 dataset from Kaggle into data/raw and verify it.

Kaggle credentials are read from the `.env` file in the project root
(copy `.env.example` to `.env` and fill it in).

Usage:
    uv run scripts/download_data.py                 # download (skipped if already valid) + verify
    uv run scripts/download_data.py --force         # delete and download again
    uv run scripts/download_data.py --verify-only   # only check what is on disk
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
DATASET = "trolukovich/food11-image-dataset"
DEFAULT_DST = ROOT / "data" / "raw" / "food11"

EXPECTED_IMAGES = {"training": 9866, "validation": 3430, "evaluation": 3347}
NUM_CLASSES = 11
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
CREDENTIAL_VARS = ("KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY")


def load_credentials() -> None:
    load_dotenv(ROOT / ".env")

    for var in CREDENTIAL_VARS:
        if os.environ.get(var) == "":
            del os.environ[var]

    has_token = bool(os.getenv("KAGGLE_API_TOKEN"))
    has_user_key = bool(os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"))
    kaggle_dir = Path.home() / ".kaggle"
    has_kaggle_files = any(
        (kaggle_dir / name).exists() for name in ("access_token", "kaggle.json")
    )

    if not (has_token or has_user_key or has_kaggle_files):
        sys.exit(
            "Kaggle credentials not found.\n"
            "Copy .env.example to .env and set KAGGLE_API_TOKEN "
            "(or the legacy KAGGLE_USERNAME + KAGGLE_KEY).\n"
            "Generate a token at https://www.kaggle.com/settings/api."
        )


def find_split_root(path: Path) -> Path | None:
    candidates = [path]
    for child in (p for p in path.iterdir() if p.is_dir()) if path.is_dir() else []:
        candidates.append(child)
        candidates.extend(p for p in child.iterdir() if p.is_dir())
    for candidate in candidates:
        if all((candidate / split).is_dir() for split in EXPECTED_IMAGES):
            return candidate
    return None


def flatten(dst: Path) -> None:
    root = find_split_root(dst)
    if root is None or root == dst:
        return
    for split in EXPECTED_IMAGES:
        shutil.move(str(root / split), str(dst / split))
    for leftover in sorted(dst.iterdir()):
        if leftover.is_dir() and leftover.name not in EXPECTED_IMAGES:
            shutil.rmtree(leftover)


def verify(dst: Path) -> bool:
    root = find_split_root(dst) if dst.exists() else None
    if root is None:
        print(f"No dataset found in {dst}")
        return False

    ok = True
    print(f"Dataset location: {root}")
    for split, expected in EXPECTED_IMAGES.items():
        class_dirs = sorted(d for d in (root / split).iterdir() if d.is_dir())
        images = sum(
            1
            for class_dir in class_dirs
            for f in class_dir.iterdir()
            if f.suffix.lower() in IMAGE_EXTENSIONS
        )
        split_ok = len(class_dirs) == NUM_CLASSES and images == expected
        ok &= split_ok
        mark = "OK " if split_ok else "BAD"
        print(
            f"  [{mark}] {split:<11} {len(class_dirs):>2}/{NUM_CLASSES} classes, "
            f"{images:>5}/{expected} images"
        )
    return ok


def download(dst: Path) -> None:
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as exc:
        sys.exit(f"Kaggle authentication failed: {exc}")

    dst.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {DATASET} into {dst} ...")
    try:
        api.dataset_download_files(DATASET, path=str(dst), unzip=True, quiet=False)
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"Download failed: {exc}")
    flatten(dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and verify the Food-11 dataset")
    parser.add_argument("--dst", type=Path, default=DEFAULT_DST, help="Target folder")
    parser.add_argument("--force", action="store_true", help="Re-download even if data exists")
    parser.add_argument("--verify-only", action="store_true", help="Only verify existing data")
    args = parser.parse_args()
    dst: Path = args.dst.resolve()

    if args.verify_only:
        sys.exit(0 if verify(dst) else 1)

    if dst.exists() and not args.force:
        if verify(dst):
            print("Dataset already present and valid. Use --force to download again.")
            return
        print("Existing data is incomplete, downloading again.")

    if dst.exists():
        shutil.rmtree(dst)

    load_credentials()
    download(dst)

    if not verify(dst):
        sys.exit("Downloaded dataset does not match the expected structure.")
    print("Done.")


if __name__ == "__main__":
    main()
