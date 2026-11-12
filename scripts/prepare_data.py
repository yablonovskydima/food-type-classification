"""Prepare Food-11 for training: data/raw/food11 -> data/processed/food11.

Steps:
    1. Remove exact duplicates (identical pixels). Food-11 has copies of the same
       image in different splits, which leaks test images into training. A duplicate
       is kept in the earliest split (training > validation > evaluation) and dropped
       from the later ones; repeats inside one split keep only the first file.
    2. Resize so the shorter side is `--size` px (default 256), keeping the aspect
       ratio. Smaller images are never upscaled. Saved as JPEG.
    3. Rename class folders to snake_case ("Dairy product" -> "dairy_product").

The output folder is recreated on every run.

Usage:
    uv run scripts/prepare_data.py
    uv run scripts/prepare_data.py --size 160 --workers 8
"""

import argparse
import hashlib
import os
import re
import shutil
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image, ImageOps
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SRC = ROOT / "data" / "raw" / "food11"
DEFAULT_DST = ROOT / "data" / "processed" / "food11"

SPLITS = ("training", "validation", "evaluation")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
JPEG_QUALITY = 95


def snake_case(name: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_").lower()


def pixel_hash(path: Path) -> str:
    with Image.open(path) as img:
        img = img.convert("RGB")
        return hashlib.md5(img.tobytes() + str(img.size).encode()).hexdigest()


def collect_images(src: Path) -> list[tuple[str, str, Path]]:
    images = []
    for split in SPLITS:
        for class_dir in sorted(p for p in (src / split).iterdir() if p.is_dir()):
            for path in sorted(class_dir.iterdir()):
                if path.suffix.lower() in IMAGE_EXTENSIONS:
                    images.append((split, class_dir.name, path))
    return images


def deduplicate(
    images: list[tuple[str, str, Path]], hashes: list[str]
) -> tuple[list[tuple[str, str, Path]], Counter]:
    seen: set[str] = set()
    kept, removed = [], Counter()
    for item, digest in zip(images, hashes, strict=True):
        if digest in seen:
            removed[item[0]] += 1
            continue
        seen.add(digest)
        kept.append(item)
    return kept, removed


def process_image(task: tuple[Path, Path, int]) -> None:
    src, dst, size = task
    with Image.open(src) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        scale = size / min(img.size)
        if scale < 1:
            new_size = (round(img.width * scale), round(img.height * scale))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        img.save(dst.with_suffix(".jpg"), "JPEG", quality=JPEG_QUALITY)


def main() -> None:
    parser = argparse.ArgumentParser(description="Deduplicate and resize Food-11")
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--dst", type=Path, default=DEFAULT_DST)
    parser.add_argument("--size", type=int, default=256, help="Target shorter side in px")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    args = parser.parse_args()
    src, dst = args.src.resolve(), args.dst.resolve()

    missing = [s for s in SPLITS if not (src / s).is_dir()]
    if missing:
        sys.exit(
            f"Raw dataset not found in {src} (missing: {', '.join(missing)}).\n"
            "Run `uv run scripts/download_data.py` first."
        )

    images = collect_images(src)
    print(f"Found {len(images)} images in {src}")

    with ProcessPoolExecutor(args.workers) as pool:
        hashes = list(
            tqdm(
                pool.map(pixel_hash, [p for _, _, p in images], chunksize=64),
                total=len(images),
                desc="Hashing",
            )
        )
    kept, removed = deduplicate(images, hashes)

    if dst.exists():
        shutil.rmtree(dst)
    tasks = []
    for split, class_name, path in kept:
        out_dir = dst / split / snake_case(class_name)
        out_dir.mkdir(parents=True, exist_ok=True)
        tasks.append((path, out_dir / path.name, args.size))

    with ProcessPoolExecutor(args.workers) as pool:
        list(
            tqdm(
                pool.map(process_image, tasks, chunksize=32),
                total=len(tasks),
                desc="Resizing",
            )
        )

    print(f"\nSaved {len(kept)} images to {dst}")
    for split in SPLITS:
        total = sum(1 for s, _, _ in images if s == split)
        kept_n = sum(1 for s, _, _ in kept if s == split)
        print(
            f"  {split:<11} {kept_n:>5} kept, "
            f"{removed[split]:>4} duplicates removed (of {total})"
        )


if __name__ == "__main__":
    main()
