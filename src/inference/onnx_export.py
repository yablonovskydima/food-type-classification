import hashlib
import json
import platform
import tempfile
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch

from src.data.constants import CLASSES, FOOD11_MEAN, FOOD11_STD, IMAGENET_MEAN, IMAGENET_STD
from src.inference.checkpoints import load_checkpoint, resolve_preprocessing
from src.inference.onnx_predictor import METADATA_KEY, OnnxFoodPredictor
from src.models import build_model


def load_export_model(
    checkpoint_path: Path, image_size: int | None, normalization: str | None
) -> tuple[torch.nn.Module, dict[str, Any]]:
    state_dict, saved = load_checkpoint(checkpoint_path, map_location="cpu")
    image_size, normalization = resolve_preprocessing(
        saved, image_size, normalization, require_metadata=True
    )
    mean, std = (
        (IMAGENET_MEAN, IMAGENET_STD) if normalization == "imagenet" else (FOOD11_MEAN, FOOD11_STD)
    )
    model = build_model(pretrained=False)
    model.load_state_dict(state_dict)
    model.eval()
    with checkpoint_path.open("rb") as stream:
        checkpoint_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    return model, {
        "schema_version": 1,
        "architecture": "resnet18",
        "classes": list(CLASSES),
        "image_size": image_size,
        "normalization": normalization,
        "mean": list(mean),
        "std": list(std),
        "resize_size": int(round(image_size * 256 / 224)),
        "checkpoint_sha256": checkpoint_hash,
    }


def _timings(run: Callable, warmup: int, iterations: int) -> dict[str, float]:
    for _ in range(warmup):
        run()
    samples = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        run()
        samples.append((time.perf_counter_ns() - start) / 1e6)
    return {
        "mean_ms": float(np.mean(samples)),
        "median_ms": float(np.median(samples)),
        "p95_ms": float(np.percentile(samples, 95)),
        "images_per_second": float(1000 / np.mean(samples)),
    }


def verify_parity(model: torch.nn.Module, predictor: OnnxFoodPredictor) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(12)
    results = []
    with torch.inference_mode():
        for batch_size in (1, 3):
            inputs = torch.randn(
                batch_size, 3, predictor.image_size, predictor.image_size, generator=generator
            )
            expected = model(inputs).numpy()
            actual = predictor.predict_logits(inputs.numpy())
            np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
            expected_probs = torch.softmax(torch.from_numpy(expected), dim=1).numpy()
            actual_probs = torch.softmax(torch.from_numpy(actual), dim=1).numpy()
            np.testing.assert_allclose(actual_probs, expected_probs, rtol=1e-4, atol=1e-5)
            if not np.array_equal(expected.argmax(axis=1), actual.argmax(axis=1)):
                raise ValueError("ONNX and PyTorch predicted different top-1 classes")
            results.append(
                {
                    "batch_size": batch_size,
                    "max_abs_logit_error": float(np.max(np.abs(actual - expected))),
                    "max_abs_probability_error": float(
                        np.max(np.abs(actual_probs - expected_probs))
                    ),
                    "top1_matches": True,
                }
            )
    return {
        "inputs": "seeded synthetic tensors",
        "seed": 12,
        "rtol": 1e-4,
        "atol": 1e-5,
        "batches": results,
        "passed": True,
    }


def export_checkpoint(
    checkpoint: str | Path,
    output_dir: str | Path,
    *,
    image_size: int | None = None,
    normalization: str | None = None,
    threads: int = 1,
    warmup: int = 5,
    iterations: int = 30,
) -> dict[str, Any]:
    """Publish a self-contained model and report only after checker/parity succeed."""
    if threads < 1 or warmup < 1 or iterations < 1:
        raise ValueError("Threads, warmup and benchmark iterations must be positive")
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Export directory already exists: {output_dir}")
    model, metadata = load_export_model(Path(checkpoint), image_size, normalization)
    size = metadata["image_size"]
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(threads)
        with tempfile.TemporaryDirectory(prefix=".onnx-", dir=output_dir.parent) as staging:
            staging = Path(staging)
            path = staging / "model.onnx"
            example = torch.zeros(2, 3, size, size)
            torch.onnx.export(
                model,
                (example,),
                str(path),
                input_names=["images"],
                output_names=["logits"],
                dynamo=True,
                opset_version=18,
                external_data=False,
                dynamic_shapes=({0: torch.export.Dim("batch", min=1)},),
            )
            graph = onnx.load(path)
            onnx.helper.set_model_props(graph, {METADATA_KEY: json.dumps(metadata)})
            onnx.checker.check_model(graph, full_check=True)
            onnx.save_model(graph, path)
            predictor = OnnxFoodPredictor(path, threads=threads)
            parity = verify_parity(model, predictor)
            inputs = torch.zeros(1, 3, size, size)
            array = inputs.numpy()
            with torch.inference_mode():
                pytorch_times = _timings(lambda: model(inputs), warmup, iterations)
                onnx_times = _timings(
                    partial(predictor.session.run, ["logits"], {"images": array}),
                    warmup,
                    iterations,
                )
            with path.open("rb") as stream:
                model_hash = hashlib.file_digest(stream, "sha256").hexdigest()
            report = {
                "metadata": metadata,
                "model_sha256": model_hash,
                "model_size_bytes": path.stat().st_size,
                "onnx_checker_passed": True,
                "parity": parity,
                "benchmark": {
                    "scope": "CPU forward pass only; excludes loading and image preprocessing",
                    "batch_size": 1,
                    "threads": threads,
                    "warmup": warmup,
                    "iterations": iterations,
                    "pytorch": pytorch_times,
                    "onnxruntime": onnx_times,
                    "speedup": pytorch_times["mean_ms"] / onnx_times["mean_ms"],
                },
                "environment": {
                    "platform": platform.platform(),
                    "processor": platform.processor(),
                    "python": platform.python_version(),
                    "torch": str(torch.__version__),
                    "onnx": onnx.__version__,
                    "onnxruntime": ort.__version__,
                    "provider": "CPUExecutionProvider",
                },
            }
            (staging / "export_report.json").write_text(
                json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
            # Release Windows file handles before moving the completed artifact directory.
            del predictor
            staging.rename(output_dir)
            return report
    finally:
        torch.set_num_threads(previous_threads)
