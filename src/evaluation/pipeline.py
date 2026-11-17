import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.data.constants import DEFAULT_IMAGE_SIZE
from src.data.dataset import get_dataset
from src.data.transforms import get_eval_geometry_transforms
from src.evaluation.gradcam import grad_cam
from src.evaluation.metrics import compute_metrics
from src.evaluation.reports import display_name, save_gradcam_plot, save_metric_plots
from src.inference.predictor import FoodPredictor


def evaluate_checkpoint(
    checkpoint: str | Path,
    data_dir: str | Path,
    output_dir: str | Path,
    *,
    split: str = "validation",
    image_size: int = DEFAULT_IMAGE_SIZE,
    use_imagenet_stats: bool = False,
    batch_size: int = 32,
    num_workers: int = 0,
    device: str | None = None,
    gradcam_examples: int = 8,
) -> dict[str, Any]:
    if split not in {"validation", "evaluation"}:
        raise ValueError("Split must be validation or evaluation")
    if image_size <= 0 or batch_size <= 0 or num_workers < 0 or gradcam_examples < 0:
        raise ValueError("Invalid image size, batch size, worker count, or Grad-CAM count")
    checkpoint = Path(checkpoint)
    output_dir = Path(output_dir)
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(f"Report output must be a new or empty directory: {output_dir}")
    predictor = FoodPredictor(
        checkpoint_path=checkpoint,
        device=device,
        image_size=image_size,
        use_imagenet_stats=use_imagenet_stats,
    )
    dataset = get_dataset(data_dir, split=split, transform=predictor.transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=predictor.device.type == "cuda",
    )
    probabilities, targets = [], []
    with torch.inference_mode():
        for images, labels in loader:
            logits = predictor.model(images.to(predictor.device))
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
            targets.append(labels.numpy())
    scores = np.concatenate(probabilities)
    labels = np.concatenate(targets)
    metrics = compute_metrics(labels, scores, predictor.classes)
    with checkpoint.open("rb") as stream:
        checkpoint_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    metrics["run"] = {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "data_dir": str(Path(data_dir).resolve()),
        "split": split,
        "image_size": image_size,
        "normalization": "imagenet" if use_imagenet_stats else "food11",
        "device": str(predictor.device),
        "torch_version": str(torch.__version__),
        "gradcam_examples_requested": gradcam_examples,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "predictions.npz", targets=labels, probabilities=scores)
    predictions = scores.argmax(axis=1)
    confidences = scores[np.arange(len(scores)), predictions]
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image", "true_class", "predicted_class", "confidence"])
        for (path, label), prediction, confidence in zip(
            dataset.samples, predictions, confidences, strict=True
        ):
            writer.writerow(
                [
                    str(Path(path).relative_to(Path(data_dir))),
                    predictor.classes[label],
                    predictor.classes[prediction],
                    float(confidence),
                ]
            )
    save_metric_plots(metrics, output_dir)

    candidates = np.flatnonzero(predictions != labels)
    if not len(candidates):
        candidates = np.arange(len(labels))
    selected = candidates[np.argsort(-confidences[candidates], kind="stable")][:gradcam_examples]
    examples = []
    if len(selected):
        cam_dir = output_dir / "gradcam"
        cam_dir.mkdir()
        geometry = get_eval_geometry_transforms(image_size=image_size)
        target_layer = predictor.model.layer4[-1]
        for index in selected:
            path, label = dataset.samples[index]
            with Image.open(path) as source:
                image = source.convert("RGB")
            inputs = predictor.transform(image).unsqueeze(0).to(predictor.device)
            prediction = int(predictions[index])
            true_name = display_name(predictor.classes[label])
            predicted_name = display_name(predictor.classes[prediction])
            heatmaps = [
                (
                    f"Grad-CAM: predicted ({predicted_name})",
                    grad_cam(predictor.model, inputs, target_layer, prediction).numpy(),
                )
            ]
            if prediction != label:
                heatmaps.append(
                    (
                        f"Grad-CAM: true ({true_name})",
                        grad_cam(predictor.model, inputs, target_layer, label).numpy(),
                    )
                )
            filename = f"{int(index):05d}_{Path(path).stem}.png"
            title = f"True: {true_name} | Predicted: {predicted_name} ({confidences[index]:.1%})"
            save_gradcam_plot(geometry(image), heatmaps, cam_dir / filename, title)
            examples.append(
                {
                    "image": str(Path(path).relative_to(Path(data_dir))),
                    "true_class": predictor.classes[label],
                    "predicted_class": predictor.classes[prediction],
                    "confidence": float(confidences[index]),
                    "plot": f"gradcam/{filename}",
                }
            )
    metrics["gradcam_examples"] = examples
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return metrics
