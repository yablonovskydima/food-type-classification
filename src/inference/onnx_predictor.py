import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

METADATA_KEY = "food11"


class OnnxFoodPredictor:
    def __init__(self, model_path: str | Path, threads: int = 1) -> None:
        if threads < 1:
            raise ValueError("CPU thread count must be positive")
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        raw_metadata = self.session.get_modelmeta().custom_metadata_map.get(METADATA_KEY)
        if raw_metadata is None:
            raise ValueError("Missing Food-11 preprocessing metadata; use the project exporter")
        self.metadata = json.loads(raw_metadata)
        self._validate_metadata()
        self.classes = tuple(self.metadata["classes"])
        self.image_size = self.metadata["image_size"]
        self.mean = np.asarray(self.metadata["mean"], dtype=np.float32)
        self.std = np.asarray(self.metadata["std"], dtype=np.float32)

    def _validate_metadata(self) -> None:
        data = self.metadata
        if not isinstance(data, dict) or data.get("schema_version") != 1:
            raise ValueError("Unsupported Food-11 metadata schema")
        classes = data.get("classes", [])
        if (
            not isinstance(classes, list)
            or len(classes) != 11
            or not all(isinstance(name, str) and name for name in classes)
            or len(set(classes)) != 11
        ):
            raise ValueError("Food-11 metadata must contain 11 unique class names")
        size = data.get("image_size")
        resize = data.get("resize_size")
        if type(size) is not int or size < 32 or type(resize) is not int or resize < size:
            raise ValueError("Invalid Food-11 input dimensions")
        for key in ("mean", "std"):
            array = np.asarray(data.get(key), dtype=np.float32)
            if array.shape != (3,) or not np.isfinite(array).all():
                raise ValueError(f"Invalid Food-11 {key}")
            if key == "std" and (array <= 0).any():
                raise ValueError("Normalization standard deviations must be positive")
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if (
            len(inputs) != 1
            or inputs[0].name != "images"
            or inputs[0].type != "tensor(float)"
            or len(inputs[0].shape) != 4
            or inputs[0].shape[1:] != [3, size, size]
            or len(outputs) != 1
            or outputs[0].name != "logits"
            or len(outputs[0].shape) != 2
            or outputs[0].shape[1] != 11
        ):
            raise ValueError("ONNX input/output contract does not match Food-11 metadata")

    def preprocess(self, image: Image.Image) -> np.ndarray:
        image = image.convert("RGB")
        width, height = image.size
        shorter = self.metadata["resize_size"]
        if width <= height:
            resized = (shorter, int(shorter * height / width))
        else:
            resized = (int(shorter * width / height), shorter)
        image = image.resize(resized, Image.Resampling.BILINEAR)
        left = int(round((resized[0] - self.image_size) / 2))
        top = int(round((resized[1] - self.image_size) / 2))
        image = image.crop((left, top, left + self.image_size, top + self.image_size))
        pixels = np.asarray(image, dtype=np.float32) / np.float32(255)
        pixels = (pixels - self.mean) / self.std
        return np.ascontiguousarray(pixels.transpose(2, 0, 1)[None])

    def predict_logits(self, images: np.ndarray) -> np.ndarray:
        if (
            images.dtype != np.float32
            or images.ndim != 4
            or images.shape[0] < 1
            or images.shape[1:] != (3, self.image_size, self.image_size)
            or not np.isfinite(images).all()
        ):
            raise ValueError("Expected a finite float32 NCHW batch at the exported image size")
        logits = self.session.run(["logits"], {"images": images})[0]
        if logits.shape != (len(images), len(self.classes)) or not np.isfinite(logits).all():
            raise ValueError("Model returned invalid logits")
        return logits

    def predict_image(self, image: Image.Image | str | Path, top_k: int = 3) -> list[dict]:
        if isinstance(image, str | Path):
            with Image.open(image) as source:
                inputs = self.preprocess(source)
        elif isinstance(image, Image.Image):
            inputs = self.preprocess(image)
        else:
            raise TypeError("Expected a PIL image or image path")
        logits = self.predict_logits(inputs)[0]
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        indices = np.argsort(-probabilities, kind="stable")[: min(max(1, top_k), len(self.classes))]
        return [
            {
                "class": self.classes[index],
                "probability": round(float(probabilities[index]), 4),
                "index": int(index),
            }
            for index in indices
        ]
