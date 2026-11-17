__all__ = ["FoodPredictor"]


def __getattr__(name: str):
    """Keep ONNX-only imports independent of PyTorch and torchvision."""
    if name == "FoodPredictor":
        from src.inference.predictor import FoodPredictor

        return FoodPredictor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
