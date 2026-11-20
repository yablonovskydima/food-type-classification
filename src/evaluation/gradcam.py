import torch
from torch import nn
from torch.nn import functional as F


def grad_cam(
    model: nn.Module,
    inputs: torch.Tensor,
    target_layer: nn.Module,
    class_index: int | None = None,
) -> torch.Tensor:
    if inputs.ndim != 4 or inputs.shape[0] != 1:
        raise ValueError("Grad-CAM expects a single NCHW image")
    activations = []

    def capture(_module, _inputs, output):
        activations.append(output)

    handle = target_layer.register_forward_hook(capture)
    was_training = model.training
    try:
        model.eval()
        with torch.enable_grad():
            logits = model(inputs.detach().clone().requires_grad_(True))
            index = int(logits.argmax(dim=1).item()) if class_index is None else class_index
            if not 0 <= index < logits.shape[1]:
                raise ValueError("Invalid Grad-CAM class index")
            if len(activations) != 1 or activations[0].ndim != 4:
                raise ValueError("Target layer must produce one NCHW activation tensor")
            activation = activations[0]
            gradients = torch.autograd.grad(logits[0, index], activation)[0]
            weights = gradients.mean(dim=(2, 3), keepdim=True)
            cam = (weights * activation).sum(dim=1, keepdim=True).relu()
            cam = F.interpolate(cam, size=inputs.shape[-2:], mode="bilinear", align_corners=False)
            cam = cam[0, 0].detach().cpu()
            return cam / cam.max().clamp_min(1e-12)
    finally:
        handle.remove()
        model.train(was_training)
