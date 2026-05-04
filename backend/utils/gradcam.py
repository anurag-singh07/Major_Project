"""
Grad-CAM Utility — generates heatmap from EfficientNet-V2-S.
Called by the /predict endpoint after classification.
"""
import numpy as np
import torch
import cv2
import logging

log = logging.getLogger("radiosight")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


class GradCAM:
    """
    Grad-CAM for RadiosightModel.
    Hooks the last convolutional block of EfficientNet-V2-S features.
    """
    def __init__(self, model):
        self.model       = model
        self.gradients   = None
        self.activations = None

        target_layer = model.features[-1]
        self._forward_handle = target_layer.register_forward_hook(self._forward_hook)
        self._backward_handle = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach().clone()

    def generate(self, image_tensor: torch.Tensor,
                 class_idx: int | None = None,
                 device: str = "cpu") -> np.ndarray:
        """
        Args:
            image_tensor : (1, 3, H, W) preprocessed tensor
            class_idx    : target class (None = highest probability class)
        Returns:
            heatmap : (H, W) float32 [0, 1]
        """
        self.model.eval()
        image_tensor = image_tensor.to(device)

        self.model.zero_grad(set_to_none=True)
        logits, _ = self.model(image_tensor)
        probs      = torch.sigmoid(logits)

        if class_idx is None:
            class_idx = probs.argmax().item()

        target = logits[0, class_idx]
        target.backward()

        grads   = self.gradients[0]                # (C, H', W')
        acts    = self.activations[0]              # (C, H', W')
        weights = grads.mean(dim=(1, 2))           # (C,)

        cam = torch.zeros(acts.shape[1:], device=device)
        for i, w in enumerate(weights):
            cam += w * acts[i]

        cam = torch.relu(cam).cpu().numpy()

        # Normalize
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)

        # Resize to match the preprocessed display image.
        height, width = image_tensor.shape[-2:]
        cam = cv2.resize(cam.astype(np.float32), (width, height))
        return cam

    def close(self):
        self._forward_handle.remove()
        self._backward_handle.remove()


def overlay_heatmap(original_img: np.ndarray,
                    heatmap: np.ndarray,
                    alpha: float = 0.5) -> np.ndarray:
    """
    Blend Grad-CAM heatmap onto original image.
    Args:
        original_img : (H, W, 3) uint8 RGB
        heatmap      : (H, W) float [0, 1]
    Returns:
        overlay : (H, W, 3) uint8 RGB
    """
    hmap_uint8 = np.uint8(255 * heatmap)
    colormap   = cv2.applyColorMap(hmap_uint8, cv2.COLORMAP_JET)
    colormap   = cv2.cvtColor(colormap, cv2.COLOR_BGR2RGB)
    overlay    = np.uint8(alpha * colormap + (1 - alpha) * original_img)
    return overlay


def compute_severity(heatmap: np.ndarray, threshold: float = 0.5) -> float:
    """
    Severity = 100 * (0.7 * Afraction + 0.3 * Imean)
    Returns float in [0, 100].
    """
    activated  = (heatmap >= threshold).astype(np.float32)
    Afraction  = float(activated.mean())
    Imean      = float(heatmap.mean())
    severity   = 100.0 * (0.7 * Afraction + 0.3 * Imean)
    return float(np.clip(severity, 0, 100))
