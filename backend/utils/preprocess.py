"""
Preprocessing utility — converts uploaded image to model-ready tensor.
"""
import io
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.Grayscale(num_output_channels=3),   # chest X-rays are grayscale
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def preprocess_image(image_bytes: bytes) -> tuple[torch.Tensor, np.ndarray]:
    """
    Args:
        image_bytes : raw bytes from uploaded file
    Returns:
        tensor      : (1, 3, 512, 512) — for model input
        display_img : (512, 512, 3) uint8 RGB — for overlay
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)

    display_img = np.array(img, dtype=np.uint8)

    tensor = _transform(img).unsqueeze(0)  # (1, 3, H, W)
    return tensor, display_img
