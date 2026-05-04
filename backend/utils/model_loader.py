"""
Model Loader — loads EfficientNet-V2-S from .pth file on startup.
Singleton pattern: model loads once, reused on every request.
"""
import os
import gc
import logging
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional

log = logging.getLogger("radiosight")

# ── Model architecture (must match Kaggle training code exactly) ──
class RadiosightModel(nn.Module):
    def __init__(self, num_classes: int = 14, dropout: float = 0.4):
        super().__init__()
        backbone        = models.efficientnet_v2_s(weights=None)
        self.features   = backbone.features
        self.avgpool    = backbone.avgpool
        self.embedding_dim = 1280
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.embedding_dim, num_classes),
        )

    def forward(self, x):
        feat      = self.features(x)
        pooled    = self.avgpool(feat)
        embedding = torch.flatten(pooled, 1)
        logits    = self.classifier(embedding)
        return logits, embedding


# ── Singleton state ───────────────────────────────────────────────
_model: Optional[RadiosightModel] = None
_device: str = "cpu"


def load_model_on_startup():
    global _model, _device
    model_path = os.getenv("MODEL_PATH", "./models/model.pth")
    _device    = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(1)

    if not os.path.exists(model_path):
        log.warning(
            f"⚠ Model not found at '{model_path}'. "
            "Place model.pth from Kaggle training output here. "
            "Endpoints will return 503 until model is loaded."
        )
        return

    log.info(f"Loading model from: {model_path}  (device={_device})")
    _model = RadiosightModel(num_classes=14)
    checkpoint = torch.load(model_path, map_location=_device, weights_only=True)

    # Handle both formats:
    # 1) Pure state_dict (torchvision pretrained)
    # 2) Wrapped checkpoint {'model_state_dict': ..., 'epoch': ...} (Kaggle training)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state = checkpoint['model_state_dict']
        epoch = checkpoint.get('epoch', '?')
        log.info(f"  Checkpoint format: wrapped (epoch={epoch})")
    else:
        state = checkpoint
        log.info("  Checkpoint format: raw state_dict")

    # strict=False: allows pretrained backbone keys even if classifier differs slightly
    missing, unexpected = _model.load_state_dict(state, strict=False)
    if unexpected:
        log.warning(f"  Unexpected keys (skipped): {len(unexpected)}")
    if missing:
        log.warning(f"  Missing keys (random init): {len(missing)}")

    del checkpoint
    del state
    gc.collect()

    _model.to(_device)
    _model.eval()
    log.info("✅ Model loaded successfully.")


def get_model() -> Optional[RadiosightModel]:
    return _model


def get_device() -> str:
    return _device
