"""
POST /predict
Accepts a chest X-ray image, returns:
  - disease classification (14 labels) with probabilities
  - Grad-CAM heatmap URL
  - 1280-d embedding (for client-side similarity if needed)
  - severity score
"""
import json
import os
import gc
import base64
import logging
import numpy as np
import torch
from typing import List, Dict

from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel

from utils.model_loader  import get_model, get_device
from utils.gradcam       import classifier_weight_cam, overlay_heatmap, compute_severity
from utils.preprocess    import preprocess_image
from utils.firebase_utils import upload_heatmap

log    = logging.getLogger("radiosight")
router = APIRouter()

# Load class names once
_CLASS_NAMES_PATH = os.getenv("CLASS_NAMES_PATH", "./data/class_names.json")
try:
    with open(_CLASS_NAMES_PATH) as f:
        CLASS_NAMES: list[str] = json.load(f)
except FileNotFoundError:
    CLASS_NAMES = [
        'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
        'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
        'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
        'Pleural_Thickening', 'Hernia'
    ]


class PredictionResult(BaseModel):
    prediction      : str
    is_normal       : bool
    probabilities   : Dict[str, float]
    top_findings    : List[Dict]
    heatmap_url     : str
    heatmap_base64  : str
    embedding       : List[float]
    severity        : float


@router.post("", response_model=PredictionResult)
async def predict(file: UploadFile = File(...)):
    """
    Upload a chest X-ray image (PNG/JPG) and get full diagnostic analysis.
    """
    model  = get_model()
    device = get_device()

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Place model.pth in backend/models/ and restart."
        )

    # ── Read & validate file ──────────────────────────────────────
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    try:
        tensor, display_img = preprocess_image(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not process image: {e}")

    # ── Model inference ───────────────────────────────────────────
    model.eval()
    input_tensor = tensor.to(device)
    captured = {}

    def capture_activations(_module, _input, output):
        captured["activations"] = output.detach()

    hook = model.features[-1].register_forward_hook(capture_activations)
    with torch.no_grad():
        logits, embedding = model(input_tensor)
        hook.remove()
        probs_tensor      = torch.sigmoid(logits)[0]  # (14,)
        probs             = probs_tensor.cpu().numpy()
        embedding_list    = embedding[0].cpu().numpy().tolist()

    # ── Probabilities dict ────────────────────────────────────────
    prob_dict = {cls: float(round(float(p), 4)) for cls, p in zip(CLASS_NAMES, probs)}

    # ── Top findings (threshold 0.3 for medical sensitivity) ─────
    THRESHOLD     = 0.3
    top_findings  = [
        {"disease": cls, "probability": float(round(float(p), 4))}
        for cls, p in sorted(zip(CLASS_NAMES, probs), key=lambda x: -x[1])
        if p >= THRESHOLD
    ][:3]

    # ── Primary prediction ────────────────────────────────────────
    max_idx    = int(np.argmax(probs))
    max_prob   = float(probs[max_idx])
    is_normal  = max_prob < THRESHOLD
    prediction = "Normal" if is_normal else CLASS_NAMES[max_idx]

    # ── Grad-CAM heatmap ─────────────────────────────────────────
    target_idx = None if is_normal else max_idx
    heatmap = classifier_weight_cam(
        captured["activations"],
        model.classifier,
        target_idx if target_idx is not None else max_idx,
        display_img.shape[:2],
    )
    overlay    = overlay_heatmap(display_img, heatmap)

    # ── Severity score ────────────────────────────────────────────
    severity   = compute_severity(heatmap)

    # ── Upload heatmap to Firebase ────────────────────────────────
    heatmap_url    = upload_heatmap(overlay, prefix="pred")

    # ── Base64 fallback (for local use without Firebase) ─────────
    import cv2, io
    _, png_buf     = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    heatmap_b64    = base64.b64encode(png_buf.tobytes()).decode("utf-8")

    log.info(
        f"Predict: {prediction} ({max_prob:.3f}) | "
        f"Severity: {severity:.1f} | "
        f"Top: {[f['disease'] for f in top_findings]}"
    )

    del input_tensor, logits, embedding, probs_tensor, captured
    gc.collect()

    return PredictionResult(
        prediction     = prediction,
        is_normal      = is_normal,
        probabilities  = prob_dict,
        top_findings   = top_findings,
        heatmap_url    = heatmap_url,
        heatmap_base64 = heatmap_b64,
        embedding      = embedding_list,
        severity       = severity,
    )
