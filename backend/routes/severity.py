"""
POST /severity
Accepts a base64 heatmap or raw heatmap array,
returns severity score 0-100 with classification.
"""
import base64
import numpy as np
import logging
import cv2

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from utils.gradcam import compute_severity

log    = logging.getLogger("radiosight")
router = APIRouter()


class SeverityRequest(BaseModel):
    heatmap_base64 : str            # base64 PNG of heatmap from /predict response
    threshold      : float = 0.5   # activation threshold


class SeverityResult(BaseModel):
    severity       : float          # 0–100
    level          : str            # "Minimal" / "Mild" / "Moderate" / "Severe"
    afraction      : float          # proportion of activated pixels
    imean          : float          # mean activation intensity
    interpretation : str


def severity_level(score: float) -> tuple[str, str]:
    if score < 20:
        return "Minimal", "Very small area of concern detected."
    elif score < 40:
        return "Mild", "Moderate localized area of concern."
    elif score < 65:
        return "Moderate", "Significant area of abnormality detected. Clinical review recommended."
    else:
        return "Severe", "Large area of abnormality. Urgent clinical attention recommended."


@router.post("", response_model=SeverityResult)
async def severity(req: SeverityRequest):
    """
    Compute severity score from a Grad-CAM heatmap.
    Accepts base64-encoded PNG heatmap image.
    """
    try:
        img_bytes = base64.b64decode(req.heatmap_base64)
        nparr     = np.frombuffer(img_bytes, np.uint8)
        overlay   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if overlay is None:
            raise ValueError("Could not decode image")

        # Convert to grayscale heatmap [0, 1]
        gray    = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
        heatmap = gray.astype(np.float32) / 255.0

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid heatmap: {e}")

    # ── Compute metrics ───────────────────────────────────────────
    activated  = (heatmap >= req.threshold).astype(np.float32)
    afraction  = float(activated.mean())
    imean      = float(heatmap.mean())
    score      = float(np.clip(100.0 * (0.7 * afraction + 0.3 * imean), 0, 100))
    level, interp = severity_level(score)

    log.info(f"Severity: {score:.1f} ({level}) | Afrac={afraction:.3f} | Imean={imean:.3f}")

    return SeverityResult(
        severity       = round(score, 2),
        level          = level,
        afraction      = round(afraction, 4),
        imean          = round(imean, 4),
        interpretation = interp,
    )
