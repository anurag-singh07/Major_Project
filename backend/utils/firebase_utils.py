"""
Firebase Storage Utility — uploads images/heatmaps and returns public URL.
"""
import os
import io
import uuid
import logging
import numpy as np
from PIL import Image

log = logging.getLogger("radiosight")

_storage_bucket = None


def _get_bucket():
    global _storage_bucket
    if _storage_bucket is not None:
        return _storage_bucket

    creds_path  = os.getenv("FIREBASE_CREDENTIALS_PATH", "./firebase_credentials.json")
    bucket_name = os.getenv("FIREBASE_STORAGE_BUCKET", "")

    if not os.path.exists(creds_path) or not bucket_name:
        log.warning("Firebase credentials not configured — using local fallback.")
        return None

    try:
        import firebase_admin
        from firebase_admin import credentials, storage

        if not firebase_admin._apps:
            cred = credentials.Certificate(creds_path)
            firebase_admin.initialize_app(cred, {"storageBucket": bucket_name})

        _storage_bucket = storage.bucket()
        log.info(f"✅ Firebase Storage connected: {bucket_name}")
    except Exception as e:
        log.error(f"Firebase init failed: {e}")
        return None

    return _storage_bucket


def upload_heatmap(heatmap_array: np.ndarray, prefix: str = "heatmap") -> str:
    """
    Upload a heatmap numpy array to Firebase Storage.
    Returns public download URL, or empty string if Firebase not configured.

    Args:
        heatmap_array : (H, W, 3) uint8 RGB overlay image
        prefix        : filename prefix
    Returns:
        url : str
    """
    bucket = _get_bucket()
    if bucket is None:
        return ""

    try:
        filename    = f"heatmaps/{prefix}_{uuid.uuid4().hex[:8]}.png"
        pil_img     = Image.fromarray(heatmap_array.astype(np.uint8))
        buf         = io.BytesIO()
        pil_img.save(buf, format="PNG")
        buf.seek(0)

        blob = bucket.blob(filename)
        blob.upload_from_file(buf, content_type="image/png")
        blob.make_public()
        url  = blob.public_url
        log.info(f"Uploaded heatmap → {url}")
        return url

    except Exception as e:
        log.error(f"Firebase upload failed: {e}")
        return ""


def upload_xray(image_bytes: bytes, patient_id: str = "unknown") -> str:
    """Upload raw X-ray image bytes and return public URL."""
    bucket = _get_bucket()
    if bucket is None:
        return ""

    try:
        filename = f"xrays/{patient_id}_{uuid.uuid4().hex[:8]}.png"
        buf      = io.BytesIO(image_bytes)

        blob = bucket.blob(filename)
        blob.upload_from_file(buf, content_type="image/png")
        blob.make_public()
        return blob.public_url

    except Exception as e:
        log.error(f"Firebase X-ray upload failed: {e}")
        return ""
