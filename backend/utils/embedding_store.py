"""
Embedding Store — loads embeddings.pkl into memory on startup.
Used by the similarity engine for fast cosine + euclidean search.
"""
import os
import pickle
import logging
import numpy as np

log = logging.getLogger("radiosight")

# ── In-memory store ───────────────────────────────────────────────
_store: list[dict] | None = None      # list of {image_id, embedding, label, ...}
_matrix: np.ndarray | None = None     # (N, 1280) float32 matrix for fast search
_image_ids: list[str] = []
_labels: list[list[str]] = []


def load_embeddings_on_startup():
    global _store, _matrix, _image_ids, _labels
    path = os.getenv("EMBEDDINGS_PATH", "./data/embeddings.pkl")

    if not os.path.exists(path):
        log.warning(
            f"⚠ Embeddings not found at '{path}'. "
            "Place embeddings.pkl from Kaggle here. "
            "Similarity search will return empty results."
        )
        return

    log.info(f"Loading embeddings from: {path}")
    with open(path, "rb") as f:
        _store = pickle.load(f)

    _image_ids = [r["image_id"]  for r in _store]
    _labels    = [r["label"]     for r in _store]
    _matrix    = np.array(
        [r["embedding"] for r in _store], dtype=np.float32
    )  # (N, 1280)

    log.info(f"✅ Loaded {len(_store):,} embeddings  shape={_matrix.shape}")


def get_store() -> list[dict] | None:
    return _store


def get_matrix() -> np.ndarray | None:
    return _matrix


def get_image_ids() -> list[str]:
    return _image_ids


def get_labels() -> list[list[str]]:
    return _labels
