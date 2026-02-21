"""
POST /similarity
Accepts a 1280-d embedding vector, returns top-5 similar cases
using rank-fusion of cosine similarity and Euclidean distance.
"""
import numpy as np
import logging
from typing import Optional, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from utils.embedding_store import get_matrix, get_image_ids, get_labels

log    = logging.getLogger("radiosight")
router = APIRouter()


class SimilarityRequest(BaseModel):
    embedding : List[float]          # 1280-d vector from /predict
    top_k     : int = 5              # Number of similar cases to return
    exclude_id: Optional[str] = None # Optional: exclude the query image itself


class SimilarCase(BaseModel):
    image_id        : str
    label           : List[str]
    cosine_sim      : float
    euclidean_dist  : float
    fusion_score    : float


class SimilarityResult(BaseModel):
    similar_cases : List[SimilarCase]
    query_label   : str


@router.post("", response_model=SimilarityResult)
async def similarity(req: SimilarityRequest):
    """
    Find top-K most similar X-ray cases from training database.
    Uses rank fusion: FinalRank = 0.6 * cosine_rank + 0.4 * euclidean_rank
    """
    matrix    = get_matrix()
    image_ids = get_image_ids()
    labels    = get_labels()

    if matrix is None or len(matrix) == 0:
        raise HTTPException(
            status_code=503,
            detail="Embedding database not loaded. Place embeddings.pkl in backend/data/ and restart."
        )

    if len(req.embedding) != 1280:
        raise HTTPException(
            status_code=400,
            detail=f"Embedding must be 1280-d, got {len(req.embedding)}-d."
        )

    query = np.array(req.embedding, dtype=np.float32)  # (1280,)

    # ── Cosine Similarity (vectorized) ────────────────────────────
    query_norm  = query / (np.linalg.norm(query) + 1e-8)
    db_norms    = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)
    cosine_sims = db_norms @ query_norm                 # (N,)

    # ── Euclidean Distance (vectorized) ──────────────────────────
    diffs        = matrix - query                       # (N, 1280)
    eucl_dists   = np.linalg.norm(diffs, axis=1)       # (N,)

    # ── Rank Fusion ───────────────────────────────────────────────
    # Rank 0 = best
    cosine_ranks = np.argsort(-cosine_sims)             # descending similarity
    eucl_ranks   = np.argsort(eucl_dists)               # ascending distance

    cosine_rank_arr = np.empty_like(cosine_ranks)
    eucl_rank_arr   = np.empty_like(eucl_ranks)
    cosine_rank_arr[cosine_ranks] = np.arange(len(cosine_ranks))
    eucl_rank_arr[eucl_ranks]     = np.arange(len(eucl_ranks))

    fusion_scores = 0.6 * cosine_rank_arr + 0.4 * eucl_rank_arr  # lower = better
    top_indices   = np.argsort(fusion_scores)

    # ── Build result ──────────────────────────────────────────────
    results = []
    added   = 0
    for idx in top_indices:
        img_id = image_ids[idx]
        if req.exclude_id and img_id == req.exclude_id:
            continue
        results.append(SimilarCase(
            image_id       = img_id,
            label          = labels[idx],
            cosine_sim     = float(round(float(cosine_sims[idx]), 4)),
            euclidean_dist = float(round(float(eucl_dists[idx]), 4)),
            fusion_score   = float(round(float(fusion_scores[idx]), 2)),
        ))
        added += 1
        if added >= req.top_k:
            break

    log.info(f"Similarity: returned top-{len(results)} cases")
    return SimilarityResult(
        similar_cases = results,
        query_label   = "Unknown",
    )
