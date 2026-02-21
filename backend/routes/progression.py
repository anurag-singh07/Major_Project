"""
POST /progression
Tracks disease severity over time for a patient.
Stores and retrieves scan records from MongoDB.
"""
import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

log    = logging.getLogger("radiosight")
router = APIRouter()


# ── MongoDB helper (lazy import) ──────────────────────────────────
def _get_collection():
    import os
    from pymongo import MongoClient
    uri     = os.getenv("MONGO_URI", "")
    db_name = os.getenv("MONGO_DB_NAME", "radiosight")
    if not uri:
        return None
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    return client[db_name]["progressions"]


# ── Schema ────────────────────────────────────────────────────────
class AddScanRequest(BaseModel):
    patient_id    : str
    scan_id       : str                   # e.g. filename or UUID
    prediction    : str
    severity      : float
    embedding     : list[float]           # 1280-d
    heatmap_url   : str = ""
    timestamp     : Optional[str] = None  # ISO format; defaults to now


class ScanRecord(BaseModel):
    scan_id       : str
    prediction    : str
    severity      : float
    heatmap_url   : str
    timestamp     : str
    delta_severity: Optional[float] = None
    embedding_shift: Optional[float] = None


class ProgressionResult(BaseModel):
    patient_id    : str
    scans         : list[ScanRecord]
    trend         : str    # "Improving" / "Stable" / "Worsening" / "Insufficient data"
    total_scans   : int


@router.post("/add", status_code=201)
async def add_scan(req: AddScanRequest):
    """Save a new scan record for a patient to MongoDB."""
    col = _get_collection()
    if col is None:
        raise HTTPException(
            status_code=503,
            detail="MongoDB not configured. Set MONGO_URI in .env"
        )

    ts = req.timestamp or datetime.now(timezone.utc).isoformat()
    doc = {
        "patient_id" : req.patient_id,
        "scan_id"    : req.scan_id,
        "prediction" : req.prediction,
        "severity"   : req.severity,
        "embedding"  : req.embedding,
        "heatmap_url": req.heatmap_url,
        "timestamp"  : ts,
    }
    col.insert_one(doc)
    log.info(f"Progression: saved scan for patient={req.patient_id}")
    return {"status": "saved", "patient_id": req.patient_id, "scan_id": req.scan_id}


@router.get("/{patient_id}", response_model=ProgressionResult)
async def get_progression(patient_id: str):
    """
    Retrieve all scans for a patient, sorted by time.
    Computes delta_severity and embedding_shift between consecutive scans.
    """
    col = _get_collection()
    if col is None:
        raise HTTPException(status_code=503, detail="MongoDB not configured.")

    import numpy as np
    docs = list(col.find(
        {"patient_id": patient_id},
        {"_id": 0}
    ).sort("timestamp", 1))

    if not docs:
        raise HTTPException(status_code=404, detail=f"No scans found for patient '{patient_id}'")

    scans = []
    for i, doc in enumerate(docs):
        delta_sev    = None
        embed_shift  = None

        if i > 0:
            prev         = docs[i - 1]
            delta_sev    = round(doc["severity"] - prev["severity"], 2)
            curr_emb     = np.array(doc["embedding"],  dtype=np.float32)
            prev_emb     = np.array(prev["embedding"], dtype=np.float32)
            embed_shift  = float(round(float(np.linalg.norm(curr_emb - prev_emb)), 4))

        scans.append(ScanRecord(
            scan_id        = doc["scan_id"],
            prediction     = doc["prediction"],
            severity       = doc["severity"],
            heatmap_url    = doc.get("heatmap_url", ""),
            timestamp      = doc["timestamp"],
            delta_severity = delta_sev,
            embedding_shift= embed_shift,
        ))

    # ── Trend analysis ─────────────────────────────────────────────
    if len(scans) < 2:
        trend = "Insufficient data"
    else:
        deltas   = [s.delta_severity for s in scans if s.delta_severity is not None]
        avg_delta = sum(deltas) / len(deltas)
        if avg_delta < -3:
            trend = "Improving"
        elif avg_delta > 3:
            trend = "Worsening"
        else:
            trend = "Stable"

    log.info(f"Progression: patient={patient_id} | scans={len(scans)} | trend={trend}")

    return ProgressionResult(
        patient_id  = patient_id,
        scans       = scans,
        trend       = trend,
        total_scans = len(scans),
    )
