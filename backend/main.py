"""
Radiosight FastAPI Backend — Main Entry Point
Run: uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

from routes import predict, similarity, severity, progression
from utils.model_loader import load_model_on_startup
from utils.embedding_store import load_embeddings_on_startup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("radiosight")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup & shutdown events."""
    log.info("═" * 50)
    log.info("  Radiosight Backend Starting...")
    log.info("═" * 50)
    load_model_on_startup()
    load_embeddings_on_startup()
    log.info("✅ Backend ready! All systems go.")
    log.info("═" * 50)
    yield
    log.info("Backend shutting down...")


app = FastAPI(
    title="Radiosight API",
    description="Explainable Chest X-ray Diagnostic Support System",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS (allow Next.js frontend) ──────────────────────────────
import os
origins = [origin.strip() for origin in os.getenv("ALLOWED_ORIGINS", "*").split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials="*" not in origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ─────────────────────────────────────────────────────
app.include_router(predict.router,     prefix="/predict",     tags=["Predict"])
app.include_router(similarity.router,  prefix="/similarity",  tags=["Similarity"])
app.include_router(severity.router,    prefix="/severity",    tags=["Severity"])
app.include_router(progression.router, prefix="/progression", tags=["Progression"])


@app.get("/", tags=["Health"])
async def root():
    return {
        "status": "ok",
        "message": "Radiosight API is running",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
async def health():
    from utils.model_loader import get_model
    from utils.embedding_store import get_store
    model   = get_model()
    store   = get_store()
    return {
        "model_loaded"      : model is not None,
        "embeddings_loaded" : store is not None and len(store) > 0,
        "embedding_count"   : len(store) if store else 0,
    }
