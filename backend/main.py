"""SatQuery AI FastAPI entry point — SIH26167.

Initializes CORS, lifespan (DB ping), and the versioned API gateway.
Inference and geospatial workers are never constructed here; routers
delegate to SatQueryController.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.core.config import settings
from app.database.session import engine
from app.utils.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info(
        "satquery_startup",
        extra={
            "env": settings.SATQUERY_ENV,
            "inference_backend": settings.INFERENCE_BACKEND,
            "models_dir": str(settings.LOCAL_MODELS_DIR),
        },
    )
    try:
        with engine.connect() as conn:
            conn.exec_driver_sql("SELECT 1")
        logger.info("database_reachable")
    except Exception as exc:  # noqa: BLE001 — boot must not crash air-gapped demos
        logger.warning("database_unreachable", extra={"error": str(exc)})
    yield
    engine.dispose()
    logger.info("satquery_shutdown")


app = FastAPI(
    title="SatQuery AI",
    description=(
        "Agentic vision-language assistant for ISRO Earth Observation analysis "
        "(Problem Statement SIH26167). Sovereign, on-premise GPU deployments."
    ),
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")


@app.get("/", tags=["meta"])
async def root() -> dict[str, str]:
    return {
        "service": "SatQuery AI",
        "problem_statement": "SIH26167",
        "docs": "/docs",
        "health": "/api/v1/health",
    }
