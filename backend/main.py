"""SatQuery AI FastAPI entry point — Phase 1 sovereign runtime (SIH26167).

CORS, versioned API gateway, and air-gapped weight-registry probes live here.
ML inference graphs and SQL/PostGIS sessions are intentionally not constructed
in this phase.
"""

from __future__ import annotations

import os
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

try:
    from backend.api.routes_health import router as health_router
except ImportError:  # running with PYTHONPATH=<repo>/backend
    from api.routes_health import router as health_router


def _cors_origins() -> List[str]:
    raw = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173")
    origins = [item.strip() for item in raw.split(",") if item.strip()]
    return origins or ["http://localhost:3000"]


app = FastAPI(
    title="SatQuery AI",
    description=(
        "Agentic vision-language assistant for ISRO Earth Observation analysis "
        "(Problem Statement SIH26167). Sovereign, on-premise, air-gapped GPU deployments."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router, prefix="/api/v1")


@app.get("/", tags=["meta"])
async def root() -> dict[str, str]:
    return {
        "service": "SatQuery AI",
        "problem_statement": "SIH26167",
        "phase": "1-sovereign-runtime",
        "docs": "/docs",
        "health": "/api/v1/health",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("SATQUERY_ENV", "development") == "development",
    )
