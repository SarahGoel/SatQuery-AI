"""SatQuery AI FastAPI gateway — health probe and multipart GeoTIFF uploads."""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path

import rasterio
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SatQueryBase")

ALLOWED_GEOTIFF_SUFFIXES = {".tif", ".tiff", ".gtiff"}
UPLOAD_ROOT = Path(os.getenv("UPLOAD_DIR", "./uploads"))

app = FastAPI(
    title="SatQuery AI API Gateway",
    description="Sovereign Multi-Sensor Agentic Remote Sensing Gateway",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _gdal_version() -> str:
    if hasattr(rasterio, "gdal_version"):
        return str(rasterio.gdal_version())
    return str(getattr(rasterio, "__gdal_version__", "unknown"))


def _include_existing_routers() -> None:
    """Keep Phase 1 air-gap health and Phase 2 analyze routes mounted."""
    try:
        from api.routes_health import router as airgap_health_router
    except ImportError:
        from backend.api.routes_health import router as airgap_health_router
    app.include_router(airgap_health_router, prefix="/api/v1")

    try:
        from app.api.endpoints.analyze import router as analyze_router

        app.include_router(analyze_router, prefix="/api/v1/satquery", tags=["satquery"])
    except ImportError:
        logger.warning("analyze router not mounted")


_include_existing_routers()


@app.get("/health")
async def health_check():
    gdal_ver = _gdal_version()
    cuda_avail = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU Only"

    return {
        "status": "healthy",
        "spatial_drivers": f"GDAL {gdal_ver} / Rasterio {rasterio.__version__}",
        "gpu_acceleration": {
            "cuda_available": cuda_avail,
            "device": device_name,
        },
    }


@app.post("/upload")
async def upload_geotiffs(
    files: list[UploadFile] = File(..., description="One or more GeoTIFF scenes"),
    query: str | None = Form(default=None),
):
    """Accept multipart GeoTIFF uploads from the React client."""
    if not files:
        raise HTTPException(status_code=400, detail="At least one GeoTIFF is required")

    batch_dir = UPLOAD_ROOT / uuid.uuid4().hex
    batch_dir.mkdir(parents=True, exist_ok=True)
    saved: list[dict[str, str | int]] = []

    for upload in files:
        suffix = Path(upload.filename or "scene.tif").suffix.lower()
        if suffix not in ALLOWED_GEOTIFF_SUFFIXES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported raster suffix: {suffix}",
            )
        target = batch_dir / f"{uuid.uuid4().hex}{suffix}"
        nbytes = 0
        with target.open("wb") as handle:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                nbytes += len(chunk)
                handle.write(chunk)
        if nbytes == 0:
            target.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail="Empty raster upload")
        saved.append(
            {
                "filename": upload.filename or target.name,
                "path": str(target),
                "bytes": nbytes,
            }
        )
        logger.info("geotiff_uploaded", extra={"path": str(target), "bytes": nbytes})

    return {
        "status": "accepted",
        "query": query,
        "count": len(saved),
        "files": saved,
    }


@app.get("/", tags=["meta"])
async def root() -> dict[str, str]:
    return {
        "service": "SatQuery AI API Gateway",
        "health": "/health",
        "upload": "/upload",
        "analyze": "/api/v1/satquery/analyze",
        "docs": "/docs",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("SATQUERY_ENV", "development") == "development",
    )
