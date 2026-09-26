"""SatQuery AI FastAPI gateway — health probe and multipart GeoTIFF uploads."""

from __future__ import annotations

import logging
import os
import uuid
import sys
from pathlib import Path

import rasterio
import torch
from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SatQueryBase")

ALLOWED_GEOTIFF_SUFFIXES = {".tif", ".tiff", ".gtiff"}
UPLOAD_ROOT = Path(os.getenv("UPLOAD_DIR", "./uploads"))

app = FastAPI(
    title="SatQuery AI API Gateway",
    description="Sovereign Multi-Sensor Agentic Remote Sensing Gateway",
    version="1.0.0",
)

# Origins explicitly including frontend on localhost:3000 and 127.0.0.1:3000
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]

env_cors = os.getenv("CORS_ORIGINS")
if env_cors:
    for o in env_cors.split(","):
        cleaned = o.strip()
        if cleaned and cleaned not in origins:
            origins.append(cleaned)

# CORSMiddleware added BEFORE router inclusions
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


def _cors_headers_for_request(request: Request) -> dict[str, str]:
    """Ensure error and exception responses retain valid CORS headers for browser requests."""
    req_origin = request.headers.get("origin")
    headers = {
        "Access-Control-Allow-Credentials": "true",
        "Access-Control-Allow-Methods": "*",
        "Access-Control-Allow-Headers": "*",
    }
    if req_origin and (
        req_origin in origins
        or req_origin.startswith("http://localhost:")
        or req_origin.startswith("http://127.0.0.1:")
    ):
        headers["Access-Control-Allow-Origin"] = req_origin
    elif origins:
        headers["Access-Control-Allow-Origin"] = origins[0]
    return headers


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
        headers=_cors_headers_for_request(request),
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()},
        headers=_cors_headers_for_request(request),
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled server exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)},
        headers=_cors_headers_for_request(request),
    )


def _gdal_version() -> str:
    if hasattr(rasterio, "__gdal_version__"):
        return str(rasterio.__gdal_version__)
    if hasattr(rasterio, "gdal_version"):
        return str(rasterio.gdal_version())
    return "unknown"


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
    current_dir = Path(__file__).resolve().parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    if str(current_dir.parent) not in sys.path:
        sys.path.insert(0, str(current_dir.parent))
    try:
        from api.routes import router as query_router
    except (ImportError, ModuleNotFoundError):
        from backend.api.routes import router as query_router
    app.include_router(query_router, prefix="/api/v1", tags=["query"])
    app.include_router(query_router, tags=["query"])


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
    files: list[UploadFile] | None = File(default=None, description="One or more GeoTIFF scenes"),
    query: str | None = Form(default=None),
):
    """Accept multipart GeoTIFF uploads from the React client."""
    if not files:
        return {
            "status": "accepted",
            "query": query,
            "count": 0,
            "files": [],
        }

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
        "query": "/api/v1/query",
        "reports": "/api/v1/reports/{trace_id}",
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
