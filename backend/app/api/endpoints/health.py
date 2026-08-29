"""System health: process, PostGIS, GPU, and local model volume."""

from __future__ import annotations

import shutil
from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.config import settings
from app.database.session import get_db

router = APIRouter()


def _gpu_metrics() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "cuda_available": False,
        "device_count": 0,
        "devices": [],
    }
    try:
        import torch

        payload["cuda_available"] = bool(torch.cuda.is_available())
        payload["device_count"] = int(torch.cuda.device_count()) if payload["cuda_available"] else 0
        for idx in range(payload["device_count"]):
            props = torch.cuda.get_device_properties(idx)
            payload["devices"].append(
                {
                    "index": idx,
                    "name": props.name,
                    "total_memory_bytes": int(props.total_memory),
                    "major": props.major,
                    "minor": props.minor,
                }
            )
    except Exception as exc:  # noqa: BLE001
        payload["error"] = str(exc)
    return payload


@router.get("/health")
async def health(db: Session = Depends(get_db)) -> dict[str, Any]:
    database_ok = False
    postgis_version = None
    try:
        row = db.execute(text("SELECT PostGIS_Version()")).scalar()
        postgis_version = row
        database_ok = True
    except Exception as exc:  # noqa: BLE001
        database_ok = False
        db_error = str(exc)
    else:
        db_error = None

    disk = shutil.disk_usage(settings.LOCAL_MODELS_DIR if settings.LOCAL_MODELS_DIR.exists() else ".")
    models_dir_present = settings.LOCAL_MODELS_DIR.exists()

    gpu = _gpu_metrics()
    status = "ok" if database_ok else "degraded"

    return {
        "status": status,
        "service": "satquery-backend",
        "problem_statement": "SIH26167",
        "environment": settings.SATQUERY_ENV,
        "inference_backend": settings.INFERENCE_BACKEND,
        "database": {
            "ok": database_ok,
            "postgis_version": postgis_version,
            "error": db_error,
        },
        "gpu": gpu,
        "gpu_available": gpu.get("cuda_available", False),
        "models": {
            "dir": str(settings.LOCAL_MODELS_DIR),
            "present": models_dir_present,
            "sam": settings.resolved_sam_weights().exists(),
            "mobilesam": settings.resolved_mobilesam_weights().exists(),
            "bigearthnet": settings.resolved_bigearthnet().exists(),
        },
        "disk": {
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
        },
    }
