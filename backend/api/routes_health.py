"""GET /health — GDAL/Rasterio, PyTorch/CUDA, and local weights registry."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter

router = APIRouter(tags=["health"])

WEIGHT_SUBDIRS = ("sam", "bigearthnet", "cdvqa", "vllm")


def local_models_root() -> Path:
    return Path(os.getenv("LOCAL_MODELS_DIR", "/local_models"))


def probe_gdal() -> dict[str, Any]:
    """Return Rasterio and native GDAL versions, or an error payload."""
    payload: dict[str, Any] = {"rasterio": None, "gdal": None, "ok": False}
    try:
        import rasterio

        payload["rasterio"] = getattr(rasterio, "__version__", None)
        gdal_ver = None
        if hasattr(rasterio, "gdal_version"):
            gdal_ver = rasterio.gdal_version()
        elif hasattr(rasterio, "__gdal_version__"):
            gdal_ver = rasterio.__gdal_version__
        payload["gdal"] = gdal_ver
        payload["ok"] = bool(gdal_ver)
    except Exception as exc:  # noqa: BLE001 — health must never 500 on missing GIS libs
        payload["error"] = str(exc)
    return payload


def probe_torch() -> dict[str, Any]:
    """Return PyTorch version and CUDA availability for air-gapped GPU hosts."""
    payload: dict[str, Any] = {
        "pytorch": None,
        "cuda_available": False,
        "cuda_compiled": None,
        "device_count": 0,
        "ok": False,
    }
    try:
        import torch

        payload["pytorch"] = torch.__version__
        payload["cuda_compiled"] = torch.version.cuda
        payload["cuda_available"] = bool(torch.cuda.is_available())
        payload["device_count"] = int(torch.cuda.device_count()) if payload["cuda_available"] else 0
        payload["ok"] = payload["pytorch"] is not None
        if payload["cuda_available"]:
            payload["devices"] = []
            for idx in range(payload["device_count"]):
                props = torch.cuda.get_device_properties(idx)
                payload["devices"].append(
                    {
                        "index": idx,
                        "name": props.name,
                        "total_memory_bytes": int(props.total_memory),
                    }
                )
    except Exception as exc:  # noqa: BLE001
        payload["error"] = str(exc)
    return payload


def probe_air_gap(models_dir: Path | None = None) -> dict[str, Any]:
    """Existence + read permission of /local_models and required weight registries."""
    root = models_dir or local_models_root()
    exists = root.exists() and root.is_dir()
    readable = bool(exists and os.access(root, os.R_OK))
    registries: dict[str, Any] = {}
    all_registries_ok = True
    for name in WEIGHT_SUBDIRS:
        child = root / name
        child_ok = child.is_dir() and os.access(child, os.R_OK)
        registries[name] = {
            "path": str(child),
            "exists": child.is_dir(),
            "readable": bool(child.is_dir() and os.access(child, os.R_OK)),
        }
        all_registries_ok = all_registries_ok and child_ok

    ready = bool(exists and readable and all_registries_ok)
    return {
        "ready": ready,
        "path": str(root),
        "exists": exists,
        "readable": readable,
        "registries": registries,
    }


@router.get("/health")
async def health() -> dict[str, Any]:
    gdal = probe_gdal()
    torch_info = probe_torch()
    air_gap = probe_air_gap()

    healthy = bool(gdal.get("ok") and torch_info.get("ok") and air_gap.get("ready"))

    return {
        "status": "healthy" if healthy else "degraded",
        "service": "satquery-backend",
        "problem_statement": "SIH26167",
        "environment": os.getenv("SATQUERY_ENV", "development"),
        "gdal_version": {
            "gdal": gdal.get("gdal"),
            "rasterio": gdal.get("rasterio"),
            "ok": gdal.get("ok"),
            "error": gdal.get("error"),
        },
        "torch_version": {
            "pytorch": torch_info.get("pytorch"),
            "cuda_available": torch_info.get("cuda_available"),
            "cuda_compiled": torch_info.get("cuda_compiled"),
            "device_count": torch_info.get("device_count"),
            "devices": torch_info.get("devices", []),
            "ok": torch_info.get("ok"),
            "error": torch_info.get("error"),
        },
        "air_gap_ready": air_gap.get("ready"),
        "air_gap": air_gap,
    }
