"""POST /api/v1/satquery/analyze — multipart GeoTIFF upload + NL query."""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.core.config import settings
from app.database.session import get_db
from app.schemas.trace import AuditableTraceLogSchema
from app.schemas.validation import AnalyzeResponseEnvelope, TaskType
from app.services.agent import SatQueryController
from app.utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)

ALLOWED_SUFFIXES = {".tif", ".tiff", ".gtiff"}


async def _persist_upload(upload: UploadFile, dest_dir: Path) -> Path:
    suffix = Path(upload.filename or "scene.tif").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=400, detail=f"Unsupported raster suffix: {suffix}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / f"{uuid.uuid4().hex}{suffix}"
    nbytes = 0
    with target.open("wb") as handle:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            nbytes += len(chunk)
            if nbytes > settings.MAX_UPLOAD_BYTES:
                handle.close()
                target.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="GeoTIFF exceeds MAX_UPLOAD_BYTES")
            handle.write(chunk)
    if nbytes == 0:
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Empty raster upload")
    return target


@router.post("/analyze", response_model=AnalyzeResponseEnvelope)
async def analyze(
    query: str = Form(..., min_length=3),
    optical: UploadFile = File(..., description="Primary optical GeoTIFF (Cartosat-2S)"),
    optical_t2: UploadFile | None = File(default=None, description="Optional T2 optical GeoTIFF"),
    sar: UploadFile | None = File(default=None, description="Optional SAR GeoTIFF (RISAT)"),
    force_task: str | None = Form(default=None),
    use_mobilesam: bool = Form(default=True),
    db: Session = Depends(get_db),
) -> AnalyzeResponseEnvelope:
    """Accept multipart rasters, route through SatQueryController, return auditable trace."""
    forced: TaskType | None = None
    if force_task:
        try:
            forced = TaskType(force_task)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="Unknown force_task") from exc

    trace_dir = settings.UPLOAD_DIR / uuid.uuid4().hex
    optical_path = await _persist_upload(optical, trace_dir)
    t2_path = await _persist_upload(optical_t2, trace_dir) if optical_t2 is not None else None
    sar_path = await _persist_upload(sar, trace_dir) if sar is not None else None

    logger.info(
        "analyze_received",
        extra={
            "query": query[:200],
            "optical": str(optical_path),
            "t2": str(t2_path) if t2_path else None,
            "sar": str(sar_path) if sar_path else None,
        },
    )

    controller = SatQueryController(db=db)
    try:
        trace: AuditableTraceLogSchema = controller.execute_workflow(
            query=query,
            optical_path=optical_path,
            optical_t2_path=t2_path,
            sar_path=sar_path,
            force_task=forced.value if forced else None,
            use_mobilesam=use_mobilesam,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("analyze_failed")
        raise HTTPException(status_code=500, detail="Workflow failed") from exc

    return AnalyzeResponseEnvelope(
        status="ok",
        geojson=controller.last_geojson,
        change_overlay_uri=controller.last_overlay_uri,
        trace=trace.model_dump(),
    )
