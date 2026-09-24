"""POST /api/v1/satquery/analyze — Legacy endpoint consolidated into query_pipeline."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, UploadFile
from sqlalchemy.orm import Session

from app.database.session import get_optional_db
from app.schemas.validation import QueryResponseEnvelope

router = APIRouter()


@router.post("/analyze", response_model=QueryResponseEnvelope, deprecated=True)
async def analyze(
    query: str = Form(..., min_length=3),
    optical: UploadFile | None = File(default=None, description="Primary optical GeoTIFF (Cartosat-2S)"),
    optical_t2: UploadFile | None = File(default=None, description="Optional T2 optical GeoTIFF"),
    sar: UploadFile | None = File(default=None, description="Optional SAR GeoTIFF (RISAT)"),
    files: list[UploadFile] | None = File(default=None),
    images: list[UploadFile] | None = File(default=None),
    file: UploadFile | None = File(default=None),
    image: UploadFile | None = File(default=None),
    image1: UploadFile | None = File(default=None),
    image2: UploadFile | None = File(default=None),
    force_task: str | None = Form(default=None),
    use_mobilesam: bool = Form(default=True),
    db: Session | None = Depends(get_optional_db),
) -> QueryResponseEnvelope:
    """Consolidated legacy analyze endpoint delegating to the unified query_pipeline."""
    try:
        from api.routes import query_pipeline
    except ImportError:
        from backend.api.routes import query_pipeline

    return await query_pipeline(
        query=query,
        files=files,
        images=images,
        file=file,
        image=image,
        image1=image1,
        image2=image2,
        optical=optical,
        optical_t2=optical_t2,
        sar=sar,
        force_task=force_task,
        use_mobilesam=use_mobilesam,
        db=db,
    )
