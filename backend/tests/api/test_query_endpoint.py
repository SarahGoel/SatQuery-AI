"""POST /api/v1/query envelope without TestClient/httpx."""

from __future__ import annotations

import asyncio
import io
import struct
import zlib
from pathlib import Path

from fastapi import UploadFile

from app.schemas.trace import (
    AuditableTraceLogSchema,
    InputMetadataSchema,
    RegistryExecutionSchema,
)


def _tiny_png() -> bytes:
    raw = b"\x00" + bytes([255, 0, 0, 0, 255, 0])
    raw += b"\x00" + bytes([0, 0, 255, 255, 255, 0])
    ihdr = struct.pack(">IIBBBBB", 2, 2, 8, 2, 0, 0, 0)

    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )


class _StubController:
    last_geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"kind": "mask"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [77.05, 28.05],
                            [77.1, 28.05],
                            [77.1, 28.1],
                            [77.05, 28.1],
                            [77.05, 28.05],
                        ]
                    ],
                },
            }
        ],
    }
    last_overlay_uri = None
    last_bbox = [77.0, 28.0, 77.2, 28.2]

    def __init__(self, db=None) -> None:  # noqa: ANN001
        self.db = db

    def execute_workflow(self, query: str, filepaths, **kwargs):  # noqa: ANN001
        assert filepaths, "orchestrator must receive persisted filepaths"
        for path in filepaths:
            assert Path(path).exists()
        return AuditableTraceLogSchema(
            trace_id="ISRO-SQ-2026-E2E001",
            task="single_image_grounding",
            query=query,
            input_metadata=InputMetadataSchema(
                crs="EPSG:4326",
                bounds=[77.0, 28.0, 77.2, 28.2],
                affine_transform=[0.0125, 0.0, 77.0, 0.0, -0.0125, 28.2],
                modalities=["RGB"],
            ),
            registry_execution=[
                RegistryExecutionSchema(model="RS-Grounding-V3", params={"threshold": 0.75}),
            ],
            confidence_score=0.88,
            output="Grounded dummy rooftops in the northern quadrant.",
        )


def test_query_endpoint_returns_answer_geometry_and_audit(monkeypatch, tmp_path: Path) -> None:
    pytest = __import__("pytest")
    sqlalchemy = pytest.importorskip("sqlalchemy")
    _ = sqlalchemy
    pytest.importorskip("rasterio")
    pytest.importorskip("cv2")

    import backend.api.routes as routes_mod

    monkeypatch.setattr(routes_mod, "SatQueryController", _StubController)
    monkeypatch.setattr(routes_mod.settings, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(routes_mod.settings, "ARTIFACT_DIR", tmp_path / "artifacts")
    (tmp_path / "uploads").mkdir()
    (tmp_path / "artifacts").mkdir()

    upload = UploadFile(filename="dummy.png", file=io.BytesIO(_tiny_png()))

    async def _run():
        return await routes_mod.query_pipeline(
            query="Highlight industrial rooftops",
            files=[upload],
            file=None,
            image=None,
            optical=None,
            optical_t2=None,
            sar=None,
            force_task=None,
            use_mobilesam=True,
            db=None,
        )

    envelope = asyncio.run(_run())
    payload = envelope.model_dump()
    assert payload["status"] == "ok"
    assert "rooftops" in payload["answer"].lower()
    assert payload["bbox"] == [77.0, 28.0, 77.2, 28.2]
    assert payload["geojson"]["type"] == "FeatureCollection"
    assert payload["change_mask"]["coordinates"]
    assert payload["audit_summary"]["selected_task"] == "single_image_grounding"
    assert "RS-Grounding-V3" in payload["audit_summary"]["model_names"]
    assert payload["audit_summary"]["confidence_score"] == 0.88
    assert payload["trace"]["trace_id"] == "ISRO-SQ-2026-E2E001"

    async def _report():
        return await routes_mod.download_audit_report(trace_id="ISRO-SQ-2026-E2E001", format="json", db=None)

    report = asyncio.run(_report())
    assert report.status_code == 200
    assert b"selected_task" in report.body


def test_query_endpoint_text_only_domain_query(monkeypatch, tmp_path: Path) -> None:
    pytest = __import__("pytest")
    pytest.importorskip("rasterio")
    import backend.api.routes as routes_mod

    monkeypatch.setattr(routes_mod.settings, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(routes_mod.settings, "ARTIFACT_DIR", tmp_path / "artifacts")
    (tmp_path / "uploads").mkdir(parents=True, exist_ok=True)
    (tmp_path / "artifacts").mkdir(parents=True, exist_ok=True)

    from app.services.models.base import VLMResult
    monkeypatch.setattr(
        "app.services.models.base.LocalVisionLanguageClient.generate",
        lambda self, prompt, **kwargs: VLMResult(
            text=f"The Sentinel-1 constellation provides C-band SAR observations with a 6-to-12 day revisit period. Analysis for: {prompt}",
            confidence=0.92,
            params={"backend": "ollama", "model": "llava"},
        ),
    )

    async def _run():
        return await routes_mod.query_pipeline(
            query="What is the revisit period of Sentinel-1?",
            files=None,
            db=None,
        )

    envelope = asyncio.run(_run())
    payload = envelope.model_dump()
    assert payload["status"] == "ok"
    assert payload["task_type"] == "domain_knowledge_qa"
    assert payload["geojson"] is None
    assert payload["bbox"] is None
    assert payload["change_mask"] is None
    assert "sentinel-1" in payload["answer"].lower()
    assert payload["audit_summary"]["selected_task"] == "domain_knowledge_qa"
    assert "Earth Observation Domain Knowledge" in payload["headline"]


def test_query_endpoint_accepts_session_id(monkeypatch, tmp_path: Path) -> None:
    pytest = __import__("pytest")
    pytest.importorskip("rasterio")
    import backend.api.routes as routes_mod

    monkeypatch.setattr(routes_mod.settings, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(routes_mod.settings, "ARTIFACT_DIR", tmp_path / "artifacts")
    (tmp_path / "uploads").mkdir(parents=True, exist_ok=True)
    (tmp_path / "artifacts").mkdir(parents=True, exist_ok=True)

    from app.services.models.base import VLMResult
    monkeypatch.setattr(
        "app.services.models.base.LocalVisionLanguageClient.generate",
        lambda self, prompt, **kwargs: VLMResult(
            text=f"Domain answer for: {prompt}",
            confidence=0.92,
            params={"backend": "ollama", "model": "llava"},
        ),
    )

    async def _run():
        return await routes_mod.query_pipeline(
            query="Explain SAR backscatter coefficient",
            files=None,
            session_id="session-xyz-789",
            db=None,
        )

    envelope = asyncio.run(_run())
    payload = envelope.model_dump()
    assert payload["status"] == "ok"
    assert payload["task_type"] == "domain_knowledge_qa"


def test_query_endpoint_unexpected_exception_returns_500_with_detail(monkeypatch, tmp_path: Path) -> None:
    pytest = __import__("pytest")
    from fastapi import HTTPException
    import backend.api.routes as routes_mod

    class _FailingController:
        def __init__(self, db=None) -> None:
            self.db = db

        def execute_workflow(self, *args, **kwargs):
            raise RuntimeError("CRITICAL_INTERNAL_MODEL_FAILURE")

    monkeypatch.setattr(routes_mod, "SatQueryController", _FailingController)
    monkeypatch.setattr(routes_mod.settings, "UPLOAD_DIR", tmp_path / "uploads")
    (tmp_path / "uploads").mkdir(parents=True, exist_ok=True)

    async def _run():
        return await routes_mod.query_pipeline(
            query="Detect flooding in Kerala",
            files=None,
            db=None,
        )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(_run())

    assert exc_info.value.status_code == 500
    assert "CRITICAL_INTERNAL_MODEL_FAILURE" in exc_info.value.detail


def test_query_endpoint_empty_files_list_default(monkeypatch, tmp_path: Path) -> None:
    pytest = __import__("pytest")
    pytest.importorskip("rasterio")
    import backend.api.routes as routes_mod

    monkeypatch.setattr(routes_mod.settings, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(routes_mod.settings, "ARTIFACT_DIR", tmp_path / "artifacts")
    (tmp_path / "uploads").mkdir(parents=True, exist_ok=True)
    (tmp_path / "artifacts").mkdir(parents=True, exist_ok=True)

    from app.services.models.base import VLMResult
    monkeypatch.setattr(
        "app.services.models.base.LocalVisionLanguageClient.generate",
        lambda self, prompt, **kwargs: VLMResult(
            text=f"Spatial resolution analysis for: {prompt}",
            confidence=0.92,
            params={"backend": "ollama", "model": "llava"},
        ),
    )

    async def _run():
        return await routes_mod.query_pipeline(
            query="Explain spatial resolution in optical satellites",
            files=[],
            db=None,
        )

    envelope = asyncio.run(_run())
    payload = envelope.model_dump()
    assert payload["status"] == "ok"
    assert payload["task_type"] == "domain_knowledge_qa"



