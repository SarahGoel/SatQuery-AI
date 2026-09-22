"""Tests for Phase 2 — Dynamic Agent Router, Modular Specialist Tools & Domain VLM.

Validates:
1. SemanticIntentRouter intent classification and air-gap deterministic fallbacks.
2. SemanticIntentRouter Ollama /api/chat structured JSON parsing.
3. Multimodal physical constraint enforcement (0 or 1 images with temporal or cross-modal queries).
4. ToolRegistry and individual specialist tools:
   - WaterGroundingTool (NDWI / dark-pixel segmentation and vectorization)
   - TemporalChangeTool (Bi-temporal differencing)
   - OpticalSARFusionTool (Joint Optical + SAR physics)
   - GeodesicMeasurementTool (WGS84 ellipsoidal area calculation in m², ha, km²)
5. RemoteSensingVLMClient (GeoChat domain adaptation).
6. SatQueryController and query_pipeline end-to-end execution with AuditableTraceLogSchema.
"""

from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from app.agents.semantic_router import (
    INTERNAL_BITEMPORAL_CHANGE,
    INTERNAL_CROSS_MODAL,
    INTERNAL_DOMAIN_KNOWLEDGE_QA,
    INTERNAL_SINGLE_GROUNDING,
    INTERNAL_SINGLE_VQA,
    TASK_BITEMPORAL_CHANGE,
    TASK_CROSS_MODAL,
    TASK_DOMAIN_KNOWLEDGE_QA,
    TASK_SINGLE_GROUNDING,
    TASK_SINGLE_VQA,
    SemanticIntentRouter,
    SemanticRoutingResult,
)
from app.schemas.validation import QueryResponseEnvelope
from app.schemas.trace import AuditableTraceLogSchema
from app.services.agent import SatQueryController
from app.services.models.base import VLMResult
from app.services.models.rs_vlm import GEOCHAT_SYSTEM_PROMPT, RemoteSensingVLMClient
from app.tools.base import BaseTool
from app.tools.registry import (
    GeodesicMeasurementTool,
    OpticalSARFusionTool,
    TemporalChangeTool,
    ToolRegistry,
    WaterGroundingTool,
    default_tool_registry,
)


def _create_synthetic_geotiff(
    path: Path,
    west: float = 77.0,
    south: float = 28.0,
    east: float = 77.2,
    north: float = 28.2,
    count: int = 4,
    has_water: bool = True,
) -> Path:
    """Helper to generate a lightweight 4-band synthetic GeoTIFF with water signatures."""
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 32, 32
    transform = from_bounds(west, south, east, north, width, height)
    # Background: moderate reflectance
    data = np.ones((count, height, width), dtype=np.float32) * 0.4
    if has_water:
        # Water: low red (band 1), moderate green (band 2), low NIR (band 4)
        data[0, 5:15, 5:15] = 0.05
        if count > 1:
            data[1, 5:15, 5:15] = 0.35
        if count > 2:
            data[2, 5:15, 5:15] = 0.10
        if count >= 4:
            data[3, 5:15, 5:15] = 0.02

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data)
    return path


# ---------------------------------------------------------------------------
# 1. Semantic Intent Router Tests
# ---------------------------------------------------------------------------


def test_semantic_router_airgap_fallbacks() -> None:
    """Validates router fallback heuristics when Ollama is unavailable."""
    router = SemanticIntentRouter(ollama_url="http://127.0.0.1:99999", timeout=0.1)

    # 1. Water Grounding query with 1 image
    res = router.route(
        query="Detect all water bodies and delineate reservoirs",
        filepaths=["/tmp/optical.tif"],
    )
    assert res.task in (TASK_SINGLE_GROUNDING, "visual_grounding")
    assert res.internal_task == INTERNAL_SINGLE_GROUNDING
    assert "water" in res.target_features
    assert "WaterGroundingTool" in res.tool_chain
    assert "GeodesicMeasurementTool" in res.tool_chain

    # 2. Bi-temporal change query with 2 images
    res = router.route(
        query="What changed between these two dates?",
        filepaths=["/tmp/t1.tif", "/tmp/t2.tif"],
    )
    assert res.task == TASK_BITEMPORAL_CHANGE
    assert res.internal_task == INTERNAL_BITEMPORAL_CHANGE
    assert "TemporalChangeTool" in res.tool_chain
    assert "GeodesicMeasurementTool" in res.tool_chain

    # 3. Cross-modal Optical+SAR with 2 images
    res = router.route(
        query="Perform joint optical and SAR analysis using both sensors",
        filepaths=["/tmp/optical.tif", "/tmp/sar.tif"],
    )
    assert res.task == TASK_CROSS_MODAL
    assert res.internal_task == INTERNAL_CROSS_MODAL
    assert "OpticalSARFusionTool" in res.tool_chain

    # 4. Text-only domain knowledge query with 0 images
    res = router.route(
        query="Explain the orbital repeat cycle of Sentinel-1",
        filepaths=[],
    )
    assert res.task == TASK_DOMAIN_KNOWLEDGE_QA
    assert res.internal_task == INTERNAL_DOMAIN_KNOWLEDGE_QA

    # 5. Single image VQA query
    res = router.route(
        query="Describe the dominant land cover class in this scene",
        filepaths=["/tmp/scene.tif"],
    )
    assert res.task == TASK_SINGLE_VQA
    assert res.internal_task == INTERNAL_SINGLE_VQA


def test_semantic_router_llm_json_mocking(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that structured JSON responses from Ollama /api/chat are parsed correctly."""
    router = SemanticIntentRouter()

    mock_json_response = {
        "message": {
            "content": json.dumps({
                "task": "visual_grounding",
                "target_features": ["infrastructure", "storage_tank"],
                "tool_chain": ["WaterGroundingTool", "GeodesicMeasurementTool"],
                "confidence": 0.97,
                "reasoning": "Target features indicate industrial infrastructure grounding.",
            })
        }
    }

    def _mock_http_post(url: str, payload: dict, **kwargs: Any) -> dict:
        return mock_json_response

    monkeypatch.setattr("app.services.models.base._http_post_json", _mock_http_post)

    res = router.route(
        query="Highlight petroleum storage tanks across the port",
        filepaths=["/tmp/port.tif"],
    )
    assert res.task in (TASK_SINGLE_GROUNDING, "visual_grounding")
    assert "infrastructure" in res.target_features or "storage_tank" in res.target_features
    assert "WaterGroundingTool" in res.tool_chain
    assert res.confidence == 0.97
    assert "industrial infrastructure" in res.reasoning


def test_semantic_router_physical_constraint_enforcement() -> None:
    """Validates that physical constraint violations raise explicit ValueError exceptions."""
    router = SemanticIntentRouter()

    # Temporal change requires 2 images
    with pytest.raises(ValueError, match="Bi-temporal change detection requires two spatially aligned images"):
        router.route(
            query="Show what changed between these two dates",
            filepaths=[],
        )

    with pytest.raises(ValueError, match="Bi-temporal change detection requires two spatially aligned images"):
        router.route(
            query="Analyze flood expansion before and after monsoon",
            filepaths=["/tmp/single.tif"],
        )

    # Cross-modal requires 2 images
    with pytest.raises(ValueError, match="Cross-modal Optical\\+SAR joint analysis requires both Optical and SAR"):
        router.route(
            query="Perform cross-modal optical and sar fusion",
            filepaths=[],
        )

    with pytest.raises(ValueError, match="Cross-modal Optical\\+SAR joint analysis requires both Optical and SAR"):
        router.route(
            query="Combine optical and radar observations",
            filepaths=["/tmp/only_one.tif"],
        )


# ---------------------------------------------------------------------------
# 2. Tool Registry & Specialist Tools Tests
# ---------------------------------------------------------------------------


def test_tool_registry_management() -> None:
    """Verifies tool registration, alias resolution, and execution retrieval."""
    registry = ToolRegistry()

    assert registry.get("WaterGroundingTool") is not None
    assert registry.get("watergroundingtool") is not None
    assert registry.get("water_grounding_tool") is not None
    assert registry.get("temporal_change_tool") is not None
    assert registry.get("optical_sar_fusion_tool") is not None
    assert registry.get("geodesic_measurement_tool") is not None

    tools = registry.list_tools()
    assert "WaterGroundingTool" in tools
    assert "TemporalChangeTool" in tools
    assert "OpticalSARFusionTool" in tools
    assert "GeodesicMeasurementTool" in tools

    assert registry.get("non_existent_tool") is None
    with pytest.raises(KeyError):
        asyncio.run(registry.execute_tool("non_existent_tool", {}))


def test_water_grounding_tool_execution(tmp_path: Path) -> None:
    """Validates WaterGroundingTool segmentation and GeoJSON vectorization."""
    img_path = _create_synthetic_geotiff(tmp_path / "water_scene.tif", has_water=True)
    tool = WaterGroundingTool()
    scratchpad: Dict[str, Any] = {"filepaths": [str(img_path)]}

    res = asyncio.run(tool.execute(scratchpad))
    assert res["status"] == "success"
    assert res["tool"] == "WaterGroundingTool"
    assert res["geojson"] is not None
    assert "features" in res["geojson"]
    assert len(res["geojson"]["features"]) >= 1

    # Verify scratchpad was updated
    assert scratchpad["water_features_count"] >= 1
    assert "water_mask" in scratchpad
    assert scratchpad["geojson"] is not None


def test_temporal_change_tool_execution(tmp_path: Path) -> None:
    """Validates TemporalChangeTool bi-temporal differencing."""
    t1_path = _create_synthetic_geotiff(tmp_path / "t1.tif", has_water=False)
    t2_path = _create_synthetic_geotiff(tmp_path / "t2.tif", has_water=True)
    tool = TemporalChangeTool()
    scratchpad: Dict[str, Any] = {
        "optical_path": str(t1_path),
        "optical_t2_path": str(t2_path),
    }

    res = asyncio.run(tool.execute(scratchpad))
    assert res["status"] == "success"
    assert res["tool"] == "TemporalChangeTool"
    assert res["change_mask"] is not None
    assert scratchpad["change_pixel_count"] >= 0


def test_optical_sar_fusion_tool_execution(tmp_path: Path) -> None:
    """Validates OpticalSARFusionTool cross-modal indicator extraction."""
    opt_path = _create_synthetic_geotiff(tmp_path / "opt.tif", count=4)
    sar_path = _create_synthetic_geotiff(tmp_path / "sar.tif", count=2)
    tool = OpticalSARFusionTool()
    scratchpad: Dict[str, Any] = {
        "optical_path": str(opt_path),
        "sar_path": str(sar_path),
    }

    res = asyncio.run(tool.execute(scratchpad))
    assert res["status"] == "success"
    assert res["tool"] == "OpticalSARFusionTool"
    assert "sar_backscatter_mean_db" in res["indicators"]
    assert "optical_surface_reflectance_mean" in res["indicators"]
    assert scratchpad["fusion_result"] is not None


def test_geodesic_measurement_tool_calculation() -> None:
    """Validates GeodesicMeasurementTool ellipsoidal WGS84 surface area computation."""
    tool = GeodesicMeasurementTool()
    # 0.1 degree x 0.1 degree polygon near equator (~11.1 km x ~11.1 km ~ 123 km²)
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"name": "Test Area"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [77.0, 20.0],
                            [77.1, 20.0],
                            [77.1, 20.1],
                            [77.0, 20.1],
                            [77.0, 20.0],
                        ]
                    ],
                },
            }
        ],
    }
    scratchpad: Dict[str, Any] = {"geojson": geojson}

    res = asyncio.run(tool.execute(scratchpad))
    assert res["status"] == "success"
    metrics = res["metrics"]
    assert metrics["ellipsoid"] == "WGS84"
    assert metrics["total_area_m2"] > 1e8
    assert metrics["total_area_ha"] > 1e4
    assert metrics["total_area_km2"] > 100.0

    # Verify individual feature properties enriched
    feature_props = scratchpad["geojson"]["features"][0]["properties"]
    assert "area_m2" in feature_props
    assert "area_ha" in feature_props
    assert "area_km2" in feature_props
    assert feature_props["area_km2"] == metrics["total_area_km2"]
    assert scratchpad["geospatial_metrics"] == metrics


# ---------------------------------------------------------------------------
# 3. Domain VLM Specialist Tests
# ---------------------------------------------------------------------------


def test_remote_sensing_vlm_client() -> None:
    """Validates RemoteSensingVLMClient domain prompt configuration."""
    vlm = RemoteSensingVLMClient()
    assert vlm.system_prompt == GEOCHAT_SYSTEM_PROMPT
    assert "GeoChat-RS" in vlm.system_prompt
    assert "Earth Observation" in vlm.system_prompt

    # Test generation passes domain adapter context
    res = vlm.generate("Describe surface water coverage in this scene")
    assert isinstance(res, VLMResult)
    assert res.params.get("domain_adapter") == "GeoChat-RS"
    assert res.text


# ---------------------------------------------------------------------------
# 4. End-to-End Controller & Trace Integration Tests
# ---------------------------------------------------------------------------


def test_satquery_controller_end_to_end_with_router_and_tools(tmp_path: Path) -> None:
    """Validates full workflow execution enriching scratchpad and AuditableTraceLogSchema."""
    img_path = _create_synthetic_geotiff(tmp_path / "water_site.tif", has_water=True)
    controller = SatQueryController(db_session=None)

    trace = controller.execute_workflow(
        query="Detect water bodies and calculate surface area",
        filepaths=[str(img_path)],
    )

    assert isinstance(trace, AuditableTraceLogSchema)
    assert trace.task in ("single_image_grounding", "visual_grounding")
    assert trace.tools_executed is not None
    assert len(trace.tools_executed) >= 1

    # Verify trace enrichment fields
    assert trace.intent_classification is not None
    assert trace.intent_classification.get("task") in ("single_grounding", "visual_grounding")
    assert "water" in trace.intent_classification.get("target_features", [])

    assert trace.geospatial_metrics is not None
    assert "total_area_m2" in trace.geospatial_metrics
    assert trace.geospatial_metrics["ellipsoid"] == "WGS84"

    assert trace.scratchpad is not None
    assert "water_features_count" in trace.scratchpad


def test_e2e_query_endpoint_response_envelope(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Validates POST /api/v1/query handler produces valid QueryResponseEnvelope."""
    pytest.importorskip("fastapi")
    from fastapi import UploadFile
    import backend.api.routes as routes_mod

    # Setup temp directories
    uploads_dir = tmp_path / "uploads"
    artifacts_dir = tmp_path / "artifacts"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(routes_mod.settings, "UPLOAD_DIR", uploads_dir)
    monkeypatch.setattr(routes_mod.settings, "ARTIFACT_DIR", artifacts_dir)

    # Create dummy GeoTIFF upload
    tiff_path = _create_synthetic_geotiff(tmp_path / "input.tif", has_water=True)
    with open(tiff_path, "rb") as f:
        content = f.read()

    upload = UploadFile(filename="input.tif", file=io.BytesIO(content))

    async def _invoke():
        return await routes_mod.query_pipeline(
            query="Locate water bodies and calculate surface area",
            files=[upload],
            file=None,
            image=None,
            optical=None,
            optical_t2=None,
            sar=None,
            force_task=None,
            use_mobilesam=False,
            db=None,
        )

    envelope = asyncio.run(_invoke())
    assert isinstance(envelope, QueryResponseEnvelope)
    payload = envelope.model_dump()
    assert payload["status"] == "ok"
    assert payload["task_type"] in ("single_image_grounding", "visual_grounding", "single_grounding")
    assert payload["trace"] is not None
    assert payload["trace"]["tools_executed"]
    assert payload["trace"]["intent_classification"] is not None
    assert payload["trace"]["geospatial_metrics"] is not None
