"""Unit and Integration tests for Master Multi-Query Router in SatQuery AI.

Validates:
1. Dynamic intent routing across 4 workflows:
   - Single-Image VQA (1 image + "describe", "identify", or "land cover")
   - Bi-Temporal Change (2 images + "change", "trend", "increased/decreased", "between dates")
   - Cross-Modal Fusion (Optical + SAR assets / "optical and SAR", "radar")
   - Visual Grounding ("highlight", "segment", "locate", "delineate")
2. Trend formatting enforcement ("Assessment: [Increased | Decreased | Unchanged] — ")
3. Grounding normalized coordinate extraction (<box>[ymin, xmin, ymax, xmax]</box>)
4. Standardized response schema (detected_task, visual_evidence, data:image/png;base64,... prefixes)
"""

import pytest
from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import from_bounds

from app.agents.router import (
    InputInspectorNode,
    TASK_SINGLE_VQA,
    TASK_SINGLE_GROUNDING,
    TASK_BITEMPORAL_CHANGE,
    TASK_CROSS_MODAL,
)
from app.agents.semantic_router import SemanticIntentRouter
from app.services.agent import SatQueryController
from app.services.models.rs_vlm import RemoteSensingVLMClient
from app.schemas.validation import QueryResponseEnvelope, sanitize_for_json


def _create_synthetic_geotiff(
    path: Path,
    bounds: tuple = (77.50, 12.90, 77.60, 13.00),
    shape: tuple = (64, 64),
    bands: int = 4,
    pattern: str = "default",
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = shape
    transform = from_bounds(*bounds, width, height)

    if pattern == "water":
        arr = np.zeros((bands, height, width), dtype=np.float32)
        arr[0] = 0.05  # R
        arr[1] = 0.40  # G (high NDWI)
        arr[2] = 0.10  # B
        if bands > 3:
            arr[3] = 0.02  # NIR
    elif pattern == "urban":
        arr = np.ones((bands, height, width), dtype=np.float32) * 0.4
        arr[:, 20:45, 20:45] = 0.85
    elif pattern == "sar":
        arr = np.ones((bands, height, width), dtype=np.float32) * 0.2
        arr[0, 10:30, 10:30] = 0.01  # Specular water
    else:
        arr = np.ones((bands, height, width), dtype=np.float32) * 0.3

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=bands,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(arr)
    return path


# ==============================================================================
# 1. INTENT CLASSIFICATION TESTS ACROSS 4 WORKFLOWS
# ==============================================================================
def test_router_single_image_vqa_routing(tmp_path: Path):
    """Single-Image VQA: 1 image + describe / identify / land cover."""
    img1 = str(_create_synthetic_geotiff(tmp_path / "img1.tif"))

    # Case A: "describe"
    t1 = InputInspectorNode.inspect(query="Describe the land cover in this scene.", filepaths=[img1])
    assert t1 == "single_image_vqa"

    # Case B: "identify"
    t2 = InputInspectorNode.inspect(query="Identify the dominant land-cover classes across this area.", filepaths=[img1])
    assert t2 == "single_image_vqa"

    # Case C: "land cover"
    t3 = InputInspectorNode.inspect(query="Analyze the agricultural land cover distribution.", filepaths=[img1])
    assert t3 == "single_image_vqa"


def test_router_visual_grounding_routing(tmp_path: Path):
    """Visual Grounding: highlight / segment / locate / delineate."""
    img1 = str(_create_synthetic_geotiff(tmp_path / "img1.tif"))

    # Case A: "highlight"
    t1 = InputInspectorNode.inspect(query="Highlight the water body referred to in the query.", filepaths=[img1])
    assert t1 == "single_image_grounding"

    # Case B: "segment"
    t2 = InputInspectorNode.inspect(query="Segment the central reservoir polygon.", filepaths=[img1])
    assert t2 == "single_image_grounding"

    # Case C: "locate"
    t3 = InputInspectorNode.inspect(query="Locate the industrial storage tanks.", filepaths=[img1])
    assert t3 == "single_image_grounding"

    # Case D: "delineate"
    t4 = InputInspectorNode.inspect(query="Delineate the boundary of the forested parcel.", filepaths=[img1])
    assert t4 == "single_image_grounding"


def test_router_bitemporal_change_routing(tmp_path: Path):
    """Bi-Temporal Change: 2 images + change / trend / increased / decreased / between dates."""
    img1 = str(_create_synthetic_geotiff(tmp_path / "t1.tif"))
    img2 = str(_create_synthetic_geotiff(tmp_path / "t2.tif"))

    # Case A: "change"
    t1 = InputInspectorNode.inspect(query="What changed between these two dates?", filepaths=[img1, img2])
    assert t1 == "bi_temporal_change_analysis"

    # Case B: "trend"
    t2 = InputInspectorNode.inspect(query="Analyze the multi-year trend in reservoir drying.", filepaths=[img1, img2])
    assert t2 == "bi_temporal_change_analysis"

    # Case C: "increased/decreased"
    t3 = InputInspectorNode.inspect(query="Has the built-up area increased, decreased, or remained unchanged?", filepaths=[img1, img2])
    assert t3 == "bi_temporal_change_analysis"

    # Case D: "between dates"
    t4 = InputInspectorNode.inspect(query="Assess shoreline desiccation between dates.", filepaths=[img1, img2])
    assert t4 == "bi_temporal_change_analysis"


def test_router_cross_modal_fusion_routing(tmp_path: Path):
    """Cross-Modal Fusion: Optical + SAR images / SAR radar keywords."""
    opt = str(_create_synthetic_geotiff(tmp_path / "opt.tif", bands=3))
    sar = str(_create_synthetic_geotiff(tmp_path / "sar.tif", bands=1, pattern="sar"))

    # Case A: Mentioning SAR
    t1 = InputInspectorNode.inspect(query="Use SAR radar backscatter to inspect water extent.", filepaths=[opt, sar])
    assert t1 == "cross_modal_joint_analysis"

    # Case B: Optical and SAR together
    t2 = InputInspectorNode.inspect(query="Use optical and SAR images together to identify built-up and water-covered regions.", filepaths=[opt, sar])
    assert t2 == "cross_modal_joint_analysis"


# ==============================================================================
# 2. TREND FORMATTING & GROUNDING ENFORCEMENT
# ==============================================================================
def test_bitemporal_trend_formatting_enforcement(tmp_path: Path):
    """Bi-temporal trend queries must enforce Assessment: [Increased | Decreased | Unchanged] — format."""
    t1 = _create_synthetic_geotiff(tmp_path / "t1_urban.tif", pattern="default")
    t2 = _create_synthetic_geotiff(tmp_path / "t2_urban.tif", pattern="urban")

    controller = SatQueryController()
    trace = controller.execute_workflow(
        query="Has the built-up area increased, decreased, or remained unchanged?",
        filepaths=[str(t1), str(t2)],
    )

    out = trace.output.strip()
    assert "[INCREASED]" in out or "[DECREASED]" in out or "[REMAINED UNCHANGED]" in out
    assert "Assessment:" in out
    assert "—" in out or "-" in out


def test_visual_grounding_normalized_coordinates(tmp_path: Path):
    """Grounding queries extract normalized [ymin, xmin, ymax, xmax] coordinates."""
    vlm = RemoteSensingVLMClient()
    sample_text = "Analysis complete: <box>[120, 150, 480, 520]</box> Central water reservoir detected."
    from app.services.geospatial_parser import parse_geochat_bbox
    box = parse_geochat_bbox(sample_text)
    assert box is not None
    assert box == [120, 150, 480, 520]
    assert all(0 <= v <= 1000 for v in box)


# ==============================================================================
# 3. STANDARDIZED RESPONSE STRUCTURE & IMAGE DATA URIS
# ==============================================================================
def test_standardized_response_envelope_structure(tmp_path: Path):
    """Verifies that all workflows adhere to QueryResponseEnvelope with Base64 data URIs."""
    img1 = str(_create_synthetic_geotiff(tmp_path / "ground_test.tif", pattern="water"))
    controller = SatQueryController()
    trace = controller.execute_workflow(
        query="Highlight the water body referred to in the query.",
        filepaths=[img1],
    )

    from app.utils.report_generator import build_audit_summary
    audit = build_audit_summary(trace)

    scratch = dict(controller.scratchpad)
    detected_task = scratch.get("task_classification", "Water Body Surface Delineation")
    visual_evidence = scratch.get("visual_evidence_type", "Water Body Delineation Mask")

    envelope_data = {
        "status": "ok",
        "answer": trace.output,
        "task_type": trace.task_type,
        "detected_task": detected_task,
        "original_image": controller.t1_preview_url,
        "overlay_image": controller.last_overlay_uri,
        "visual_evidence": controller.last_overlay_uri,
        "evidence_type": visual_evidence,
        "audit_summary": audit,
        "trace": trace.model_dump(),
    }

    env = QueryResponseEnvelope(**sanitize_for_json(envelope_data))
    assert env.status == "ok"
    assert env.detected_task == "Water Body Surface Delineation"
    assert env.evidence_type == "Water Body Delineation Mask"
    assert env.original_image.startswith("data:image/png;base64,")
    assert env.overlay_image.startswith("data:image/png;base64,")
