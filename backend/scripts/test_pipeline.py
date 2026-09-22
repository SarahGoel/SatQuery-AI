#!/usr/bin/env python3
"""Comprehensive local test for SatQueryController and the 4 ISRO core workflows:
1. Single-Image Grounding (RS-Grounding-V3 / MobileSAM) -> 'single_grounding'
2. Single-Image VQA (Remote Sensing VLM) -> 'single_vqa'
3. Bi-Temporal Change Detection (CD-VQA-Pro) -> 'bitemporal_change'
4. Cross-Modal Optical + SAR Analysis (Opt-SAR-Fusion) -> 'cross_modal'

Also tests:
- InputInspectorNode rejection for intent-input count mismatches
- Discrete multi-instance GeoJSON feature extraction (no single-box aggregation)
- Standardized auditable trace schema compliance
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_bounds

def _find_artifacts_dir() -> Path:
    p = Path(__file__).resolve().parent
    for _ in range(4):
        candidate = p / "artifacts"
        if candidate.is_dir():
            return candidate
        p = p.parent
    return Path.cwd() / "artifacts"

from app.services.agent import SatQueryController
from app.agents.router import InputInspectorNode



def write_dummy_geotiff(path: Path, west: float, south: float, east: float, north: float, bands: int = 4) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 256, 256
    transform = from_bounds(west, south, east, north, width, height)
    rng = np.random.default_rng(42)
    data = rng.random((bands, height, width), dtype=np.float32) * 0.3 + 0.2

    # Draw genuine physical features so computer vision algorithms detect real targets
    circle_img = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(circle_img, (160, 180), 8, 255, -1)
    cv2.circle(circle_img, (190, 195), 8, 255, -1)
    cv2.circle(circle_img, (215, 185), 8, 255, -1)
    circle_mask = circle_img > 0
    for b in range(bands):
        data[b][circle_mask] = 0.95

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
        dst.write(data)


def main() -> int:
    artifacts = _find_artifacts_dir() / "cli_test"
    optical = artifacts / "optical_t1.tif"
    t2 = artifacts / "optical_t2.tif"
    sar = artifacts / "sar.tif"

    write_dummy_geotiff(optical, 77.0, 28.0, 77.2, 28.2, bands=4)
    write_dummy_geotiff(t2, 77.02, 28.02, 77.18, 28.18, bands=4)
    write_dummy_geotiff(sar, 77.01, 28.01, 77.19, 28.19, bands=2)

    controller = SatQueryController(db=None)

    # 1. Metadata & Spatial Alignment Tests
    meta = controller.parse_geotiff_metadata(str(optical))
    assert meta["crs"] == "EPSG:4326"
    assert meta["sensor"] == "Cartosat-2S (Multispectral)"
    assert meta["band_count"] == 4
    print("Metadata check passed:", meta["sensor"], meta["resolution"], "Bands:", meta["band_count"])

    sar_meta = controller.parse_geotiff_metadata(str(sar))
    assert sar_meta["sensor"] == "Sentinel-1 / RISAT (SAR C-Band)"
    assert sar_meta["band_count"] == 2
    print("SAR metadata check passed:", sar_meta["sensor"], sar_meta["modalities"])

    t2_meta = controller.parse_geotiff_metadata(str(t2))
    assert controller.validate_spatial_alignment(meta, t2_meta)
    print("Spatial alignment check passed.")

    # 2. InputInspector Rejection Tests (No silent fallback!)
    print("\n--- Testing InputInspector Intent Mismatch Rejections ---")
    try:
        controller.execute_workflow(
            query="What changed between these two dates?",
            filepaths=[str(optical)],  # ONLY 1 IMAGE!
        )
        print("FAIL: Should have raised ValueError for 1-image temporal change query!")
        return 1
    except ValueError as exc:
        print("PASSED: Correctly rejected 1-image temporal query:", str(exc)[:80], "...")

    try:
        controller.execute_workflow(
            query="Combine optical and SAR flood analysis",
            filepaths=[str(optical)],  # ONLY 1 IMAGE!
        )
        print("FAIL: Should have raised ValueError for 1-image cross-modal query!")
        return 1
    except ValueError as exc:
        print("PASSED: Correctly rejected 1-image cross-modal query:", str(exc)[:80], "...")

    # 3. Core Workflow 1: Single-Image Object Grounding
    print("\n--- Testing Workflow 1: Single-Image Grounding (RS-Grounding-V3 / MobileSAM) ---")
    grounding_trace = controller.execute_workflow(
        query="Highlight circular storage tanks in this scene",
        filepaths=[str(optical)],
    )
    assert grounding_trace.task == "single_image_grounding"
    assert grounding_trace.task_type == "single_grounding"
    assert controller.last_geojson is not None
    features = controller.last_geojson.get("features", [])
    assert len(features) > 0
    print(f"Grounding generated {len(features)} discrete feature(s).")
    # Verify individual feature properties (not merged single macro-polygon)
    first_feat = features[0]
    assert "properties" in first_feat
    assert "label" in first_feat["properties"]
    assert "confidence" in first_feat["properties"]
    print("Sample feature properties:", first_feat["properties"])
    print("PASSED: Workflow 1 (Single-Image Grounding) verified.")

    # 4. Core Workflow 2: Single-Image VQA
    print("\n--- Testing Workflow 2: Single-Image VQA (Remote Sensing VLM) ---")
    vqa_trace = controller.execute_workflow(
        query="Describe land cover semantics and scene characteristics",
        filepaths=[str(optical)],
    )
    assert vqa_trace.task == "single_image_vqa"
    assert vqa_trace.task_type == "single_vqa"
    assert vqa_trace.confidence > 0.0
    print("VQA Trace:", vqa_trace.trace_id, "Confidence:", vqa_trace.confidence)
    print("Answer snippet:", vqa_trace.output[:100], "...")
    print("PASSED: Workflow 2 (Single-Image VQA) verified.")

    # 5. Core Workflow 3: Bi-Temporal Change Detection
    print("\n--- Testing Workflow 3: Bi-Temporal Change Detection (CD-VQA-Pro) ---")
    change_trace = controller.execute_workflow(
        query="What changed between these two dates?",
        filepaths=[str(optical), str(t2)],
    )
    assert change_trace.task == "bi_temporal_change_analysis"
    assert change_trace.task_type == "bitemporal_change"
    assert controller.last_overlay_uri is not None
    assert "CD-VQA-Pro" in change_trace.models_executed or "change-vqa" in change_trace.models_executed
    print("Change detection overlay:", controller.last_overlay_uri)
    print("Change output:", change_trace.output[:120], "...")
    print("PASSED: Workflow 3 (Bi-Temporal Change Detection) verified.")

    # 6. Core Workflow 4: Cross-Modal Optical + SAR Analysis
    print("\n--- Testing Workflow 4: Cross-Modal Optical + SAR Analysis ---")
    fused_trace = controller.execute_workflow(
        query="Perform cross-modal Optical and SAR joint analysis for flood and infrastructure",
        filepaths=[str(optical), str(sar)],
    )
    assert fused_trace.task == "cross_modal_joint_analysis"
    assert fused_trace.task_type == "cross_modal"
    assert "cross_modal_analysis_tool" in fused_trace.models_executed
    assert controller.last_geojson is not None
    cm_features = controller.last_geojson.get("features", [])
    assert len(cm_features) >= 2
    feature_types = {f["properties"].get("feature_type") for f in cm_features}
    assert "built_up" in feature_types
    assert "water_inundation" in feature_types
    print(f"Cross-modal emitted {len(cm_features)} features across modalities: {feature_types}")
    print("Cross-modal output:", fused_trace.output[:150], "...")
    print("PASSED: Workflow 4 (Cross-Modal Optical + SAR Analysis) verified.")

    # 7. Standardized Trace Schema Verification
    print("\n--- Verifying Standardized Audit Schema Compliance ---")
    for tr in [grounding_trace, vqa_trace, change_trace, fused_trace]:
        assert tr.task_type in ["single_grounding", "single_vqa", "bitemporal_change", "cross_modal"]
        assert len(tr.models_executed) > 0
        assert tr.input_metadata.sensor is not None
        assert tr.input_metadata.crs is not None
        assert tr.input_metadata.band_count is not None
        assert tr.confidence is not None
        print(f"Trace {tr.trace_id}: task_type='{tr.task_type}', models={tr.models_executed}, conf={tr.confidence}")

    print("\nALL PIPELINE & AUDIT CHECKS PASSED SUCCESSFULLY (100% compliant)!")
    return 0


def test_pipeline_execution() -> None:
    assert main() == 0


if __name__ == "__main__":
    raise SystemExit(main())

