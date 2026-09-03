"""Phase 5 Step 11 — LangGraph SatQueryController classification and overlap."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_bounds

from app.services.agent import SatQueryController, compile_satquery_graph


def _write_geotiff(path: Path, west: float, south: float, east: float, north: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 16, 16
    transform = from_bounds(west, south, east, north, width, height)
    data = np.ones((4, height, width), dtype=np.float32)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=4,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data)


def test_classify_query_rule_based_mapping() -> None:
    controller = SatQueryController(db_session=None)
    assert controller.classify_query("What changed between these dates?") == "bi_temporal_change_analysis"
    assert controller.classify_query("before vs after monsoon") == "bi_temporal_change_analysis"
    assert controller.classify_query("ground the airport apron") == "single_image_grounding"
    assert controller.classify_query("highlight industrial rooftops") == "single_image_grounding"
    assert controller.classify_query("where is the reservoir") == "single_image_grounding"
    assert controller.classify_query("combine optical and sar") == "cross_modal_joint_analysis"
    assert controller.classify_query("use both sensors") == "cross_modal_joint_analysis"
    assert controller.classify_query("Is there vegetation?") == "single_image_vqa"


def test_execute_workflow_change_task(tmp_path: Path) -> None:
    t1 = tmp_path / "test_t1.tif"
    t2 = tmp_path / "test_t2.tif"
    _write_geotiff(t1, 77.0, 28.0, 77.2, 28.2)
    _write_geotiff(t2, 77.05, 28.05, 77.15, 28.15)
    controller = SatQueryController()
    trace = controller.execute_workflow(
        query="What changed between these dates?",
        filepaths=[str(t1), str(t2)],
    )
    assert trace.task == "bi_temporal_change_analysis"
    assert trace.registry_execution[0].model == "CD-VQA-Pro"
    assert trace.trace_id.startswith("ISRO-SQ-2026-")
    dumped = trace.model_dump_json(indent=2)
    assert "bi_temporal_change_analysis" in dumped


def test_validate_spatial_alignment_rejects_disjoint(tmp_path: Path) -> None:
    t1 = tmp_path / "a.tif"
    t2 = tmp_path / "b.tif"
    _write_geotiff(t1, 77.0, 28.0, 77.1, 28.1)
    _write_geotiff(t2, 78.0, 29.0, 78.1, 29.1)
    controller = SatQueryController()
    meta_t1 = controller.parse_geotiff_metadata(str(t1))
    meta_t2 = controller.parse_geotiff_metadata(str(t2))
    assert controller.validate_spatial_alignment(meta_t1, meta_t2) is False


def test_langgraph_tracks_file_states(tmp_path: Path) -> None:
    t1 = tmp_path / "scene.tif"
    _write_geotiff(t1, 77.0, 28.0, 77.2, 28.2)
    controller = SatQueryController()
    graph = compile_satquery_graph(controller)
    if graph is None:
        return
    result = graph.invoke(
        {
            "query": "Is there vegetation?",
            "filepaths": [str(t1)],
            "file_states": {str(t1): "queued"},
        }
    )
    assert result["task"] == "single_image_vqa"
    assert result["file_states"][str(t1)] == "persisted"
    assert result["trace"]["task"] == "single_image_vqa"
