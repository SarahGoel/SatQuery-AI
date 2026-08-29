#!/usr/bin/env python3
"""Local CLI smoke test for SatQueryController using mocked GeoTIFF metadata.

Does not require GPU. Uses in-memory GeoTIFFs so classification and routing
can be exercised without ISRO scenes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_bounds

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.schemas.trace import InputMetadataSchema  # noqa: E402
from app.schemas.validation import TaskType  # noqa: E402
from app.services.agent import SatQueryController  # noqa: E402


def write_dummy_geotiff(path: Path, west: float, south: float, east: float, north: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 32, 32
    transform = from_bounds(west, south, east, north, width, height)
    data = np.random.default_rng(42).random((4, height, width), dtype=np.float32)
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


def main() -> int:
    artifacts = ROOT / "artifacts" / "cli_test"
    optical = artifacts / "optical_t1.tif"
    t2 = artifacts / "optical_t2.tif"
    sar = artifacts / "sar.tif"
    write_dummy_geotiff(optical, 77.0, 28.0, 77.2, 28.2)
    write_dummy_geotiff(t2, 77.02, 28.02, 77.18, 28.18)
    write_dummy_geotiff(sar, 77.01, 28.01, 77.19, 28.19)

    controller = SatQueryController(db=None)

    meta = controller.parse_geotiff_metadata(optical, ["optical"])
    assert isinstance(meta, InputMetadataSchema)
    print("parsed_crs", meta.crs, "bounds", meta.bounds)

    t2_meta = controller.parse_geotiff_metadata(t2, ["optical-t2"])
    controller.validate_spatial_alignment(meta, [t2_meta])
    print("alignment_ok")

    cases = [
        ("What changed between T1 and T2?", True, False, TaskType.BI_TEMPORAL_CHANGE_ANALYSIS),
        ("Highlight industrial rooftops", False, False, TaskType.SINGLE_IMAGE_GROUNDING),
        ("Fuse RISAT with Cartosat for flood extent", False, True, TaskType.CROSS_MODAL_JOINT_ANALYSIS),
        ("Describe land cover in this scene", False, False, TaskType.SINGLE_IMAGE_VQA),
    ]
    for query, has_t2, has_sar, expected in cases:
        predicted = controller.classify_query(query, has_t2=has_t2, has_sar=has_sar)
        print(f"classify {expected.value!r} -> {predicted.value}")
        assert predicted == expected, (query, predicted, expected)

    trace = controller.execute_workflow(
        query="Describe land cover in this scene",
        optical_path=optical,
    )
    print("vqa_trace", trace.trace_id, trace.task, trace.confidence_score)

    change = controller.execute_workflow(
        query="What changed between these dates?",
        optical_path=optical,
        optical_t2_path=t2,
    )
    print("change_trace", change.task, "overlay", controller.last_overlay_uri)

    fused = controller.execute_workflow(
        query="Joint optical-SAR flood analysis",
        optical_path=optical,
        sar_path=sar,
    )
    print("fusion_trace", fused.task, "geojson_features", len((controller.last_geojson or {}).get("features", [])))

    print("pipeline_ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
