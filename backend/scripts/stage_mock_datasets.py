#!/usr/bin/env python3
"""ISRO SIH Synthetic Benchmark Data Stager.

Generates local mock datasets and annotations for the four ISRO public benchmarks:
1. VRSBench (Single-image VQA, captioning, and visual grounding with bboxes)
2. RSVQA (Remote sensing VQA across presence, count, comparison question types)
3. CDVQA (Bitemporal change detection and directional QA with change masks)
4. BigEarthNet (Multi-modal Optical-SAR land-cover classification over 19 classes)

Runs completely offline without external network calls or API keys, writing
256x256 GeoTIFF satellite patches with standard EPSG:4326 WGS84 coordinates and
affine transforms, alongside valid benchmark JSON splits.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import rasterio
from rasterio.transform import from_bounds

# Ensure backend directory is in sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))
repo_root = backend_dir.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("stage_mock_datasets")

BIGEARTHNET_19_CLASSES: List[str] = [
    "Urban fabric",
    "Industrial or commercial units",
    "Arable land",
    "Permanent crops",
    "Pastures",
    "Complex cultivation patterns",
    "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "Agro-forestry areas",
    "Broad-leaved forest",
    "Coniferous forest",
    "Mixed forest",
    "Natural grassland and sparsely vegetated areas",
    "Moors, heathland and sclerophyllous vegetation",
    "Transitional woodland/shrub",
    "Beaches, dunes, sands",
    "Inland wetlands",
    "Coastal wetlands",
    "Inland waters",
    "Marine waters",
]


def create_synthetic_geotiff(
    filepath: Path,
    shape: Tuple[int, int] = (256, 256),
    bands: int = 3,
    dtype: str = "uint8",
    bounds: Tuple[float, float, float, float] = (77.0, 28.0, 77.05, 28.05),
    crs: str = "EPSG:4326",
    pattern: str = "urban",
    seed: Optional[int] = None,
) -> Path:
    """Generates a synthetic 256x256 GeoTIFF patch with WGS84 CRS and affine transform."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    h, w = shape
    west, south, east, north = bounds
    transform = from_bounds(west, south, east, north, w, h)

    rng = np.random.default_rng(seed)

    if pattern == "urban":
        # Base terrain background
        base = rng.uniform(100, 140, (bands, h, w))
        # Add high-contrast building structures and rectangular footprints
        for _ in range(6):
            r1, c1 = rng.integers(16, h - 64), rng.integers(16, w - 64)
            bh, bw = rng.integers(24, 56), rng.integers(24, 56)
            base[:, r1 : r1 + bh, c1 : c1 + bw] = rng.uniform(200, 245)
        # Add intersecting linear roadway grid
        base[:, h // 2 - 4 : h // 2 + 4, :] = 60.0
        base[:, :, w // 2 - 4 : w // 2 + 4] = 60.0
        data = base

    elif pattern == "water":
        # Water body: low reflectance/absorption in optical bands
        data = rng.uniform(120, 160, (bands, h, w))
        # Reservoir in central-west region
        r1, r2 = h // 4, 3 * h // 4
        c1, c2 = w // 8, 5 * w // 8
        data[:, r1:r2, c1:c2] = rng.uniform(20, 45)

    elif pattern == "vegetation":
        # High NIR, moderate green, lower red
        data = np.zeros((bands, h, w), dtype=np.float32)
        if bands >= 4:
            data[0] = rng.uniform(30, 60, (h, w))   # Red (low)
            data[1] = rng.uniform(90, 140, (h, w))  # Green
            data[2] = rng.uniform(30, 50, (h, w))   # Blue
            data[3] = rng.uniform(190, 240, (h, w)) # NIR (high -> strong NDVI)
        else:
            data[0] = rng.uniform(40, 70, (h, w))   # Red
            data[1] = rng.uniform(130, 180, (h, w)) # Green
            data[2] = rng.uniform(40, 70, (h, w))   # Blue

    elif pattern == "sar_water":
        # SAR C-band radar: smooth specular water backscatter (< -18 dB -> very dark)
        base = rng.uniform(110, 170, (bands, h, w))
        # Low specular backscatter region
        base[:, h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = rng.uniform(10, 25)
        data = base

    elif pattern == "sar_urban":
        # Strong double-bounce corner reflectors in urban infrastructure
        base = rng.uniform(70, 120, (bands, h, w))
        for _ in range(8):
            r1, c1 = rng.integers(20, h - 40), rng.integers(20, w - 40)
            bh, bw = rng.integers(10, 25), rng.integers(10, 25)
            base[:, r1 : r1 + bh, c1 : c1 + bw] = rng.uniform(220, 255)
        data = base

    elif pattern == "t1_baseline":
        # T1 baseline for change detection: open agrarian / bare soil parcel
        data = rng.uniform(80, 110, (bands, h, w))

    elif pattern == "t2_developed":
        # T2 follow-up for change detection: new developed buildings installed
        data = rng.uniform(80, 110, (bands, h, w))
        # Changed zone in center-east
        r1, r2 = h // 4, 3 * h // 4
        c1, c2 = w // 2, 7 * w // 8
        data[:, r1:r2, c1:c2] = rng.uniform(210, 245)

    elif pattern == "t2_decreased":
        # T2 follow-up with decreased surface (e.g. water reservoir dried out or deforestation)
        data = rng.uniform(130, 160, (bands, h, w))

    else:
        data = rng.uniform(50, 200, (bands, h, w))

    if dtype == "uint8":
        np_data = np.clip(data, 0, 255).astype(np.uint8)
    else:
        np_data = data.astype(np.float32)

    with rasterio.open(
        filepath,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=bands,
        dtype=dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(np_data)

    return filepath


def create_synthetic_mask(
    filepath: Path,
    shape: Tuple[int, int] = (256, 256),
    bbox: Tuple[int, int, int, int] = (64, 128, 192, 224),
    bounds: Tuple[float, float, float, float] = (77.0, 28.0, 77.05, 28.05),
    crs: str = "EPSG:4326",
) -> Path:
    """Generates a binary 256x256 GeoTIFF change/grounding mask."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    h, w = shape
    west, south, east, north = bounds
    transform = from_bounds(west, south, east, north, w, h)

    mask = np.zeros((1, h, w), dtype=np.uint8)
    r1, c1, r2, c2 = bbox
    r1, r2 = max(0, min(r1, h)), max(0, min(r2, h))
    c1, c2 = max(0, min(c1, w)), max(0, min(c2, w))
    mask[0, r1:r2, c1:c2] = 255

    with rasterio.open(
        filepath,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=1,
        dtype="uint8",
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(mask)

    return filepath


def stage_vrsbench(base_dir: Path, num_samples: int = 8, overwrite: bool = False) -> Dict[str, Any]:
    """Generates synthetic dataset and annotations for VRSBench."""
    vrs_dir = base_dir / "vrsbench"
    img_dir = vrs_dir / "images"
    splits_dir = vrs_dir / "splits"
    split_file = splits_dir / "test_benchmark.json"

    img_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    templates = [
        {
            "filename": "vrs_001.tif",
            "pattern": "urban",
            "question": "Locate the primary aircraft runway on the tarmac.",
            "answer": "The primary aircraft runway is positioned along the central corridor.",
            "bbox": [32.0, 112.0, 224.0, 144.0],
        },
        {
            "filename": "vrs_002.tif",
            "pattern": "urban",
            "question": "Detect the commercial industrial warehouse rooftops.",
            "answer": "Commercial warehouse rooftops are arranged in the northern quadrant.",
            "bbox": [24.0, 24.0, 120.0, 120.0],
        },
        {
            "filename": "vrs_003.tif",
            "pattern": "water",
            "question": "Highlight the municipal water reservoir basin.",
            "answer": "The municipal water reservoir basin is clearly visible in the central-western sector.",
            "bbox": [32.0, 64.0, 160.0, 192.0],
        },
        {
            "filename": "vrs_004.tif",
            "pattern": "urban",
            "question": "Identify the multi-track railway transport interchange.",
            "answer": "The multi-track railway transport interchange crosses the southern portion.",
            "bbox": [40.0, 160.0, 216.0, 220.0],
        },
        {
            "filename": "vrs_005.tif",
            "pattern": "vegetation",
            "question": "Delineate the dense agricultural crop parcel.",
            "answer": "The dense agricultural crop parcel occupies the southeastern sector.",
            "bbox": [128.0, 128.0, 240.0, 240.0],
        },
        {
            "filename": "vrs_006.tif",
            "pattern": "urban",
            "question": "Locate the rectangular residential housing block.",
            "answer": "The residential housing block is clustered in the western sector.",
            "bbox": [20.0, 40.0, 100.0, 180.0],
        },
        {
            "filename": "vrs_007.tif",
            "pattern": "water",
            "question": "Detect the drainage detention canal bordering the road.",
            "answer": "The drainage canal extends adjacent to the primary thoroughfare.",
            "bbox": [60.0, 70.0, 150.0, 160.0],
        },
        {
            "filename": "vrs_008.tif",
            "pattern": "urban",
            "question": "Highlight the electrical substation transformer yard.",
            "answer": "The electrical transformer yard is bounded by high security perimeter fencing.",
            "bbox": [80.0, 80.0, 180.0, 180.0],
        },
    ]

    selected = templates[: max(1, min(num_samples, len(templates)))]
    records: List[Dict[str, Any]] = []

    for idx, tmpl in enumerate(selected, 1):
        tif_path = img_dir / tmpl["filename"]
        if overwrite or not tif_path.exists():
            create_synthetic_geotiff(
                filepath=tif_path,
                shape=(256, 256),
                bands=3,
                dtype="uint8",
                pattern=tmpl["pattern"],
                seed=100 + idx,
            )

        records.append({
            "image": f"images/{tmpl['filename']}",
            "question": tmpl["question"],
            "answer": tmpl["answer"],
            "bbox": tmpl["bbox"],
        })

    with open(split_file, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    logger.info("Staged VRSBench: %d images and annotations at %s", len(records), vrs_dir)
    return {"benchmark": "vrsbench", "staged_samples": len(records), "directory": str(vrs_dir)}


def stage_rsvqa(base_dir: Path, num_samples: int = 8, overwrite: bool = False) -> Dict[str, Any]:
    """Generates synthetic dataset and annotations for RSVQA across multiple question types."""
    rsvqa_dir = base_dir / "rsvqa"
    img_dir = rsvqa_dir / "images"
    splits_dir = rsvqa_dir / "splits"
    split_file = splits_dir / "test_benchmark.json"

    img_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    templates = [
        {
            "filename": "rsvqa_001.tif",
            "pattern": "water",
            "question": "Is there a water body visible in this satellite patch?",
            "answer": "yes",
            "type": "presence",
        },
        {
            "filename": "rsvqa_002.tif",
            "pattern": "vegetation",
            "question": "Are cargo vessels docked in this agricultural zone?",
            "answer": "no",
            "type": "presence",
        },
        {
            "filename": "rsvqa_003.tif",
            "pattern": "urban",
            "question": "How many large industrial structures are present?",
            "answer": "2",
            "type": "count",
        },
        {
            "filename": "rsvqa_004.tif",
            "pattern": "urban",
            "question": "How many airport runways are visible?",
            "answer": "1",
            "type": "count",
        },
        {
            "filename": "rsvqa_005.tif",
            "pattern": "vegetation",
            "question": "Are there more residential buildings than vegetation zones?",
            "answer": "no",
            "type": "comparison",
        },
        {
            "filename": "rsvqa_006.tif",
            "pattern": "water",
            "question": "Is the water reservoir larger than the urban parcel?",
            "answer": "yes",
            "type": "comparison",
        },
        {
            "filename": "rsvqa_007.tif",
            "pattern": "urban",
            "question": "Is this scene predominantly rural or urban?",
            "answer": "urban",
            "type": "rural_urban",
        },
        {
            "filename": "rsvqa_008.tif",
            "pattern": "urban",
            "question": "Is an active transportation roadway crossing the region?",
            "answer": "yes",
            "type": "presence",
        },
    ]

    selected = templates[: max(1, min(num_samples, len(templates)))]
    records: List[Dict[str, Any]] = []

    for idx, tmpl in enumerate(selected, 1):
        tif_path = img_dir / tmpl["filename"]
        if overwrite or not tif_path.exists():
            create_synthetic_geotiff(
                filepath=tif_path,
                shape=(256, 256),
                bands=3,
                dtype="uint8",
                pattern=tmpl["pattern"],
                seed=200 + idx,
            )

        records.append({
            "image": f"images/{tmpl['filename']}",
            "question": tmpl["question"],
            "answer": tmpl["answer"],
            "type": tmpl["type"],
        })

    with open(split_file, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    logger.info("Staged RSVQA: %d images and annotations at %s", len(records), rsvqa_dir)
    return {"benchmark": "rsvqa", "staged_samples": len(records), "directory": str(rsvqa_dir)}


def stage_cdvqa(base_dir: Path, num_samples: int = 8, overwrite: bool = False) -> Dict[str, Any]:
    """Generates synthetic bitemporal datasets, masks, and annotations for CDVQA."""
    cdvqa_dir = base_dir / "cdvqa"
    t1_dir = cdvqa_dir / "t1"
    t2_dir = cdvqa_dir / "t2"
    masks_dir = cdvqa_dir / "masks"
    splits_dir = cdvqa_dir / "splits"
    split_file = splits_dir / "test_benchmark.json"

    t1_dir.mkdir(parents=True, exist_ok=True)
    t2_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    templates = [
        {
            "id": "cdvqa_001",
            "p1": "t1_baseline",
            "p2": "t2_developed",
            "question": "What land cover changes occurred between the initial and follow-up observation dates?",
            "answer": "Residential building infrastructure has [INCREASED] across the cleared agricultural sector.",
            "bbox": (64, 128, 192, 224),
        },
        {
            "id": "cdvqa_002",
            "p1": "water",
            "p2": "t2_decreased",
            "question": "Did the reservoir surface water extent expand or contract between observation passes?",
            "answer": "Water surface area has [DECREASED] due to seasonal evaporation.",
            "bbox": (64, 32, 192, 160),
        },
        {
            "id": "cdvqa_003",
            "p1": "vegetation",
            "p2": "t2_decreased",
            "question": "What change is observable in the dense forest tract between observation years?",
            "answer": "Forest canopy density has [DECREASED] following logging activities.",
            "bbox": (32, 32, 128, 128),
        },
        {
            "id": "cdvqa_004",
            "p1": "t1_baseline",
            "p2": "t2_developed",
            "question": "How did industrial warehouse expansion develop over the monitoring interval?",
            "answer": "Industrial logistics facilities have [INCREASED] in the eastern corridor.",
            "bbox": (80, 110, 200, 220),
        },
        {
            "id": "cdvqa_005",
            "p1": "urban",
            "p2": "urban",
            "question": "Did the coastal shoreline or dock structures alter over the bitemporal period?",
            "answer": "The shoreline and docking installations [REMAINED UNCHANGED] during this period.",
            "bbox": (0, 0, 0, 0),
        },
        {
            "id": "cdvqa_006",
            "p1": "t1_baseline",
            "p2": "t2_developed",
            "question": "What is the status of the urban highway development across the two passes?",
            "answer": "Paved road network length has [INCREASED] through the newly leveled terrain.",
            "bbox": (100, 40, 160, 220),
        },
        {
            "id": "cdvqa_007",
            "p1": "t1_baseline",
            "p2": "vegetation",
            "question": "Have vegetation indices changed in the agricultural parcels between T1 and T2?",
            "answer": "Crop cultivation and green canopy coverage has [INCREASED] post-monsoon.",
            "bbox": (40, 40, 200, 200),
        },
        {
            "id": "cdvqa_008",
            "p1": "water",
            "p2": "t2_decreased",
            "question": "Did river floodplain inundation increase or recede between the sensor acquisitions?",
            "answer": "Flood inundation extent has [DECREASED] as waters receded back to main channel.",
            "bbox": (48, 64, 180, 160),
        },
    ]

    selected = templates[: max(1, min(num_samples, len(templates)))]
    records: List[Dict[str, Any]] = []

    for idx, tmpl in enumerate(selected, 1):
        t1_path = t1_dir / f"{tmpl['id']}_t1.tif"
        t2_path = t2_dir / f"{tmpl['id']}_t2.tif"
        mask_path = masks_dir / f"{tmpl['id']}_mask.tif"

        if overwrite or not t1_path.exists():
            create_synthetic_geotiff(
                filepath=t1_path,
                shape=(256, 256),
                bands=3,
                dtype="uint8",
                pattern=tmpl["p1"],
                seed=300 + idx,
            )

        if overwrite or not t2_path.exists():
            create_synthetic_geotiff(
                filepath=t2_path,
                shape=(256, 256),
                bands=3,
                dtype="uint8",
                pattern=tmpl["p2"],
                seed=400 + idx,
            )

        if overwrite or not mask_path.exists():
            create_synthetic_mask(
                filepath=mask_path,
                shape=(256, 256),
                bbox=tmpl["bbox"],
            )

        records.append({
            "image_t1": f"t1/{tmpl['id']}_t1.tif",
            "image_t2": f"t2/{tmpl['id']}_t2.tif",
            "question": tmpl["question"],
            "answer": tmpl["answer"],
            "mask": f"masks/{tmpl['id']}_mask.tif",
        })

    with open(split_file, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    logger.info("Staged CDVQA: %d bitemporal pairs and annotations at %s", len(records), cdvqa_dir)
    return {"benchmark": "cdvqa", "staged_samples": len(records), "directory": str(cdvqa_dir)}


def stage_bigearthnet(base_dir: Path, num_samples: int = 8, overwrite: bool = False) -> Dict[str, Any]:
    """Generates synthetic multi-modal Optical-SAR dataset and annotations for BigEarthNet."""
    ben_dir = base_dir / "bigearthnet"
    s2_dir = ben_dir / "sentinel2"
    s1_dir = ben_dir / "sentinel1"
    splits_dir = ben_dir / "splits"
    split_file = splits_dir / "test_benchmark.json"

    s2_dir.mkdir(parents=True, exist_ok=True)
    s1_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    templates = [
        {
            "id": "ben_001",
            "opt_pattern": "urban",
            "sar_pattern": "sar_urban",
            "labels": ["Urban fabric", "Industrial or commercial units"],
        },
        {
            "id": "ben_002",
            "opt_pattern": "water",
            "sar_pattern": "sar_water",
            "labels": ["Inland waters", "Inland wetlands"],
        },
        {
            "id": "ben_003",
            "opt_pattern": "vegetation",
            "sar_pattern": "sar_uniform",
            "labels": ["Broad-leaved forest", "Mixed forest"],
        },
        {
            "id": "ben_004",
            "opt_pattern": "vegetation",
            "sar_pattern": "sar_uniform",
            "labels": ["Arable land", "Complex cultivation patterns"],
        },
        {
            "id": "ben_005",
            "opt_pattern": "vegetation",
            "sar_pattern": "sar_uniform",
            "labels": ["Pastures", "Natural grassland and sparsely vegetated areas"],
        },
        {
            "id": "ben_006",
            "opt_pattern": "urban",
            "sar_pattern": "sar_urban",
            "labels": ["Urban fabric", "Complex cultivation patterns"],
        },
        {
            "id": "ben_007",
            "opt_pattern": "vegetation",
            "sar_pattern": "sar_uniform",
            "labels": ["Coniferous forest", "Transitional woodland/shrub"],
        },
        {
            "id": "ben_008",
            "opt_pattern": "water",
            "sar_pattern": "sar_water",
            "labels": ["Inland waters", "Urban fabric"],
        },
    ]

    selected = templates[: max(1, min(num_samples, len(templates)))]
    records: List[Dict[str, Any]] = []

    for idx, tmpl in enumerate(selected, 1):
        opt_path = s2_dir / f"{tmpl['id']}_opt.tif"
        sar_path = s1_dir / f"{tmpl['id']}_sar.tif"

        # Optical patch: 4 bands (Red, Green, Blue, NIR)
        if overwrite or not opt_path.exists():
            create_synthetic_geotiff(
                filepath=opt_path,
                shape=(256, 256),
                bands=4,
                dtype="uint8",
                pattern=tmpl["opt_pattern"],
                seed=500 + idx,
            )

        # SAR patch: 2 bands (VV, VH amplitude)
        if overwrite or not sar_path.exists():
            create_synthetic_geotiff(
                filepath=sar_path,
                shape=(256, 256),
                bands=2,
                dtype="uint8",
                pattern=tmpl["sar_pattern"],
                seed=600 + idx,
            )

        records.append({
            "image_opt": f"sentinel2/{tmpl['id']}_opt.tif",
            "image_sar": f"sentinel1/{tmpl['id']}_sar.tif",
            "labels": tmpl["labels"],
        })

    with open(split_file, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    logger.info("Staged BigEarthNet: %d Optical-SAR pairs and annotations at %s", len(records), ben_dir)
    return {"benchmark": "bigearthnet", "staged_samples": len(records), "directory": str(ben_dir)}


def stage_all_benchmarks(
    base_dir: Path,
    num_samples: int = 8,
    overwrite: bool = False,
    benchmarks: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Stages mock datasets for specified or all benchmarks."""
    target_benchmarks = [b.lower() for b in (benchmarks or ["all"])]
    if "all" in target_benchmarks:
        target_benchmarks = ["vrsbench", "rsvqa", "cdvqa", "bigearthnet"]

    results: Dict[str, Any] = {}

    if "vrsbench" in target_benchmarks:
        results["vrsbench"] = stage_vrsbench(base_dir, num_samples=num_samples, overwrite=overwrite)
    if "rsvqa" in target_benchmarks:
        results["rsvqa"] = stage_rsvqa(base_dir, num_samples=num_samples, overwrite=overwrite)
    if "cdvqa" in target_benchmarks:
        results["cdvqa"] = stage_cdvqa(base_dir, num_samples=num_samples, overwrite=overwrite)
    if "bigearthnet" in target_benchmarks:
        results["bigearthnet"] = stage_bigearthnet(base_dir, num_samples=num_samples, overwrite=overwrite)

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage mock synthetic benchmark datasets for SatQuery AI")
    parser.add_argument(
        "--base-dir",
        type=str,
        default=None,
        help="Root base directory for raw benchmark data (default: backend/data/raw)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=8,
        help="Number of synthetic samples per benchmark (range 5 to 10, default: 8)",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["all"],
        choices=["vrsbench", "rsvqa", "cdvqa", "bigearthnet", "all"],
        help="List of benchmarks to stage (default: all)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing staged image and split files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Determine default base directory: backend/data/raw
    if args.base_dir:
        base_dir = Path(args.base_dir).resolve()
    else:
        # Default to backend/data/raw
        base_dir = (backend_dir / "data" / "raw").resolve()

    logger.info("Staging synthetic benchmark datasets under: %s", base_dir)
    res = stage_all_benchmarks(
        base_dir=base_dir,
        num_samples=args.num_samples,
        overwrite=args.overwrite,
        benchmarks=args.benchmarks,
    )
    print("\n" + "=" * 60)
    print("SATQUERY AI: SYNTHETIC BENCHMARK DATA STAGING COMPLETE")
    print("=" * 60)
    for b_name, meta in res.items():
        print(f"- {b_name.upper():12}: {meta['staged_samples']} samples staged -> {meta['directory']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
