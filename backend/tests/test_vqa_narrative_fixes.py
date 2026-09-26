"""Tests for Single-Image VQA Narrative Template and Prompt Leak Fixes."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from app.services.agent import SatQueryController
from app.services.heuristic_vlm import clean_user_query_text, generate_heuristic_summary
from app.services.models.rs_vlm import RemoteSensingVLMClient


def _create_synthetic_optical_geotiff(path: Path) -> Path:
    """Creates a 4-band synthetic optical GeoTIFF (R, G, B, NIR)."""
    width, height = 128, 128
    transform = from_bounds(77.0, 28.0, 77.2, 28.2, width, height)
    # Band 1: Red, Band 2: Green, Band 3: Blue, Band 4: NIR
    data = np.zeros((4, height, width), dtype=np.float32)
    data[0] = 0.3  # Red
    data[1] = 0.5  # Green
    data[2] = 0.2  # Blue
    data[3] = 0.6  # NIR

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
        dst.update_tags(SENSOR="Cartosat-2S", MODALITY="Optical/Multispectral")
    return path


class TestVQANarrativeAndPromptLeakFixes:
    """Validates that general VQA and land cover descriptions return natural narratives without prompt leaks or forced boxes."""

    def test_clean_user_query_text_strips_engineered_prompts(self) -> None:
        raw = (
            'User Question: "Describe the land cover in this satellite image."\n\n'
            'Analyze the satellite imagery carefully and provide a direct, authoritative, and non-technical response '
            'to the user\'s question based strictly on the visual evidence.'
        )
        cleaned = clean_user_query_text(raw)
        assert cleaned == "Describe the land cover in this satellite image."

    def test_clean_user_query_text_strips_land_cover_context_tag(self) -> None:
        raw = '[Surface Land Cover Context: Urban fabric, Inland waters] Describe the visible terrain classes'
        cleaned = clean_user_query_text(raw)
        assert cleaned == "Describe the visible terrain classes"

    def test_vqa_heuristic_summary_does_not_use_object_localization_template(self) -> None:
        query = "Describe the land cover and terrain features"
        summary = generate_heuristic_summary(
            query=query,
            task="single_vqa",
            confidence=0.92,
        )
        assert "Object localization completed for" not in summary
        assert "detected and mapped" not in summary
        assert "land cover" in summary.lower() or "terrain" in summary.lower()
        assert "Spectral analysis confirms" not in summary
        assert "NDVI" not in summary

    def test_single_image_vqa_bypasses_forced_bounding_boxes_and_overlays(self, tmp_path: Path) -> None:
        optical_path = _create_synthetic_optical_geotiff(tmp_path / "optical.tif")
        controller = SatQueryController()

        trace = controller.execute_workflow(
            query="Describe the land cover and dominant terrain classes",
            filepaths=[str(optical_path)],
        )

        assert trace.task in ["single_image_vqa", "single_vqa"]
        # Must not generate forced bounding box polygons on general VQA description
        assert controller.last_geojson is None
        assert controller.last_overlay_uri is None
        # Narrative must not contain prompt leakage or object localization templates
        assert "You are an expert remote sensing" not in trace.output
        assert "User Question:" not in trace.output
        assert "Object localization completed for" not in trace.output
        assert "separate Infrastructure Object" not in trace.output

    def test_single_image_grounding_with_action_verb_generates_grounding(self, tmp_path: Path) -> None:
        optical_path = _create_synthetic_optical_geotiff(tmp_path / "optical_grounding.tif")
        controller = SatQueryController()

        trace = controller.execute_workflow(
            query="Highlight the water body in this image",
            filepaths=[str(optical_path)],
        )

        assert trace.task in ["single_image_grounding", "single_grounding"]
        assert controller.last_geojson is not None
        assert len(controller.last_geojson.get("features", [])) >= 1

    def test_dynamic_sensor_metadata_for_unlabeled_imagery(self, tmp_path: Path) -> None:
        path = tmp_path / "untagged.tif"
        width, height = 64, 64
        transform = from_bounds(77.0, 28.0, 77.2, 28.2, width, height)
        data = np.zeros((3, height, width), dtype=np.float32)

        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=3,
            dtype="float32",
            crs="EPSG:4326",
            transform=transform,
        ) as dst:
            dst.write(data)

        controller = SatQueryController()
        meta = controller.parse_geotiff_metadata(str(path))
        assert meta["sensor"] == "Optical Imagery (True Color)"

    def test_vlm_vqa_prompt_structure_and_invocation(self) -> None:
        client = RemoteSensingVLMClient()
        res = client.generate_vqa("Describe the land cover", image_path=None)
        assert res.text is not None
        assert "You are an expert remote sensing" not in res.text
        assert "Spectral analysis confirms" not in res.text
        assert "NDVI:" not in res.text
