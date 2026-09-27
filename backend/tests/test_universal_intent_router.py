"""Tests for Universal Intent Router & Zero-Crash Multi-Tool Pipeline.

Validates:
1. AutonomousIntentRouter archetype classification (SCENE_VQA, FEATURE_GROUNDING,
   BI_TEMPORAL_ANALYSIS, MULTI_MODAL_FUSION, DOMAIN_QA).
2. Point-Prompt Index Guidance (NDWI for water, NDBI for built-up, NDVI for vegetation).
3. Zero-crash graceful error boundary intercepts (ERR_MISSING_SAR_OPTICAL, ERR_MISSING_TEMPORAL_PAIR).
4. Strict 6-7 sentence cohesive narrative formatting across all VLM and heuristic paths.
5. Zero-crash MobileSAM fallback returning null mask without raising exceptions.
6. End-to-end query endpoint integration with zero-crash boundary envelopes.
"""

from __future__ import annotations

import asyncio
import io
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from app.schemas.validation import QueryResponseEnvelope
from app.services.analytical.cross_modal import CrossModalAnalysisTool
from app.services.heuristic_vlm import generate_heuristic_summary
from app.services.models.base import (
    ISRO_EXACT_NARRATIVE_INSTRUCTION,
    get_effective_system_prompt,
)
from app.services.models.grounding import (
    GroundingResult,
    TextGuidedGrounder,
    compute_spectral_index_points,
)
from app.services.models.router import (
    BI_TEMPORAL_ANALYSIS,
    DOMAIN_QA,
    ERR_MISSING_SAR_OPTICAL,
    ERR_MISSING_TEMPORAL_PAIR,
    FEATURE_GROUNDING,
    MULTI_MODAL_FUSION,
    SCENE_VQA,
    AutonomousIntentRouter,
)
from app.services.models.rs_vlm import (
    GEOCHAT_SYSTEM_PROMPT,
    TEMPORAL_VLM_SYSTEM_PROMPT,
    RemoteSensingVLMClient,
)


def count_sentences(text: str) -> int:
    """Accurately count sentences in a paragraph."""
    cleaned = text.strip()
    # Remove bracketed headers like [INCREASED], [DECREASED], [REMAINED UNCHANGED]
    cleaned = re.sub(r"^\[[A-Z\s]+\]\s*(?:Assessment:\s*[A-Za-z]+(?:\s*—|\s*[-:]))?\s*", "", cleaned)
    # Split on sentence terminals followed by whitespace or end of string
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned) if s.strip() and len(s.strip()) > 3]
    return len(sentences)


# ============================================================================
# 1. Autonomous Intent Router Archetype & Index Guidance Tests
# ============================================================================

def test_router_domain_qa_zero_images():
    """Verify 0 images classify as DOMAIN_QA."""
    res = AutonomousIntentRouter.classify("What is the revisit period of Sentinel-1?", num_images=0)
    assert res.archetype == DOMAIN_QA
    assert res.requires_images == 0
    assert res.graceful_error is None


def test_router_scene_vqa_single_image():
    """Verify single image scene overview classifies as SCENE_VQA."""
    res = AutonomousIntentRouter.classify("Describe the land cover and prominent terrain features", num_images=1)
    assert res.archetype == SCENE_VQA
    assert res.index_guidance == "NONE"
    assert res.requires_images == 1
    assert res.graceful_error is None


def test_router_feature_grounding_index_guidance():
    """Verify action verbs and target features resolve correct spectral index guidance."""
    # Water -> NDWI
    res_water = AutonomousIntentRouter.classify("Highlight the water body and delineate reservoir boundaries", num_images=1)
    assert res_water.archetype == FEATURE_GROUNDING
    assert res_water.target_feature == "water"
    assert res_water.index_guidance == "NDWI"

    # Built-up -> NDBI
    res_built = AutonomousIntentRouter.classify("Segment and locate industrial rooftops and buildings", num_images=1)
    assert res_built.archetype == FEATURE_GROUNDING
    assert res_built.target_feature == "built_up"
    assert res_built.index_guidance == "NDBI"

    # Vegetation -> NDVI
    res_veg = AutonomousIntentRouter.classify("Highlight dense forest and agricultural vegetation parcels", num_images=1)
    assert res_veg.archetype == FEATURE_GROUNDING
    assert res_veg.target_feature == "vegetation"
    assert res_veg.index_guidance == "NDVI"


def test_router_missing_temporal_pair_boundary():
    """Verify temporal change with < 2 images yields exact graceful error message."""
    res = AutonomousIntentRouter.classify("What changed between these two dates?", num_images=1)
    assert res.archetype == BI_TEMPORAL_ANALYSIS
    assert res.requires_images == 2
    assert res.graceful_error == ERR_MISSING_TEMPORAL_PAIR
    assert "two temporal images (Before and After)" in res.graceful_error


def test_router_valid_temporal_pair():
    """Verify temporal change with 2 images proceeds without graceful error."""
    res = AutonomousIntentRouter.classify("What changed between these two dates?", num_images=2)
    assert res.archetype == BI_TEMPORAL_ANALYSIS
    assert res.graceful_error is None
    assert "SiameseChangeNet" in res.tool_chain


def test_router_missing_sar_optical_boundary():
    """Verify multi-modal fusion with < 2 images yields exact graceful error message."""
    res = AutonomousIntentRouter.classify("Perform optical and SAR fusion to detect water and buildings", num_images=1)
    assert res.archetype == MULTI_MODAL_FUSION
    assert res.requires_images == 2
    assert res.graceful_error == ERR_MISSING_SAR_OPTICAL
    assert "requires both Optical and SAR imagery" in res.graceful_error


def test_router_valid_sar_optical():
    """Verify multi-modal fusion with 2 images proceeds without graceful error."""
    res = AutonomousIntentRouter.classify("Combine optical and radar observations", num_images=2)
    assert res.archetype == MULTI_MODAL_FUSION
    assert res.graceful_error is None
    assert "OpticalSARFusionTool" in res.tool_chain


# ============================================================================
# 2. Strict 6 to 7 Sentence Mandate & System Prompt Injection Tests
# ============================================================================

def test_isro_exact_narrative_instruction_in_prompts():
    """Verify the exact ISRO mandate is injected into base, GeoChat, and Temporal system prompts."""
    expected_clause = "You must answer the user's query in EXACTLY 6 to 7 sentences."
    assert expected_clause in ISRO_EXACT_NARRATIVE_INSTRUCTION
    assert expected_clause in GEOCHAT_SYSTEM_PROMPT
    assert expected_clause in TEMPORAL_VLM_SYSTEM_PROMPT

    # Verify get_effective_system_prompt injects it when custom prompt is provided
    custom_sys = "You are a custom assistant."
    effective = get_effective_system_prompt(custom_sys)
    assert effective.startswith(ISRO_EXACT_NARRATIVE_INSTRUCTION)
    assert custom_sys in effective


def test_temporal_synthetic_narratives_sentence_count():
    """Verify all synthetic temporal narrative archetypes contain strictly 6 sentences."""
    archetypes = [
        {"classification": "Bi-Temporal Water Body Retreat / Desiccation", "cf": 0.12, "wd": -0.05},
        {"classification": "Bi-Temporal Urban Expansion / Built-up Growth", "cf": 0.08, "wd": 0.0},
        {"classification": "Bi-Temporal Water Expansion / Inundation", "cf": 0.15, "wd": 0.08},
        {"classification": "Bi-Temporal Surface Stability", "cf": 0.02, "wd": 0.0},
        {"classification": "General Surface Alteration", "cf": 0.09, "wd": 0.0},
    ]

    for arch in archetypes:
        narrative = RemoteSensingVLMClient._synthesize_temporal_narrative(
            classification=arch["classification"],
            change_fraction=arch["cf"],
            water_delta=arch["wd"],
            bbox=[77.0, 28.0, 77.2, 28.2],
            query="Analyze surface change between dates",
        )
        count = count_sentences(narrative)
        assert count in (6, 7), f"Failed for {arch['classification']}: count={count}, text={narrative}"


def test_heuristic_fallbacks_sentence_count():
    """Verify heuristic summary fallbacks across domain QA, grounding, and VQA conform to 6-7 sentences."""
    # 1. Domain QA Sentinel-1
    d1 = generate_heuristic_summary(query="What is the revisit period of Sentinel-1?", task="domain_knowledge_qa")
    assert count_sentences(d1) in (6, 7)

    # 2. Domain QA Sentinel-2
    d2 = generate_heuristic_summary(query="Tell me about Sentinel-2 revisit and spectral bands", task="domain_knowledge_qa")
    assert count_sentences(d2) in (6, 7)

    # 3. Domain QA Cartosat
    d3 = generate_heuristic_summary(query="Explain ISRO Cartosat orbit and resolution", task="domain_knowledge_qa")
    assert count_sentences(d3) in (6, 7)

    # 4. Grounding with zero features (SAM failed / low contrast water)
    g_empty = generate_heuristic_summary(
        query="Highlight water body",
        task="single_image_grounding",
        geojson={"type": "FeatureCollection", "features": []},
        metadata={"bounds": [78.0, 20.0, 78.1, 20.1]},
    )
    assert count_sentences(g_empty) in (6, 7)
    assert "pixel coordinates [" not in g_empty  # No raw pixel coordinates leaked

    # 5. Scene VQA
    vqa = generate_heuristic_summary(
        query="Describe the land cover",
        task="single_image_vqa",
        metadata={"bounds": [78.0, 20.0, 78.1, 20.1]},
    )
    assert count_sentences(vqa) in (6, 7)


# ============================================================================
# 3. Spectral Index Point-Prompt Guidance & MobileSAM Zero-Crash Tests
# ============================================================================

def test_spectral_index_point_generation():
    """Verify compute_spectral_index_points generates valid foreground and background coordinate points."""
    img = np.ones((64, 64, 3), dtype=np.uint8) * 128
    # Simulate a bright green / high-NDWI feature in the center
    img[20:40, 20:40, 1] = 230  # High green
    img[20:40, 20:40, 0] = 50   # Low red
    img[20:40, 20:40, 2] = 50   # Low blue

    box_px = [15.0, 15.0, 45.0, 45.0]
    coords, labels = compute_spectral_index_points(img, box_px, index_type="NDWI", top_k=2)

    assert len(coords) == 4  # 2 foreground + 2 background
    assert len(labels) == 4
    assert list(labels) == [1, 1, 0, 0]
    for pt in coords:
        assert 15.0 <= pt[0] <= 45.0
        assert 15.0 <= pt[1] <= 45.0


def test_text_guided_grounder_zero_crash_on_corrupt_input(tmp_path: Path):
    """Verify TextGuidedGrounder returns null mask gracefully on unreadable/corrupt files without crashing."""
    corrupt_file = tmp_path / "corrupt.tif"
    corrupt_file.write_bytes(b"not a valid geotiff image")

    grounder = TextGuidedGrounder()
    result = grounder.ground(
        image_path=corrupt_file,
        prompt="Highlight storage tanks",
        use_mobilesam=True,
        index_guidance="NDBI",
    )
    assert isinstance(result, GroundingResult)
    assert result.mask is None
    assert result.geojson is None
    assert result.params.get("sam_failed") is True


# ============================================================================
# 4. End-to-End Query Pipeline Zero-Crash Boundary Test
# ============================================================================

def test_query_pipeline_intercepts_missing_temporal_pair(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Verify /api/v1/query returns HTTP 200 with graceful guidance envelope when temporal change has 1 image."""
    from fastapi import UploadFile
    import backend.api.routes as routes_mod

    uploads_dir = tmp_path / "uploads"
    artifacts_dir = tmp_path / "artifacts"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(routes_mod.settings, "UPLOAD_DIR", uploads_dir)
    monkeypatch.setattr(routes_mod.settings, "ARTIFACT_DIR", artifacts_dir)

    upload = UploadFile(filename="single_scene.tif", file=io.BytesIO(b"dummy image bytes"))

    async def _run():
        return await routes_mod.query_pipeline(
            query="What changed between these two dates?",
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

    envelope = asyncio.run(_run())
    assert isinstance(envelope, QueryResponseEnvelope)
    assert envelope.status == "ok"
    assert envelope.answer == ERR_MISSING_TEMPORAL_PAIR
    assert envelope.geojson is None
    assert envelope.change_mask is None
    assert envelope.change_overlay_uri is None
