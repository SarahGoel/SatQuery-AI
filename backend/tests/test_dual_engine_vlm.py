"""Unit tests for the Dual-Engine VLM Architecture (Gemini / Ollama / vLLM)."""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from PIL import Image
import pytest

from app.core.config import Settings
from app.services.models.base import LocalVisionLanguageClient, VLMResult
from app.services.models.rs_vlm import RemoteSensingVLMClient


@pytest.fixture
def dummy_image_bytes() -> bytes:
    """Generates a small test PNG image as raw bytes."""
    img = Image.new("RGB", (64, 64), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def dummy_image_b64(dummy_image_bytes: bytes) -> str:
    """Generates a small test PNG image as base64 string."""
    return base64.b64encode(dummy_image_bytes).decode("ascii")


def test_gemini_engine_execution_when_configured(monkeypatch: Any, dummy_image_bytes: bytes) -> None:
    """Verify that when VLM_PROVIDER=gemini and GEMINI_API_KEY is present, Gemini is called."""
    monkeypatch.setenv("VLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test_gemini_key_12345")

    mock_client_instance = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "The satellite scene reveals prominent agricultural parcels and a central water reservoir."
    mock_client_instance.models.generate_content.return_value = mock_response

    mock_genai = MagicMock()
    mock_genai.Client.return_value = mock_client_instance
    mock_types = MagicMock()

    with patch("app.services.models.base.genai", mock_genai), \
         patch("app.services.models.base.types", mock_types):
        
        client = LocalVisionLanguageClient()
        result = client.generate(
            prompt="Analyze this satellite scene and describe prominent features.",
            images=[dummy_image_bytes],
        )

        assert isinstance(result, VLMResult)
        assert "agricultural parcels" in result.text
        assert result.params.get("backend") == "gemini"
        assert result.confidence >= 0.90
        assert mock_client_instance.models.generate_content.called


def test_gemini_injects_spatial_context_and_domain_directives(monkeypatch: Any, dummy_image_bytes: bytes) -> None:
    """Verify that geospatial context (bbox, center) and domain directives are injected into Gemini prompt."""
    monkeypatch.setenv("VLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test_gemini_key_12345")

    mock_client_instance = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "Analysis shows prominent agricultural fields and adjacent water bodies."
    mock_client_instance.models.generate_content.return_value = mock_response

    mock_genai = MagicMock()
    mock_genai.Client.return_value = mock_client_instance
    mock_types = MagicMock()

    with patch("app.services.models.base.genai", mock_genai), \
         patch("app.services.models.base.types", mock_types):

        client = LocalVisionLanguageClient()
        result = client.generate(
            prompt="Describe the land cover",
            images=[dummy_image_bytes],
            extra_context={
                "metadata": {
                    "bbox": [78.1, 20.2, 78.9, 20.8],
                    "center": "20.5000°N, 78.5000°E",
                }
            },
        )

        assert mock_client_instance.models.generate_content.called
        call_kwargs = mock_client_instance.models.generate_content.call_args[1]
        sent_contents = call_kwargs["contents"]
        prompt_text = sent_contents[0]

        assert "Geographic Extent: Bounding Box" in prompt_text
        assert "78.1000" in prompt_text
        assert "20.5000°N, 78.5000°E" in prompt_text
        assert "Domain Directives:" in prompt_text
        assert "distinguish irrigated agricultural cropland from dense forest" in prompt_text
        assert "Never categorize water bodies as built-up infrastructure" in prompt_text
        assert "Reference major geographic features" in prompt_text


def test_gemini_fallback_to_ollama_on_exception(monkeypatch: Any, dummy_image_bytes: bytes) -> None:
    """Verify that if Gemini encounters an API error or network exception, it seamlessly falls back to Ollama."""
    monkeypatch.setenv("VLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test_gemini_key_12345")

    mock_client_instance = MagicMock()
    mock_client_instance.models.generate_content.side_effect = RuntimeError("Gemini 429 Quota Exceeded")

    mock_genai = MagicMock()
    mock_genai.Client.return_value = mock_client_instance
    mock_types = MagicMock()

    with patch("app.services.models.base.genai", mock_genai), \
         patch("app.services.models.base.types", mock_types), \
         patch.object(LocalVisionLanguageClient, "_ollama") as mock_ollama:
        
        mock_ollama.return_value = VLMResult(
            text="Ollama fallback: The scene contains dense vegetation and surface water.",
            confidence=0.88,
            params={"backend": "ollama", "model": "llava"},
        )

        client = LocalVisionLanguageClient()
        result = client.generate(
            prompt="Describe the land cover.",
            images=[dummy_image_bytes],
        )

        assert mock_ollama.called
        assert "Ollama fallback" in result.text
        assert result.params.get("backend") == "ollama"


def test_ollama_default_when_vlm_provider_is_ollama(monkeypatch: Any, dummy_image_bytes: bytes) -> None:
    """Verify that when VLM_PROVIDER=ollama, Ollama pipeline is invoked directly."""
    monkeypatch.setenv("VLM_PROVIDER", "ollama")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with patch.object(LocalVisionLanguageClient, "_ollama") as mock_ollama:
        mock_ollama.return_value = VLMResult(
            text="Ollama generated scene description.",
            confidence=0.88,
            params={"backend": "ollama"},
        )

        client = LocalVisionLanguageClient()
        result = client.generate(
            prompt="Describe the scene.",
            images=[dummy_image_bytes],
        )

        assert mock_ollama.called
        assert result.text == "Ollama generated scene description."


def test_missing_api_key_falls_back_to_ollama(monkeypatch: Any, dummy_image_bytes: bytes) -> None:
    """Verify that if VLM_PROVIDER=gemini but GEMINI_API_KEY is unset, it gracefully falls back to Ollama."""
    monkeypatch.setenv("VLM_PROVIDER", "gemini")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    # Mock settings so GEMINI_API_KEY is None
    with patch("app.core.config.settings.GEMINI_API_KEY", None), \
         patch.object(LocalVisionLanguageClient, "_ollama") as mock_ollama:
        
        mock_ollama.return_value = VLMResult(
            text="Ollama fallback output due to missing API key.",
            confidence=0.88,
            params={"backend": "ollama"},
        )

        client = LocalVisionLanguageClient()
        result = client.generate(
            prompt="Analyze satellite imagery.",
            images=[dummy_image_bytes],
        )

        assert mock_ollama.called
        assert "Ollama fallback" in result.text


def test_rs_vlm_client_generate_visual_narrative_alias(dummy_image_bytes: bytes) -> None:
    """Verify that RemoteSensingVLMClient.generate_visual_narrative operates seamlessly."""
    rs_client = RemoteSensingVLMClient()
    
    with patch.object(LocalVisionLanguageClient, "generate") as mock_generate:
        mock_generate.return_value = VLMResult(
            text="Visual narrative: Extensive river basin with surrounding agricultural tracts.",
            confidence=0.95,
            params={"mode": "conversational_text"},
        )

        result = rs_client.generate_visual_narrative(
            prompt="Analyze this remote sensing scene and describe the land cover.",
            images=[dummy_image_bytes],
        )

        assert isinstance(result, VLMResult)
        assert "Extensive river basin" in result.text
        assert mock_generate.called


def test_remote_sensing_vlm_client_gemini_routing(monkeypatch: Any, dummy_image_bytes: bytes) -> None:
    """Verify that RemoteSensingVLMClient switches backend to gemini and model to gemini-3.8-flash."""
    monkeypatch.setenv("VLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test_key_gemini_abc")

    mock_client_instance = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "High-altitude satellite capture displaying a major reservoir with clear shorelines."
    mock_client_instance.models.generate_content.return_value = mock_response

    mock_genai = MagicMock()
    mock_genai.Client.return_value = mock_client_instance
    mock_types = MagicMock()

    with patch("app.services.models.base.genai", mock_genai), \
         patch("app.services.models.base.types", mock_types):
        
        rs_client = RemoteSensingVLMClient()
        assert rs_client.backend == "gemini"
        assert rs_client.model_name in ("gemini-3.8-flash", "gemini-2.5-flash", "gemini-2.0-flash")

        res = rs_client.generate_vqa(
            prompt="Describe the water body.",
            images=[dummy_image_bytes],
        )

        assert isinstance(res, VLMResult)
        assert "major reservoir" in res.text
        assert res.params.get("backend") == "gemini"
        assert res.params.get("model") in ("gemini-3.8-flash", "gemini-2.5-flash", "gemini-2.0-flash")


def test_trace_persistence_prevents_foreign_key_violation() -> None:
    """Verify that SatQueryController._persist_trace maps model names safely to avoid FK violations."""
    from app.services.agent import SatQueryController
    from app.schemas.trace import AuditableTraceLogSchema, InputMetadataSchema, RegistryExecutionSchema

    mock_db = MagicMock()
    mock_db.query.return_value.filter_by.return_value.first.return_value = None

    controller = SatQueryController(db=mock_db)
    
    meta = InputMetadataSchema(
        crs="EPSG:4326",
        bounds=[78.0, 20.0, 78.1, 20.1],
        affine_transform=[0.0001, 0.0, 78.0, 0.0, -0.0001, 20.0],
        modalities=["Optical"],
    )

    trace = AuditableTraceLogSchema(
        trace_id="test-trace-12345",
        task="single_vqa",
        query="Describe land cover",
        timestamp=1700000000.0,
        input_metadata=meta,
        tools_executed=[
            RegistryExecutionSchema(model="RemoteSensingVLMClient", params={"backend": "gemini", "model": "gemini-3.8-flash"}),
            RegistryExecutionSchema(model="WaterGroundingTool", params={"confidence": 0.95}),
        ],
        confidence_score=0.95,
        output="Complete visual narrative",
    )

    controller._persist_trace(trace, meta)
    assert mock_db.commit.called
