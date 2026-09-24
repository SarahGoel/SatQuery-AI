"""Global test configuration and fixtures for SatQuery AI.

Mocks heavy multi-billion-parameter neural network inference during standard
unit test runs while allowing explicit live model and integration testing.
"""

from __future__ import annotations

from typing import Any
import pytest

try:
    from app.services.models.base import VLMResult
except Exception as exc:
    import traceback
    traceback.print_exc()
    import sys
    print("DEBUG: sys.path is", sys.path)
    print("DEBUG: sys.modules['app'] is", sys.modules.get('app'))
    if 'app' in sys.modules:
        print("DEBUG: app.__path__ is", getattr(sys.modules['app'], '__path__', None))
        print("DEBUG: app.__file__ is", getattr(sys.modules['app'], '__file__', None))
    raise



@pytest.fixture(autouse=True)
def mock_vlm_for_unit_tests(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> None:
    """Mocks LocalVisionLanguageClient.generate for fast, deterministic unit test runs."""
    # Allow tests that specifically probe the client methods to run unmocked
    if (
        "live_model" in request.keywords
        or "test_vlm_client_candidate_endpoints" in request.node.nodeid
        or "test_ollama_payload" in request.node.nodeid
    ):
        return

    def _mock_generate(
        self: Any,
        prompt: str,
        image_path: Any = None,
        images: Any = None,
        extra_context: Any = None,
        **kwargs: Any,
    ) -> VLMResult:
        ctx = dict(extra_context or {})
        if ctx.get("is_directional") and ctx.get("directional_verdict"):
            mock_text = str(ctx["directional_verdict"])
        elif ctx.get("direction_label") and ctx.get("is_directional"):
            mock_text = f"{ctx['direction_label']} Analysis indicates surface alterations."
        else:
            mock_text = f"Satellite observation analysis: '{prompt.strip()[:80]}'. The Sentinel-1 SAR constellation confirms stable surface features."
        return VLMResult(
            text=mock_text,
            confidence=0.92,
            params={"backend": "ollama", "model": "llava", "mocked": True, **ctx},
        )

    monkeypatch.setattr("app.services.models.base.LocalVisionLanguageClient.generate", _mock_generate)
