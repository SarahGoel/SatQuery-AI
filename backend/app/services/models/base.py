"""On-premise open-weight VLM client (Ollama or vLLM OpenAI-compatible)."""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class VLMResult:
    text: str
    confidence: float
    params: dict[str, Any] = field(default_factory=dict)


class LocalVisionLanguageClient:
    """Talks only to loopback / cluster-local serving. Never hits the public internet."""

    def __init__(self) -> None:
        self.backend = settings.INFERENCE_BACKEND.lower()
        self.model = settings.VLM_MODEL_NAME
        self.timeout = 120.0

    def generate(
        self,
        prompt: str,
        image_path: Path | None = None,
        extra_context: dict[str, Any] | None = None,
    ) -> VLMResult:
        payload_note = extra_context or {}
        try:
            if self.backend == "vllm":
                return self._vllm(prompt, image_path, payload_note)
            return self._ollama(prompt, image_path, payload_note)
        except Exception as exc:  # noqa: BLE001 — keep analyst loop alive without GPU
            logger.warning("vlm_unavailable_using_stub", extra={"error": str(exc)})
            return VLMResult(
                text=(
                    f"[offline stub] Query accepted. Backend `{self.backend}` is unreachable. "
                    f"Stage weights under {settings.LOCAL_MODELS_DIR} and start Ollama/vLLM. "
                    f"Prompt: {prompt[:240]}"
                ),
                confidence=0.35,
                params={"backend": self.backend, "stub": True, "error": str(exc), **payload_note},
            )

    def _ollama(self, prompt: str, image_path: Path | None, extra: dict[str, Any]) -> VLMResult:
        body: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        if image_path is not None and image_path.exists():
            body["images"] = [_b64(image_path)]
        url = f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/generate"
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(url, json=body)
            response.raise_for_status()
            data = response.json()
        text = data.get("response") or data.get("message", {}).get("content") or ""
        return VLMResult(
            text=text.strip() or "(empty VLM response)",
            confidence=0.7,
            params={"backend": "ollama", "model": self.model, **extra},
        )

    def _vllm(self, prompt: str, image_path: Path | None, extra: dict[str, Any]) -> VLMResult:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        if image_path is not None and image_path.exists():
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/tiff;base64,{_b64(image_path)}"},
                }
            )
        url = f"{settings.VLLM_BASE_URL.rstrip('/')}/v1/chat/completions"
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 512,
        }
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(url, json=body)
            response.raise_for_status()
            data = response.json()
        text = data["choices"][0]["message"]["content"]
        return VLMResult(
            text=text.strip(),
            confidence=0.72,
            params={"backend": "vllm", "model": self.model, **extra},
        )


def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")
