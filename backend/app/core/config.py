"""Global configuration for sovereign / air-gapped deployments.

Database URLs, CORS, and local model paths are environment-driven so the
same image runs on a laptop and on an ISRO on-premise GPU node.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_local_models_dir() -> Path:
    env_dir = os.environ.get("LOCAL_MODELS_DIR")
    if env_dir:
        p = Path(env_dir)
        if p.exists():
            return p

    # Docker container root mount
    if os.path.exists("/.dockerenv"):
        if Path("/local_models").exists():
            return Path("/local_models")
        if Path("/app/models").exists():
            return Path("/app/models")

    # Host development candidates (relative to backend/app/core/config.py)
    backend_dir = Path(__file__).resolve().parents[2]
    repo_root = backend_dir.parent
    candidates = [
        Path("/local_models"),
        Path("/app/models"),
        backend_dir / "local_models",
        backend_dir / "models",
        repo_root / "backend" / "local_models",
        repo_root / "backend" / "models",
        repo_root / "local_models",
        repo_root / "models",
        Path("./backend/local_models"),
        Path("./backend/models"),
        Path("./local_models"),
        Path("./models"),
        Path("../local_models"),
        Path("../models"),
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    return Path("/local_models") if os.path.exists("/.dockerenv") else (repo_root / "local_models").resolve()


def _default_database_url() -> str:
    env_db = os.environ.get("DATABASE_URL")
    if env_db:
        return env_db
    if os.path.exists("/.dockerenv"):
        return "postgresql://satquery_admin:isro_secure_db@db:5432/satquery_gis"
    return "postgresql://satquery_admin:isro_secure_db@localhost:5432/satquery_gis"


def _default_ollama_base_url() -> str:
    env_url = os.environ.get("OLLAMA_BASE_URL") or os.environ.get("OLLAMA_HOST")
    if env_url:
        return env_url
    if os.path.exists("/.dockerenv"):
        return "http://ollama:11434"
    return "http://localhost:11434"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )

    SATQUERY_ENV: str = "development"
    DATABASE_URL: str = Field(default_factory=_default_database_url)

    INFERENCE_BACKEND: str = "ollama"  # ollama | vllm
    OLLAMA_BASE_URL: str = Field(default_factory=_default_ollama_base_url)
    OLLAMA_HOST: str | None = None
    VLLM_BASE_URL: str = "http://localhost:8001"
    VLM_MODEL_NAME: str = "llava"

    @property
    def resolved_ollama_url(self) -> str:
        if os.path.exists("/.dockerenv"):
            env_url = (
                os.environ.get("OLLAMA_BASE_URL")
                or os.environ.get("OLLAMA_HOST")
                or self.OLLAMA_HOST
                or self.OLLAMA_BASE_URL
            )
            if (
                env_url
                and "localhost" not in env_url
                and "127.0.0.1" not in env_url
                and "host.docker.internal" not in env_url
            ):
                return env_url.rstrip("/")
            return "http://ollama:11434"

        url = (
            os.environ.get("OLLAMA_BASE_URL")
            or os.environ.get("OLLAMA_HOST")
            or self.OLLAMA_HOST
            or self.OLLAMA_BASE_URL
            or "http://localhost:11434"
        )
        return url.rstrip("/")

    LOCAL_MODELS_DIR: Path = Field(default_factory=_default_local_models_dir)
    SAM_WEIGHTS_PATH: Path | None = None
    MOBILESAM_WEIGHTS_PATH: Path | None = None
    BIGEARTHNET_CHECKPOINT: Path | None = None
    FUSION_CHECKPOINT: Path | None = None
    CHANGE_VQA_CHECKPOINT: Path | None = None

    UPLOAD_DIR: Path = Path("/tmp/satquery_uploads") if os.path.exists("/.dockerenv") or os.name != "nt" else Path("./uploads")
    ARTIFACT_DIR: Path = Path("./artifacts")
    MAX_UPLOAD_BYTES: int = 512 * 1024 * 1024  # 512 MiB GeoTIFF cap

    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:5173"

    ALIGNMENT_IOU_THRESHOLD: float = 0.15
    SIFT_MIN_INLIERS: int = 8
    SPECTRAL_NIR_BAND_INDEX: int = 4
    SPECTRAL_RED_BAND_INDEX: int = 3
    SPECTRAL_GREEN_BAND_INDEX: int = 2
    SPECTRAL_SWIR_BAND_INDEX: int = 5

    @field_validator("LOCAL_MODELS_DIR", "UPLOAD_DIR", "ARTIFACT_DIR", mode="before")
    @classmethod
    def _coerce_path(cls, value: str | Path) -> Path:
        return Path(value)

    @property
    def cors_origin_list(self) -> List[str]:
        return [item.strip() for item in self.CORS_ORIGINS.split(",") if item.strip()]

    def resolved_sam_weights(self) -> Path:
        if self.SAM_WEIGHTS_PATH:
            p = Path(self.SAM_WEIGHTS_PATH)
            if p.exists():
                return p
        primary = self.LOCAL_MODELS_DIR / "sam" / "sam_vit_b.pth"
        if primary.exists():
            return primary
        backend_dir = Path(__file__).resolve().parents[2]
        candidates = [
            backend_dir / "local_models" / "sam" / "sam_vit_b.pth",
            backend_dir / "models" / "sam" / "sam_vit_b.pth",
            backend_dir.parent / "backend" / "local_models" / "sam" / "sam_vit_b.pth",
            backend_dir.parent / "local_models" / "sam" / "sam_vit_b.pth",
            backend_dir.parent / "models" / "sam" / "sam_vit_b.pth",
            Path("/local_models/sam/sam_vit_b.pth"),
            Path("/app/models/sam/sam_vit_b.pth"),
        ]
        for c in candidates:
            if c.exists():
                return c.resolve()
        return primary

    def resolved_mobilesam_weights(self) -> Path:
        if self.MOBILESAM_WEIGHTS_PATH:
            p = Path(self.MOBILESAM_WEIGHTS_PATH)
            if p.exists():
                return p
        primary = self.LOCAL_MODELS_DIR / "sam" / "mobile_sam.pt"
        if primary.exists():
            return primary
        backend_dir = Path(__file__).resolve().parents[2]
        candidates = [
            backend_dir / "local_models" / "sam" / "mobile_sam.pt",
            backend_dir / "models" / "sam" / "mobile_sam.pt",
            backend_dir.parent / "backend" / "local_models" / "sam" / "mobile_sam.pt",
            backend_dir.parent / "local_models" / "sam" / "mobile_sam.pt",
            backend_dir.parent / "models" / "sam" / "mobile_sam.pt",
            Path("/local_models/sam/mobile_sam.pt"),
            Path("/app/models/sam/mobile_sam.pt"),
        ]
        for c in candidates:
            if c.exists():
                return c.resolve()
        return primary

    def resolved_bigearthnet(self) -> Path:
        if self.BIGEARTHNET_CHECKPOINT:
            p = Path(self.BIGEARTHNET_CHECKPOINT)
            if p.exists():
                return p
        primary = self.LOCAL_MODELS_DIR / "bigearthnet" / "checkpoint.pt"
        if primary.exists():
            return primary
        backend_dir = Path(__file__).resolve().parents[2]
        candidates = [
            self.LOCAL_MODELS_DIR / "bigearthnet" / "adapter_model.bin",
            backend_dir / "local_models" / "bigearthnet" / "checkpoint.pt",
            backend_dir / "local_models" / "bigearthnet" / "adapter_model.bin",
            backend_dir / "models" / "bigearthnet" / "checkpoint.pt",
            backend_dir / "models" / "bigearthnet" / "adapter_model.bin",
            backend_dir.parent / "backend" / "local_models" / "bigearthnet" / "checkpoint.pt",
            backend_dir.parent / "backend" / "local_models" / "bigearthnet" / "adapter_model.bin",
            backend_dir.parent / "local_models" / "bigearthnet" / "checkpoint.pt",
            backend_dir.parent / "local_models" / "bigearthnet" / "adapter_model.bin",
            backend_dir.parent / "models" / "bigearthnet" / "checkpoint.pt",
            backend_dir.parent / "models" / "bigearthnet" / "adapter_model.bin",
            Path("/local_models/bigearthnet/checkpoint.pt"),
            Path("/local_models/bigearthnet/adapter_model.bin"),
            Path("/app/models/bigearthnet/checkpoint.pt"),
            Path("/app/models/bigearthnet/adapter_model.bin"),
        ]
        for c in candidates:
            if c.exists():
                return c.resolve()
        return primary

    def resolved_cdvqa(self) -> Path:
        if self.CHANGE_VQA_CHECKPOINT:
            p = Path(self.CHANGE_VQA_CHECKPOINT)
            if p.exists():
                return p
        primary = self.LOCAL_MODELS_DIR / "cdvqa" / "checkpoint.pt"
        if primary.exists():
            return primary
        backend_dir = Path(__file__).resolve().parents[2]
        candidates = [
            self.LOCAL_MODELS_DIR / "cdvqa" / "temporal_attn.pt",
            backend_dir / "local_models" / "cdvqa" / "checkpoint.pt",
            backend_dir / "local_models" / "cdvqa" / "temporal_attn.pt",
            backend_dir / "models" / "cdvqa" / "checkpoint.pt",
            backend_dir / "models" / "cdvqa" / "temporal_attn.pt",
            backend_dir.parent / "backend" / "local_models" / "cdvqa" / "checkpoint.pt",
            backend_dir.parent / "backend" / "local_models" / "cdvqa" / "temporal_attn.pt",
            backend_dir.parent / "local_models" / "cdvqa" / "checkpoint.pt",
            backend_dir.parent / "local_models" / "cdvqa" / "temporal_attn.pt",
            backend_dir.parent / "models" / "cdvqa" / "checkpoint.pt",
            backend_dir.parent / "models" / "cdvqa" / "temporal_attn.pt",
            Path("/local_models/cdvqa/checkpoint.pt"),
            Path("/local_models/cdvqa/temporal_attn.pt"),
            Path("/app/models/cdvqa/checkpoint.pt"),
            Path("/app/models/cdvqa/temporal_attn.pt"),
        ]
        for c in candidates:
            if c.exists():
                return c.resolve()
        return primary


settings = Settings()
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)