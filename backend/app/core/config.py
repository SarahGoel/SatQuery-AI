"""Global configuration for sovereign / air-gapped deployments.

Database URLs, CORS, and local model paths are environment-driven so the
same image runs on a laptop and on an ISRO on-premise GPU node.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )

    SATQUERY_ENV: str = "development"
    DATABASE_URL: str = (
        "postgresql://satquery_admin:isro_secure_db@localhost:5432/satquery_gis"
    )

    INFERENCE_BACKEND: str = "ollama"  # ollama | vllm
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    VLLM_BASE_URL: str = "http://localhost:8001"
    VLM_MODEL_NAME: str = "llava-3b"

    LOCAL_MODELS_DIR: Path = Path("/local_models")
    SAM_WEIGHTS_PATH: Path | None = None
    MOBILESAM_WEIGHTS_PATH: Path | None = None
    BIGEARTHNET_CHECKPOINT: Path | None = None
    FUSION_CHECKPOINT: Path | None = None
    CHANGE_VQA_CHECKPOINT: Path | None = None

    UPLOAD_DIR: Path = Path("./uploads")
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
            return Path(self.SAM_WEIGHTS_PATH)
        return self.LOCAL_MODELS_DIR / "sam" / "sam_vit_b.pth"

    def resolved_mobilesam_weights(self) -> Path:
        if self.MOBILESAM_WEIGHTS_PATH:
            return Path(self.MOBILESAM_WEIGHTS_PATH)
        return self.LOCAL_MODELS_DIR / "sam" / "mobile_sam.pt"

    def resolved_bigearthnet(self) -> Path:
        if self.BIGEARTHNET_CHECKPOINT:
            return Path(self.BIGEARTHNET_CHECKPOINT)
        return self.LOCAL_MODELS_DIR / "bigearthnet" / "checkpoint.pt"


settings = Settings()
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
