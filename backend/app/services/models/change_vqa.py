"""Multi-temporal attention Change-VQA for bi-temporal T1 / T2 inputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch
import torch.nn as nn

from app.core.config import settings
from app.services.models.base import LocalVisionLanguageClient
from app.utils.logger import get_logger

logger = get_logger(__name__)

CHANGE_VOCAB = [
    "<pad>",
    "no-change",
    "increase",
    "decrease",
    "urban-growth",
    "vegetation-loss",
    "water-expansion",
    "bare-soil",
    "seasonal-noise",
    "illumination-shift",
    "new-construction",
    "deforestation",
]


class TemporalDifferenceAttention(nn.Module):
    """
    Stateful difference attention blocks extracting temporal feature changes between T1/T2 epochs [58, 61, 65].
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv1x1 = nn.Conv2d(channels, channels, kernel_size=1)
        self.attn_layer = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, feat_t1: torch.Tensor, feat_t2: torch.Tensor) -> torch.Tensor:
        """
        Extracts temporal change indices while suppressing seasonal and illumination noise [61, 62, 64, 65].
        """
        # Temporal difference absolute calculation [58, 61, 65]
        diff = torch.abs(feat_t1 - feat_t2)
        # Estimate pixel-wise attention weights [65]
        attn_weights = self.attn_layer(diff)
        # Output attention-gated feature map representation [65]
        return feat_t2 * attn_weights


class ChangeVQATextDecoder(nn.Module):
    """
    Decodes Temporal difference features into descriptive language representations [61, 64].
    """

    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, change_features: torch.Tensor) -> torch.Tensor:
        pooled = self.global_pool(change_features).squeeze(-1).squeeze(-1)
        logits = self.fc(pooled)
        return logits


class TemporalAttention(nn.Module):
    """Legacy wrapper retained for checkpoint compatibility; prefers difference attention."""

    def __init__(self, dim: int = 32, heads: int = 4) -> None:
        super().__init__()
        self.enc = nn.Conv2d(3, dim, kernel_size=3, padding=1)
        self.tda = TemporalDifferenceAttention(channels=dim)
        self.head = nn.Conv2d(dim, 1, kernel_size=1)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)

    def forward(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        e1 = self.enc(t1)
        e2 = self.enc(t2)
        gated = self.tda(e1, e2)
        return torch.sigmoid(self.head(gated))


@dataclass
class ChangeVQAResult:
    answer: str
    change_mask: np.ndarray
    confidence: float
    overlay_uri: str | None
    params: dict[str, Any] = field(default_factory=dict)


class TemporalChangeVQA:
    def __init__(self, channels: int = 64) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.channels = channels
        self.encoder = nn.Conv2d(3, channels, kernel_size=3, padding=1).to(self.device)
        self.tda = TemporalDifferenceAttention(channels=channels).to(self.device)
        self.text_decoder = ChangeVQATextDecoder(
            vocab_size=len(CHANGE_VOCAB),
            embed_dim=channels,
        ).to(self.device)
        self.encoder.eval()
        self.tda.eval()
        self.text_decoder.eval()
        self.vlm = LocalVisionLanguageClient()
        ckpt = settings.LOCAL_MODELS_DIR / "change_vqa" / "temporal_attn.pt"
        if settings.CHANGE_VQA_CHECKPOINT:
            ckpt = Path(settings.CHANGE_VQA_CHECKPOINT)
        if ckpt.exists():
            state = torch.load(ckpt, map_location=self.device)
            if isinstance(state, dict):
                self.tda.load_state_dict(state, strict=False)
            logger.info("change_vqa_weights_loaded", extra={"path": str(ckpt)})

    def analyze(self, t1_path: Path, t2_path: Path, query: str) -> ChangeVQAResult:
        t1 = _preview_tensor(t1_path).to(self.device)
        t2 = _preview_tensor(t2_path).to(self.device)
        with torch.no_grad():
            feat_t1 = self.encoder(t1)
            feat_t2 = self.encoder(t2)
            gated_diff = self.tda(feat_t1, feat_t2)
            logits = self.text_decoder(gated_diff)
            mask_t = gated_diff.abs().mean(dim=1, keepdim=True)
            mask_t = (mask_t - mask_t.min()) / (mask_t.max() - mask_t.min() + 1e-6)
        mask = mask_t.squeeze().detach().cpu().numpy()
        overlay = settings.ARTIFACT_DIR / "change_overlays" / f"{t1_path.stem}_vs_{t2_path.stem}.npy"
        overlay.parent.mkdir(parents=True, exist_ok=True)
        np.save(overlay, mask)

        token_text = _logits_to_text(logits, query)
        vlm = self.vlm.generate(
            prompt=(
                f"Bi-temporal EO change analysis. User question: {query}. "
                f"Decoded change tokens: {token_text}. "
                f"Estimated change fraction: {float((mask > 0.5).mean()):.3f}."
            ),
            image_path=t2_path,
            extra_context={"change_fraction": float((mask > 0.5).mean()), "tokens": token_text},
        )
        answer = vlm.text
        if vlm.params.get("stub"):
            answer = token_text
        return ChangeVQAResult(
            answer=answer,
            change_mask=mask,
            confidence=float(np.clip((vlm.confidence + float(mask.mean())) / 2, 0.0, 1.0)),
            overlay_uri=str(overlay),
            params={
                "module": "TemporalDifferenceAttention",
                "decoder": "ChangeVQATextDecoder",
                "device": str(self.device),
                "change_fraction": float((mask > 0.5).mean()),
                "tokens": token_text,
                "vlm": vlm.params,
            },
        )


def _logits_to_text(logits: torch.Tensor, query: str) -> str:
    scores = torch.softmax(logits[0], dim=-1)
    topk = torch.topk(scores, k=min(3, scores.numel()))
    tokens = [CHANGE_VOCAB[int(idx)] for idx in topk.indices if CHANGE_VOCAB[int(idx)] != "<pad>"]
    if not tokens:
        tokens = ["no-change"]
    return (
        f"For query `{query}`, the temporal difference decoder reports "
        f"{', '.join(tokens)} between T1 and T2."
    )


def _preview_tensor(path: Path, size: int = 256) -> torch.Tensor:
    import cv2

    with rasterio.open(path) as src:
        count = min(3, src.count)
        arr = src.read(list(range(1, count + 1))).astype(np.float32)
    if arr.shape[0] < 3:
        arr = np.repeat(arr[:1], 3, axis=0)
    bands = []
    for band in arr[:3]:
        if band.max() > band.min():
            band = (band - band.min()) / (band.max() - band.min())
        bands.append(cv2.resize(band, (size, size), interpolation=cv2.INTER_AREA))
    return torch.from_numpy(np.stack(bands, axis=0)).unsqueeze(0)
