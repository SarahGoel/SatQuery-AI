"""Autonomous Intent Router for SatQuery AI.

Enforces deterministic classification into 4 core ISRO SIH query archetypes:
1. SCENE_VQA: Single-image scene description, LULC identification, general VQA.
2. FEATURE_GROUNDING: Specific feature/object localization using MobileSAM guided
   by spectral index point-prompts (NDWI for water, NDBI for built-up, NDVI for vegetation).
3. BI_TEMPORAL_ANALYSIS: Multi-date change detection requiring exactly 2 temporal images
   with mathematical pixel differencing context passed to the VLM.
4. MULTI_MODAL_FUSION: Optical + SAR joint analysis comparing surface reflectance against
   radar backscatter physics, requiring both sensor modalities.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.config import settings

logger = logging.getLogger("AutonomousIntentRouter")

# Core Archetype Identifiers
SCENE_VQA = "SCENE_VQA"
FEATURE_GROUNDING = "FEATURE_GROUNDING"
BI_TEMPORAL_ANALYSIS = "BI_TEMPORAL_ANALYSIS"
MULTI_MODAL_FUSION = "MULTI_MODAL_FUSION"
DOMAIN_QA = "DOMAIN_QA"

# Internal Controller Task Mapping
ARCHETYPE_TO_INTERNAL = {
    SCENE_VQA: "single_image_vqa",
    FEATURE_GROUNDING: "single_image_grounding",
    BI_TEMPORAL_ANALYSIS: "bi_temporal_change_analysis",
    MULTI_MODAL_FUSION: "cross_modal_joint_analysis",
    DOMAIN_QA: "domain_knowledge_qa",
}

INTERNAL_TO_ARCHETYPE = {v: k for k, v in ARCHETYPE_TO_INTERNAL.items()}
INTERNAL_TO_ARCHETYPE["single_vqa"] = SCENE_VQA
INTERNAL_TO_ARCHETYPE["single_grounding"] = FEATURE_GROUNDING
INTERNAL_TO_ARCHETYPE["bitemporal_change"] = BI_TEMPORAL_ANALYSIS
INTERNAL_TO_ARCHETYPE["cross_modal"] = MULTI_MODAL_FUSION

# Graceful Error Messages for Missing Modality / Image Count Boundaries
ERR_MISSING_SAR_OPTICAL = (
    "Unable to complete analysis: This query requires both Optical and SAR imagery, "
    "but only one source was provided."
)
ERR_MISSING_TEMPORAL_PAIR = (
    "Unable to complete analysis: This query requires two temporal images (Before and After), "
    "but only one source was provided."
)


@dataclass
class IntentRoutingResult:
    archetype: str
    target_feature: str = "general"
    index_guidance: str = "NONE"  # NDWI | NDBI | NDVI | NONE
    confidence: float = 0.95
    reasoning: str = ""
    internal_task: str = ""
    requires_images: int = 1
    graceful_error: Optional[str] = None
    tool_chain: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "archetype": self.archetype,
            "target_feature": self.target_feature,
            "index_guidance": self.index_guidance,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "internal_task": self.internal_task or ARCHETYPE_TO_INTERNAL.get(self.archetype, "single_image_vqa"),
            "requires_images": self.requires_images,
            "graceful_error": self.graceful_error,
            "tool_chain": self.tool_chain,
        }


class AutonomousIntentRouter:
    """Universal Autonomous Intent Router for multimodal Earth Observation queries."""

    # Keywords for Multi-Modal Fusion (Optical + SAR)
    FUSION_TRIGGERS = [
        "optical and sar",
        "sar and optical",
        "optical and radar",
        "radar and optical",
        "both sensors",
        "cross-modal",
        "cross modal",
        "combine optical and sar",
        "optical+sar",
        "sar+optical",
        "radar backscatter and optical",
        "reflectance and backscatter",
        "c-band and optical",
        "fusion",
    ]

    # Keywords for Bi-Temporal Change Detection
    TEMPORAL_TRIGGERS = [
        "what changed",
        "what has changed",
        "changed between",
        "change between",
        "difference between",
        "has the built-up area increased",
        "has the water increased",
        "has the water decreased",
        "increased or decreased",
        "increased, decreased, or remained unchanged",
        "increased",
        "decreased",
        "remained unchanged",
        "between these two dates",
        "between two dates",
        "between the two dates",
        "between these dates",
        "between dates",
        "between two images",
        "between these two images",
        "before and after",
        "before vs after",
        "t1 vs t2",
        "t1 and t2",
        "newly flooded",
        "receding water",
        "desiccation",
        "temporal change",
        "change detection",
    ]

    # Action verbs for Feature Grounding
    GROUNDING_VERBS = [
        "highlight",
        "segment",
        "locate",
        "delineate",
        "point out",
        "draw a box",
        "bounding box",
        "isolate",
        "outline",
        "where is",
        "where are",
        "find the",
    ]

    # Specific Target Feature Categorization
    WATER_FEATURES = [
        "water",
        "water body",
        "water bodies",
        "lake",
        "lakes",
        "river",
        "rivers",
        "reservoir",
        "reservoirs",
        "pond",
        "ponds",
        "canal",
        "wetland",
        "flood",
        "floodwater",
        "inundation",
    ]

    BUILTUP_FEATURES = [
        "built-up",
        "built up",
        "building",
        "buildings",
        "urban",
        "urban fabric",
        "rooftop",
        "rooftops",
        "industrial roof",
        "storage tank",
        "tanks",
        "infrastructure",
        "residential",
        "facility",
        "aircraft",
        "runway",
        "bridge",
        "road",
    ]

    VEGETATION_FEATURES = [
        "vegetation",
        "forest",
        "dense forest",
        "broad-leaved forest",
        "trees",
        "crop",
        "crops",
        "cropland",
        "agricultural",
        "agriculture",
        "farmland",
        "greenery",
        "pastures",
    ]

    @classmethod
    def resolve_index_guidance(cls, query: str) -> tuple[str, str]:
        """Resolves target feature category and corresponding spectral index guidance.
        
        Returns:
            (target_feature, index_guidance) where index_guidance in ["NDWI", "NDBI", "NDVI", "NONE"]
        """
        q = query.lower()
        if any(w in q for w in cls.WATER_FEATURES):
            return "water", "NDWI"
        if any(w in q for w in cls.BUILTUP_FEATURES):
            return "built_up", "NDBI"
        if any(w in q for w in cls.VEGETATION_FEATURES):
            return "vegetation", "NDVI"
        return "general", "NONE"

    @classmethod
    def classify(
        cls,
        query: str,
        num_images: int = 0,
        parsed_meta: Optional[List[Dict[str, Any]]] = None,
        force_task: Optional[str] = None,
    ) -> IntentRoutingResult:
        """Classify user query and available assets into an executable archetype."""
        q = (query or "").lower().strip()
        target_feature, index_guidance = cls.resolve_index_guidance(q)

        # 0. Forced task override if explicitly specified
        if force_task:
            task_norm = force_task.lower()
            if task_norm in ARCHETYPE_TO_INTERNAL:
                arch = task_norm
            else:
                arch = INTERNAL_TO_ARCHETYPE.get(task_norm, SCENE_VQA)
            return IntentRoutingResult(
                archetype=arch,
                target_feature=target_feature,
                index_guidance=index_guidance,
                confidence=1.0,
                reasoning=f"Task forced by caller: {force_task}",
                internal_task=ARCHETYPE_TO_INTERNAL.get(arch, "single_image_vqa"),
                tool_chain=["RemoteSensingVLMClient"],
            )

        # 1. Text-Only / 0 images check
        if num_images == 0:
            return IntentRoutingResult(
                archetype=DOMAIN_QA,
                target_feature=target_feature,
                index_guidance="NONE",
                confidence=0.92,
                reasoning="No satellite imagery provided; conversational Earth Observation QA",
                internal_task=ARCHETYPE_TO_INTERNAL[DOMAIN_QA],
                requires_images=0,
                tool_chain=["RemoteSensingVLMClient"],
            )

        # 2. Check Multi-Modal Fusion Intent
        has_sar_kw = any(k in q for k in ["sar", "radar", "sentinel-1", "risat", "backscatter"])
        has_optical_kw = any(k in q for k in ["optical", "cartosat", "rgb", "multispectral", "reflectance"])
        is_fusion_intent = any(trig in q for trig in cls.FUSION_TRIGGERS) or (has_sar_kw and has_optical_kw)

        if is_fusion_intent:
            if num_images < 2:
                logger.warning("Zero-Crash: Fusion intent with only %d image(s)", num_images)
                return IntentRoutingResult(
                    archetype=MULTI_MODAL_FUSION,
                    target_feature=target_feature,
                    index_guidance=index_guidance,
                    confidence=0.95,
                    reasoning="Query requests Optical and SAR fusion, but fewer than 2 images provided",
                    internal_task=ARCHETYPE_TO_INTERNAL[MULTI_MODAL_FUSION],
                    requires_images=2,
                    graceful_error=ERR_MISSING_SAR_OPTICAL,
                    tool_chain=["OpticalSARFusionTool", "RemoteSensingVLMClient"],
                )
            return IntentRoutingResult(
                archetype=MULTI_MODAL_FUSION,
                target_feature=target_feature,
                index_guidance=index_guidance,
                confidence=0.96,
                reasoning="Optical and SAR multimodal joint feature fusion with backscatter physics",
                internal_task=ARCHETYPE_TO_INTERNAL[MULTI_MODAL_FUSION],
                requires_images=2,
                tool_chain=["SpatialAligner", "OpticalSARFusionTool", "RemoteSensingVLMClient"],
            )

        # 3. Check Bi-Temporal Change Intent
        is_temporal_intent = any(trig in q for trig in cls.TEMPORAL_TRIGGERS)
        if is_temporal_intent:
            if num_images < 2:
                logger.warning("Zero-Crash: Bi-temporal change intent with only %d image(s)", num_images)
                return IntentRoutingResult(
                    archetype=BI_TEMPORAL_ANALYSIS,
                    target_feature=target_feature,
                    index_guidance=index_guidance if index_guidance != "NONE" else "NDWI",
                    confidence=0.95,
                    reasoning="Query asks for temporal change comparison, but only 1 image provided",
                    internal_task=ARCHETYPE_TO_INTERNAL[BI_TEMPORAL_ANALYSIS],
                    requires_images=2,
                    graceful_error=ERR_MISSING_TEMPORAL_PAIR,
                    tool_chain=["TemporalChangeTool", "RemoteSensingVLMClient"],
                )
            return IntentRoutingResult(
                archetype=BI_TEMPORAL_ANALYSIS,
                target_feature=target_feature,
                index_guidance=index_guidance if index_guidance != "NONE" else "NDWI",
                confidence=0.95,
                reasoning="Bi-temporal pixel differencing and directional alteration evaluation",
                internal_task=ARCHETYPE_TO_INTERNAL[BI_TEMPORAL_ANALYSIS],
                requires_images=2,
                tool_chain=["SpatialAligner", "SiameseChangeNet", "TemporalChangeTool", "RemoteSensingVLMClient"],
            )

        # 4. If 2 or more images provided without explicit fusion or change trigger -> Bi-temporal default
        if num_images >= 2:
            return IntentRoutingResult(
                archetype=BI_TEMPORAL_ANALYSIS,
                target_feature=target_feature,
                index_guidance=index_guidance if index_guidance != "NONE" else "NDWI",
                confidence=0.90,
                reasoning="Multiple images provided; defaulting to bi-temporal comparison analysis",
                internal_task=ARCHETYPE_TO_INTERNAL[BI_TEMPORAL_ANALYSIS],
                requires_images=2,
                tool_chain=["SpatialAligner", "SiameseChangeNet", "TemporalChangeTool", "RemoteSensingVLMClient"],
            )

        # 5. Single-Image Workflows (1 image provided)
        # Check Grounding Action Verbs vs General VQA Description
        is_grounding = any(verb in q for verb in cls.GROUNDING_VERBS) or any(
            t in q for t in ["water body", "lake", "reservoir", "storage tank", "building", "where is", "where are", "draw a box"]
        )

        # If general description / land cover overview without specific localization request -> SCENE_VQA
        is_vqa_overview = any(w in q for w in ["describe", "what is this", "what does this", "overview", "land cover", "terrain", "scene semantics"])
        if is_vqa_overview and not any(v in q for v in ["highlight", "segment", "locate", "point out", "draw a box"]):
            return IntentRoutingResult(
                archetype=SCENE_VQA,
                target_feature=target_feature,
                index_guidance="NONE",
                confidence=0.95,
                reasoning="Single-image scene visual description and land-cover taxonomy",
                internal_task=ARCHETYPE_TO_INTERNAL[SCENE_VQA],
                requires_images=1,
                tool_chain=["RemoteSensingVLMClient"],
            )

        if is_grounding:
            return IntentRoutingResult(
                archetype=FEATURE_GROUNDING,
                target_feature=target_feature,
                index_guidance=index_guidance,
                confidence=0.94,
                reasoning=f"Feature grounding targeting {target_feature} with {index_guidance} guidance",
                internal_task=ARCHETYPE_TO_INTERNAL[FEATURE_GROUNDING],
                requires_images=1,
                tool_chain=["TextGuidedGrounder", "MobileSAM", "GeodesicMeasurementTool", "RemoteSensingVLMClient"],
            )

        # Default single image query -> SCENE_VQA
        return IntentRoutingResult(
            archetype=SCENE_VQA,
            target_feature=target_feature,
            index_guidance="NONE",
            confidence=0.92,
            reasoning="Default single-image visual reasoning and question answering",
            internal_task=ARCHETYPE_TO_INTERNAL[SCENE_VQA],
            requires_images=1,
            tool_chain=["RemoteSensingVLMClient"],
        )
