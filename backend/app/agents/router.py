"""Hardened Multi-Modal Router & Input Inspector for SatQuery AI.

Enforces deterministic routing constraints across:
1. Single-Image Grounding (RS-Grounding-V3 / MobileSAM) -> 'single_grounding'
2. Single-Image VQA (Remote Sensing VLM) -> 'single_vqa'
3. Bi-Temporal Change Detection (CD-VQA-Pro) -> 'bitemporal_change'
4. Cross-Modal Optical + SAR Analysis (Opt-SAR-Fusion) -> 'cross_modal'

Prevents silent fallbacks and strictly rejects intent-input mismatches.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("SatQueryRouter")

# Mandated 4 core task types
TASK_SINGLE_GROUNDING = "single_grounding"
TASK_SINGLE_VQA = "single_vqa"
TASK_BITEMPORAL_CHANGE = "bitemporal_change"
TASK_CROSS_MODAL = "cross_modal"
TASK_DOMAIN_KNOWLEDGE_QA = "domain_knowledge_qa"

# Controller internal task identifiers
INTERNAL_SINGLE_GROUNDING = "single_image_grounding"
INTERNAL_SINGLE_VQA = "single_image_vqa"
INTERNAL_BITEMPORAL_CHANGE = "bi_temporal_change_analysis"
INTERNAL_CROSS_MODAL = "cross_modal_joint_analysis"
INTERNAL_DOMAIN_KNOWLEDGE_QA = "domain_knowledge_qa"

# Standard mapping: maps all variants to the mandated task types
STANDARDIZED_TASK_MAP: Dict[str, str] = {
    INTERNAL_SINGLE_GROUNDING: TASK_SINGLE_GROUNDING,
    TASK_SINGLE_GROUNDING: TASK_SINGLE_GROUNDING,
    "grounding": TASK_SINGLE_GROUNDING,
    "visual_grounding": TASK_SINGLE_GROUNDING,
    "single_image_grounding": TASK_SINGLE_GROUNDING,
    INTERNAL_SINGLE_VQA: TASK_SINGLE_VQA,
    TASK_SINGLE_VQA: TASK_SINGLE_VQA,
    "vqa": TASK_SINGLE_VQA,
    "single_image_vqa": TASK_SINGLE_VQA,
    INTERNAL_BITEMPORAL_CHANGE: TASK_BITEMPORAL_CHANGE,
    TASK_BITEMPORAL_CHANGE: TASK_BITEMPORAL_CHANGE,
    "change_detection": TASK_BITEMPORAL_CHANGE,
    "bitemporal": TASK_BITEMPORAL_CHANGE,
    "bitemporal_change": TASK_BITEMPORAL_CHANGE,
    "bi_temporal_change_analysis": TASK_BITEMPORAL_CHANGE,
    INTERNAL_CROSS_MODAL: TASK_CROSS_MODAL,
    TASK_CROSS_MODAL: TASK_CROSS_MODAL,
    "fusion": TASK_CROSS_MODAL,
    "cross_modal": TASK_CROSS_MODAL,
    "cross_modal_joint_analysis": TASK_CROSS_MODAL,
    INTERNAL_DOMAIN_KNOWLEDGE_QA: TASK_DOMAIN_KNOWLEDGE_QA,
    TASK_DOMAIN_KNOWLEDGE_QA: TASK_DOMAIN_KNOWLEDGE_QA,
    "domain_knowledge_qa": TASK_DOMAIN_KNOWLEDGE_QA,
    "domain_qa": TASK_DOMAIN_KNOWLEDGE_QA,
    "text_only": TASK_DOMAIN_KNOWLEDGE_QA,
}

INTERNAL_TASK_MAP: Dict[str, str] = {
    TASK_SINGLE_GROUNDING: INTERNAL_SINGLE_GROUNDING,
    TASK_SINGLE_VQA: INTERNAL_SINGLE_VQA,
    TASK_BITEMPORAL_CHANGE: INTERNAL_BITEMPORAL_CHANGE,
    TASK_CROSS_MODAL: INTERNAL_CROSS_MODAL,
    TASK_DOMAIN_KNOWLEDGE_QA: INTERNAL_DOMAIN_KNOWLEDGE_QA,
}

# Temporal phrases and keywords denoting bi-temporal intent
TEMPORAL_PHRASES = [
    "between these two dates",
    "between two dates",
    "between the two dates",
    "between these dates",
    "between dates",
    "between two images",
    "between these two images",
    "between the dates",
    "across both dates",
    "two dates",
    "both dates",
    "before and after",
    "before vs after",
    "t1 vs t2",
    "t1 and t2",
    "newly flooded",
    "new flooding",
    "recent flooding",
    "flood expansion",
    "flood inundation",
    "flood water",
    "flooded areas",
    "flooded area difference",
    "change detection",
    "land cover change",
    "landcover change",
    "urban expansion",
    "water expansion",
    "vegetation loss",
    "built-up growth",
    "what changed",
    "what has changed",
    "changed between",
    "difference between",
    "has the built-up area increased, decreased, or remained unchanged",
    "increased, decreased, or remained unchanged",
    "increased or decreased",
    "remained unchanged",
    "built-up area increased",
    "built-up increased",
]

TEMPORAL_WORDS = {
    "change",
    "changes",
    "changed",
    "difference",
    "differences",
    "differencing",
    "expansion",
    "expanded",
    "transition",
    "growth",
    "reduction",
    "loss",
    "before",
    "after",
    "pre-flood",
    "post-flood",
    "pre-monsoon",
    "post-monsoon",
    "pre",
    "post",
    "t1",
    "t2",
    "dates",
    "flooded",
    "flooding",
    "inundation",
    "inundated",
    "increased",
    "decreased",
    "increase",
    "decrease",
    "unchanged",
}

GROUNDING_TRIGGERS = [
    "ground",
    "highlight",
    "where is",
    "where are",
    "detect",
    "find",
    "locate",
    "outline",
    "box",
    "delineate",
    "segment",
    "identify",
    "tank",
    "storage",
    "silo",
    "rooftop",
    "roof",
    "building",
    "structure",
    "aircraft",
    "airplane",
    "bridge",
    "facility",
]

# Triggers denoting land cover and domain classification (BigEarthNet domain adapter)
LANDCOVER_TRIGGERS = [
    "land cover",
    "landcover",
    "land-cover",
    "land use",
    "terrain",
    "corine",
    "bigearthnet",
    "vegetation",
    "crop",
    "forest",
    "urban fabric",
    "dominant class",
    "surface classification",
    "scene semantics",
    "scene classification",
]


class InputInspectorNode:
    """LangGraph node and deterministic router inspecting input images and analyst intent.
    
    Guarantees:
    - 1 Image + Temporal query -> Explicit ValueError (no silent fallback to grounding).
    - 1 Image + Optical+SAR query -> Explicit ValueError (requires both modalities).
    - 1 Image -> Deterministically routes to SingleImageGrounding or SingleImageVQA.
    - 2 Images + Temporal query -> Strictly routes to BiTemporalChangeDetection.
    - 2 Images + Optical+SAR -> Strictly routes to CrossModalFusionAnalysis.
    - 2 Images + other intent -> Strictly routes to BiTemporalChangeDetection (never defaults to single-image grounding).
    """

    @classmethod
    def inspect(
        cls,
        query: str,
        filepaths: Optional[List[str]] = None,
        parsed_meta: Optional[List[Dict[str, Any]]] = None,
        force_task: Optional[str] = None,
    ) -> str:
        """Inspects inputs and returns the internal controller task name."""
        files = list(filepaths) if filepaths is not None else None
        num_images = len(files) if files is not None else 0
        q = query.lower().strip()

        # Handle forced override if provided
        if force_task:
            normalized = STANDARDIZED_TASK_MAP.get(force_task.lower())
            if normalized:
                target_internal = INTERNAL_TASK_MAP[normalized]
                logger.info("InputInspector: force_task applied -> %s (%s)", normalized, target_internal)
                return target_internal

        # Temporal intent evaluation
        has_temporal_phrase = any(phrase in q for phrase in TEMPORAL_PHRASES)
        has_temporal_token = any(word in q.split() for word in TEMPORAL_WORDS)
        has_temporal = has_temporal_phrase or has_temporal_token or "between" in q

        # Cross-modal intent evaluation
        has_sar_keyword = any(k in q for k in ["sar", "radar", "sentinel-1", "risat", "c-band"])
        has_optical_keyword = any(k in q for k in ["optical", "cartosat", "rgb", "multispectral"])
        has_cross_modal_keyword = (
            "cross-modal" in q
            or "cross modal" in q
            or (has_optical_keyword and has_sar_keyword)
            or ("optical" in q and "radar" in q)
            or "both sensors" in q
            or "combine optical and sar" in q
            or "fusion" in q
        )

        if files is not None and len(files) == 0:
            if has_temporal:
                logger.error("InputInspector REJECTION: 0 images provided but query implies bi-temporal change ('%s')", query)
                raise ValueError(
                    "Bi-temporal change detection requires two spatially aligned images (Before and After). Please attach an image pair to analyze temporal change."
                )
            if has_cross_modal_keyword or (has_optical_keyword and has_sar_keyword):
                logger.error("InputInspector REJECTION: 0 images provided but query implies cross-modal fusion ('%s')", query)
                raise ValueError(
                    "Cross-modal Optical+SAR joint analysis requires both Optical and SAR imagery, but 0 images were provided. Please upload both modalities to execute joint analysis."
                )
            logger.info("InputInspector: 0 images provided in workflow -> classifying as %s", INTERNAL_DOMAIN_KNOWLEDGE_QA)
            return INTERNAL_DOMAIN_KNOWLEDGE_QA

        if files is None:
            if has_cross_modal_keyword or (has_optical_keyword and has_sar_keyword):
                return INTERNAL_CROSS_MODAL
            if has_temporal:
                return INTERNAL_BITEMPORAL_CHANGE
            if any(gt in q for gt in GROUNDING_TRIGGERS):
                return INTERNAL_SINGLE_GROUNDING
            return INTERNAL_SINGLE_VQA

        # Metadata modality check
        meta_list = list(parsed_meta or [])
        has_sar_modality = any("SAR-C-Band" in m.get("modalities", []) for m in meta_list)
        has_optical_modality = any(
            any(b in m.get("modalities", []) for b in ["RGB", "Red", "Green", "Blue", "NIR"])
            for m in meta_list
        )
        is_cross_modal_inputs = (has_sar_modality and has_optical_modality)

        # ==========================================
        # 1 IMAGE CONSTRAINTS
        # ==========================================
        if num_images == 1:
            if has_temporal:
                logger.error(
                    "InputInspector REJECTION: 1 image provided but query implies bi-temporal change ('%s')",
                    query,
                )
                raise ValueError(
                    "Bi-temporal change detection requires two spatially aligned images (Before and After). Please attach an image pair to analyze temporal change."
                )

            if has_cross_modal_keyword or (has_optical_keyword and has_sar_keyword):
                logger.error(
                    "InputInspector REJECTION: 1 image provided but query implies cross-modal Optical+SAR fusion ('%s')",
                    query,
                )
                raise ValueError(
                    "Cross-modal Optical+SAR joint analysis requires both Optical and SAR imagery, "
                    "but only 1 image was provided. Please upload both modalities to execute joint analysis."
                )

            # Route land cover and terrain classification to single_vqa (BigEarthNet domain adapter)
            if any(lt in q for lt in LANDCOVER_TRIGGERS):
                logger.info("InputInspector: 1 image + land-cover intent -> %s (BigEarthNet adapter)", INTERNAL_SINGLE_VQA)
                return INTERNAL_SINGLE_VQA

            # Route to Grounding or VQA
            if any(gt in q for gt in GROUNDING_TRIGGERS):
                logger.info("InputInspector: 1 image + grounding trigger -> %s", INTERNAL_SINGLE_GROUNDING)
                return INTERNAL_SINGLE_GROUNDING

            logger.info("InputInspector: 1 image + scene semantic query -> %s", INTERNAL_SINGLE_VQA)
            return INTERNAL_SINGLE_VQA

        # ==========================================
        # 2+ IMAGES CONSTRAINTS
        # ==========================================
        # Check cross-modal priority
        if is_cross_modal_inputs or has_cross_modal_keyword:
            logger.info("InputInspector: 2 images with Optical+SAR modalities/intent -> %s", INTERNAL_CROSS_MODAL)
            return INTERNAL_CROSS_MODAL

        if has_sar_keyword and not has_temporal_phrase:
            logger.info("InputInspector: 2 images with SAR radar context -> %s", INTERNAL_CROSS_MODAL)
            return INTERNAL_CROSS_MODAL

        # For 2 images, default strictly to bi-temporal change analysis (never fall back to single grounding)
        logger.info(
            "InputInspector: 2 images detected (temporal_intent=%s) -> strictly routing to %s",
            has_temporal,
            INTERNAL_BITEMPORAL_CHANGE,
        )
        return INTERNAL_BITEMPORAL_CHANGE

    @classmethod
    def inspect_semantic(
        cls,
        query: str,
        filepaths: Optional[List[str]] = None,
        parsed_meta: Optional[List[Dict[str, Any]]] = None,
        force_task: Optional[str] = None,
    ):
        from app.agents.semantic_router import SemanticIntentRouter
        return SemanticIntentRouter().route(
            query=query,
            filepaths=filepaths,
            parsed_meta=parsed_meta,
            force_task=force_task,
        )

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """LangGraph callable node interface."""
        query = state.get("query", "")
        filepaths = state.get("filepaths", [])
        parsed_meta = state.get("parsed_meta", [])
        force_task = state.get("force_task")

        semantic_res = self.inspect_semantic(
            query=query,
            filepaths=filepaths,
            parsed_meta=parsed_meta,
            force_task=force_task,
        )
        task = semantic_res.internal_task or self.inspect(
            query=query,
            filepaths=filepaths,
            parsed_meta=parsed_meta,
            force_task=force_task,
        )
        return {
            **state,
            "task": task,
            "task_type": STANDARDIZED_TASK_MAP.get(task, semantic_res.task),
            "intent_classification": semantic_res.to_dict(),
        }
