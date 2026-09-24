"""Semantic Intent Router for SatQuery AI.

Replaces procedural keyword matching with an autonomous LLM-based intent parser
using Ollama's /api/chat endpoint with structured JSON output format.
Extracts task, target features, tool chain, confidence, and reasoning.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.config import settings

logger = logging.getLogger("SemanticIntentRouter")

# Standard task types
TASK_SINGLE_VQA = "single_vqa"
TASK_VISUAL_GROUNDING = "visual_grounding"
TASK_SINGLE_GROUNDING = "single_grounding"
TASK_BITEMPORAL_CHANGE = "bitemporal_change"
TASK_CROSS_MODAL = "cross_modal"
TASK_DOMAIN_KNOWLEDGE_QA = "domain_knowledge_qa"

# Controller internal task identifiers
INTERNAL_SINGLE_GROUNDING = "single_image_grounding"
INTERNAL_SINGLE_VQA = "single_image_vqa"
INTERNAL_BITEMPORAL_CHANGE = "bi_temporal_change_analysis"
INTERNAL_CROSS_MODAL = "cross_modal_joint_analysis"
INTERNAL_DOMAIN_KNOWLEDGE_QA = "domain_knowledge_qa"

TASK_TO_INTERNAL: Dict[str, str] = {
    TASK_VISUAL_GROUNDING: INTERNAL_SINGLE_GROUNDING,
    TASK_SINGLE_GROUNDING: INTERNAL_SINGLE_GROUNDING,
    "grounding": INTERNAL_SINGLE_GROUNDING,
    TASK_SINGLE_VQA: INTERNAL_SINGLE_VQA,
    "vqa": INTERNAL_SINGLE_VQA,
    TASK_BITEMPORAL_CHANGE: INTERNAL_BITEMPORAL_CHANGE,
    "change_detection": INTERNAL_BITEMPORAL_CHANGE,
    TASK_CROSS_MODAL: INTERNAL_CROSS_MODAL,
    "fusion": INTERNAL_CROSS_MODAL,
    TASK_DOMAIN_KNOWLEDGE_QA: INTERNAL_DOMAIN_KNOWLEDGE_QA,
    "domain_qa": INTERNAL_DOMAIN_KNOWLEDGE_QA,
    "text_only": INTERNAL_DOMAIN_KNOWLEDGE_QA,
}

STANDARDIZED_TASKS = {
    INTERNAL_SINGLE_GROUNDING: TASK_SINGLE_GROUNDING,
    TASK_VISUAL_GROUNDING: TASK_SINGLE_GROUNDING,
    TASK_SINGLE_GROUNDING: TASK_SINGLE_GROUNDING,
    INTERNAL_SINGLE_VQA: TASK_SINGLE_VQA,
    TASK_SINGLE_VQA: TASK_SINGLE_VQA,
    INTERNAL_BITEMPORAL_CHANGE: TASK_BITEMPORAL_CHANGE,
    TASK_BITEMPORAL_CHANGE: TASK_BITEMPORAL_CHANGE,
    INTERNAL_CROSS_MODAL: TASK_CROSS_MODAL,
    TASK_CROSS_MODAL: TASK_CROSS_MODAL,
    INTERNAL_DOMAIN_KNOWLEDGE_QA: TASK_DOMAIN_KNOWLEDGE_QA,
    TASK_DOMAIN_KNOWLEDGE_QA: TASK_DOMAIN_KNOWLEDGE_QA,
}

ROUTER_SYSTEM_PROMPT = """You are the Semantic Intent Router for SatQuery AI, an autonomous Earth Observation and Satellite Imagery Intelligence system.
Your job is to analyze the user's natural language query and input image metadata, then output a structured JSON execution plan.

Allowed tasks:
- "single_vqa": Visual question answering, scene classification, land cover description on a single image.
- "visual_grounding": Detecting, locating, highlighting, delineating, or segmenting specific objects or features (rooftops, water, tanks, aircraft) on a single image.
- "bitemporal_change": Detecting changes, expansion, inundation, or differences between two timestamps (T1 baseline vs T2 post-event).
- "cross_modal": Joint analysis combining optical RGB texture with SAR (Synthetic Aperture Radar) backscatter physics.
- "domain_knowledge_qa": Earth observation theory or mission knowledge queries without images.

Allowed target features: ["water", "built-up", "vegetation", "infrastructure", "agriculture", "aircraft", "storage_tank", "unknown"]
Available tools: ["WaterGroundingTool", "TemporalChangeTool", "OpticalSARFusionTool", "GeodesicMeasurementTool", "RemoteSensingVLMClient"]

You MUST respond strictly with valid JSON conforming to this schema:
{
  "task": "single_vqa" | "visual_grounding" | "bitemporal_change" | "cross_modal" | "domain_knowledge_qa",
  "target_features": ["feature_name"],
  "tool_chain": ["tool_name_1", "tool_name_2"],
  "confidence": 0.95,
  "reasoning": "Brief justification of the classification"
}"""


@dataclass
class SemanticRoutingResult:
    task: str
    target_features: List[str] = field(default_factory=list)
    tool_chain: List[str] = field(default_factory=list)
    confidence: float = 0.90
    reasoning: str = ""
    internal_task: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "target_features": self.target_features,
            "tool_chain": self.tool_chain,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "internal_task": self.internal_task or TASK_TO_INTERNAL.get(self.task, INTERNAL_SINGLE_VQA),
        }


class SemanticIntentRouter:
    """Decoupled Semantic Intent Router using Ollama /api/chat with structured JSON output.
    
    Includes robust heuristic fallbacks and strictly enforces multimodal input constraints.
    """

    def __init__(self, ollama_url: Optional[str] = None, model: str = "llava", timeout: float = 3.0) -> None:
        self.ollama_url = (ollama_url or settings.resolved_ollama_url).rstrip("/")
        self.model = model
        self.timeout = timeout

    def route(
        self,
        query: str,
        filepaths: Optional[List[str]] = None,
        parsed_meta: Optional[List[Dict[str, Any]]] = None,
        force_task: Optional[str] = None,
    ) -> SemanticRoutingResult:
        """Route query and imagery metadata into an auditable semantic execution plan."""
        files = list(filepaths) if filepaths is not None else None
        num_images = len(files) if files is not None else 0
        q = (query or "").strip()

        # Handle manual override
        if force_task:
            forced_std = STANDARDIZED_TASKS.get(force_task.lower(), force_task.lower())
            internal = TASK_TO_INTERNAL.get(forced_std, INTERNAL_SINGLE_VQA)
            return self._build_plan(
                task=forced_std,
                internal=internal,
                query=q,
                reasoning=f"Task forced by request parameter: {force_task}",
                confidence=1.0,
            )

        # Handle query-only classification when filepaths is None
        if files is None:
            from app.agents.router import InputInspectorNode
            internal = InputInspectorNode.inspect(query=q, filepaths=None, parsed_meta=parsed_meta, force_task=force_task)
            std_task = STANDARDIZED_TASKS.get(internal, TASK_SINGLE_VQA)
            tool_map = {
                INTERNAL_BITEMPORAL_CHANGE: ["TemporalChangeTool", "GeodesicMeasurementTool"],
                INTERNAL_CROSS_MODAL: ["OpticalSARFusionTool", "GeodesicMeasurementTool"],
                INTERNAL_SINGLE_GROUNDING: ["WaterGroundingTool" if any(w in q.lower() for w in ["water", "reservoir", "lake", "flood"]) else "RemoteSensingVLMClient", "GeodesicMeasurementTool"],
                INTERNAL_SINGLE_VQA: ["RemoteSensingVLMClient"],
            }
            return self._build_plan(
                task=std_task,
                internal=internal,
                query=q,
                tool_chain=tool_map.get(internal, ["RemoteSensingVLMClient"]),
                reasoning="Query-level intent classification without attached imagery.",
                confidence=0.95,
            )

        # Enforce physical input constraints before LLM routing
        self._validate_input_constraints(q, files, parsed_meta)

        # 0 images: Domain knowledge QA
        if files is not None and num_images == 0:
            return self._build_plan(
                task=TASK_DOMAIN_KNOWLEDGE_QA,
                internal=INTERNAL_DOMAIN_KNOWLEDGE_QA,
                query=q,
                reasoning="No imagery provided; routed to Earth Observation domain knowledge QA.",
                confidence=0.98,
            )

        # Try LLM-based semantic routing via Ollama /api/chat
        llm_result = self._query_llm_router(q, num_images, parsed_meta)
        if llm_result:
            # Re-check constraints on LLM output to prevent hallucinated task-input mismatches
            valid_result = self._sanitize_llm_decision(llm_result, num_images, parsed_meta, q)
            if valid_result:
                return valid_result

        # Fallback to deterministic semantic heuristic
        return self._heuristic_route(q, num_images, parsed_meta)

    def _validate_input_constraints(
        self,
        query: str,
        files: Optional[List[str]],
        parsed_meta: Optional[List[Dict[str, Any]]],
    ) -> None:
        """Strictly reject intent-input mismatches matching SatQuery system guarantees."""
        if files is None:
            return

        q_lower = query.lower()
        num_images = len(files)

        is_temporal_intent = any(
            t in q_lower
            for t in [
                "between these two dates",
                "between two dates",
                "before and after",
                "what changed",
                "change detection",
                "land cover change",
                "newly flooded",
                "flood expansion",
                "increased, decreased",
                "increased or decreased",
                "remained unchanged",
            ]
        ) or ("between" in q_lower and ("date" in q_lower or "image" in q_lower or "t1" in q_lower))

        is_cross_modal_intent = (
            "cross-modal" in q_lower
            or "cross modal" in q_lower
            or ("optical" in q_lower and "sar" in q_lower)
            or ("optical" in q_lower and "radar" in q_lower)
            or "both sensors" in q_lower
        )

        if num_images == 0:
            if is_temporal_intent:
                logger.error("SemanticIntentRouter REJECTION: 0 images provided for temporal change query ('%s')", query)
                raise ValueError(
                    "Bi-temporal change detection requires two spatially aligned images (Before and After). Please attach an image pair to analyze temporal change."
                )
            if is_cross_modal_intent:
                logger.error("SemanticIntentRouter REJECTION: 0 images provided for cross-modal query ('%s')", query)
                raise ValueError(
                    "Cross-modal Optical+SAR joint analysis requires both Optical and SAR imagery, but 0 images were provided. Please upload both modalities to execute joint analysis."
                )

        if num_images == 1:
            if is_temporal_intent:
                logger.error("SemanticIntentRouter REJECTION: 1 image provided for temporal change query ('%s')", query)
                raise ValueError(
                    "Bi-temporal change detection requires two spatially aligned images (Before and After). Please attach an image pair to analyze temporal change."
                )
            if is_cross_modal_intent:
                logger.error("SemanticIntentRouter REJECTION: 1 image provided for cross-modal query ('%s')", query)
                raise ValueError(
                    "Cross-modal Optical+SAR joint analysis requires both Optical and SAR imagery, "
                    "but only 1 image was provided. Please upload both modalities to execute joint analysis."
                )

    def _query_llm_router(
        self,
        query: str,
        num_images: int,
        parsed_meta: Optional[List[Dict[str, Any]]],
    ) -> Optional[SemanticRoutingResult]:
        """Queries Ollama /api/chat with format='json' for structured intent prediction."""
        meta_summary = []
        for idx, m in enumerate(parsed_meta or []):
            meta_summary.append({
                "image_index": idx + 1,
                "modalities": m.get("modalities", ["RGB"]),
                "sensor": m.get("sensor", "Unknown"),
            })

        user_content = json.dumps({
            "query": query,
            "image_count": num_images,
            "images_metadata": meta_summary,
        })

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "format": "json",
            "stream": False,
            "options": {"temperature": 0.0},
        }

        candidates = [self.ollama_url]
        if os.path.exists("/.dockerenv"):
            candidates.extend(["http://ollama:11434", "http://satquery_ollama:11434"])
        else:
            candidates.extend(["http://127.0.0.1:11434", "http://localhost:11434"])
        unique_urls = list(dict.fromkeys([u for u in candidates if u]))

        for base_url in unique_urls:
            url = f"{base_url.rstrip('/')}/api/chat"
            try:
                from app.services.models.base import _http_post_json

                resp_data = _http_post_json(url, payload, connect_timeout=0.3, read_timeout=self.timeout)
                content = (
                    resp_data.get("message", {}).get("content")
                    or resp_data.get("response")
                    or ""
                )
                if content:
                    parsed = json.loads(content)
                    task = parsed.get("task", "")
                    features = parsed.get("target_features", [])
                    tools = parsed.get("tool_chain", [])
                    conf = float(parsed.get("confidence", 0.90))
                    reasoning = parsed.get("reasoning", "LLM-inferred plan")

                    std_task = STANDARDIZED_TASKS.get(task, task)
                    internal = TASK_TO_INTERNAL.get(std_task, INTERNAL_SINGLE_VQA)
                    return SemanticRoutingResult(
                        task=std_task,
                        target_features=features,
                        tool_chain=tools,
                        confidence=conf,
                        reasoning=reasoning,
                        internal_task=internal,
                    )
            except Exception as err:
                logger.debug("Ollama semantic chat router failed on %s: %s", url, err)
                continue

        return None

    def _sanitize_llm_decision(
        self,
        result: SemanticRoutingResult,
        num_images: int,
        parsed_meta: Optional[List[Dict[str, Any]]],
        query: str,
    ) -> Optional[SemanticRoutingResult]:
        """Ensures LLM output adheres to system image count and modality realities."""
        task = result.task
        meta_list = list(parsed_meta or [])
        has_sar = any("SAR-C-Band" in m.get("modalities", []) for m in meta_list)
        has_optical = any(
            any(b in m.get("modalities", []) for b in ["RGB", "Red", "Green", "Blue", "NIR"])
            for m in meta_list
        )

        if num_images >= 2:
            if has_sar and has_optical:
                result.task = TASK_CROSS_MODAL
                result.internal_task = INTERNAL_CROSS_MODAL
                if "OpticalSARFusionTool" not in result.tool_chain:
                    result.tool_chain.append("OpticalSARFusionTool")
            elif task not in [TASK_BITEMPORAL_CHANGE, TASK_CROSS_MODAL]:
                result.task = TASK_BITEMPORAL_CHANGE
                result.internal_task = INTERNAL_BITEMPORAL_CHANGE
                if "TemporalChangeTool" not in result.tool_chain:
                    result.tool_chain.append("TemporalChangeTool")
            return result

        if num_images == 1:
            if task in [TASK_BITEMPORAL_CHANGE, TASK_CROSS_MODAL]:
                return None  # reject mismatch and fall back to heuristic
            if task in [TASK_VISUAL_GROUNDING, TASK_SINGLE_GROUNDING]:
                result.task = TASK_SINGLE_GROUNDING
                result.internal_task = INTERNAL_SINGLE_GROUNDING
            else:
                result.task = TASK_SINGLE_VQA
                result.internal_task = INTERNAL_SINGLE_VQA
            return result

        return result

    def _heuristic_route(
        self,
        query: str,
        num_images: int,
        parsed_meta: Optional[List[Dict[str, Any]]],
    ) -> SemanticRoutingResult:
        """Resilient semantic heuristic fallback mimicking domain specialist rules."""
        q_lower = query.lower()
        meta_list = list(parsed_meta or [])
        has_sar = any("SAR-C-Band" in m.get("modalities", []) for m in meta_list)
        has_optical = any(
            any(b in m.get("modalities", []) for b in ["RGB", "Red", "Green", "Blue", "NIR"])
            for m in meta_list
        )

        target_features: List[str] = []
        for feat in ["water", "built-up", "vegetation", "infrastructure", "agriculture", "tank", "aircraft", "rooftop"]:
            if feat in q_lower or (feat == "built-up" and "urban" in q_lower):
                target_features.append(feat)
        if not target_features:
            target_features = ["infrastructure"]

        if num_images >= 2:
            if (has_sar and has_optical) or "cross-modal" in q_lower or "sar" in q_lower:
                return self._build_plan(
                    task=TASK_CROSS_MODAL,
                    internal=INTERNAL_CROSS_MODAL,
                    query=query,
                    target_features=target_features,
                    tool_chain=["OpticalSARFusionTool", "GeodesicMeasurementTool", "RemoteSensingVLMClient"],
                    confidence=0.94,
                    reasoning="2 multi-sensor images (Optical + SAR) detected; routing to cross-modal fusion pipeline.",
                )
            return self._build_plan(
                task=TASK_BITEMPORAL_CHANGE,
                internal=INTERNAL_BITEMPORAL_CHANGE,
                query=query,
                target_features=target_features,
                tool_chain=["TemporalChangeTool", "GeodesicMeasurementTool", "RemoteSensingVLMClient"],
                confidence=0.91,
                reasoning="2 multi-temporal scenes detected; routing to bi-temporal change detection pipeline.",
            )

        # 1 Image
        is_water = "water" in q_lower or "flood" in q_lower or "lake" in q_lower or "river" in q_lower
        is_grounding = any(
            w in q_lower
            for w in ["ground", "highlight", "detect", "locate", "outline", "segment", "delineate", "find", "where is"]
        )

        if is_grounding or (is_water and any(w in q_lower for w in ["mask", "boundary", "extent", "show"])):
            tools = ["WaterGroundingTool" if is_water else "RemoteSensingVLMClient", "GeodesicMeasurementTool"]
            return self._build_plan(
                task=TASK_SINGLE_GROUNDING,
                internal=INTERNAL_SINGLE_GROUNDING,
                query=query,
                target_features=target_features,
                tool_chain=tools,
                confidence=0.89,
                reasoning="Feature localization and vector bounding requested on single image.",
            )

        return self._build_plan(
            task=TASK_SINGLE_VQA,
            internal=INTERNAL_SINGLE_VQA,
            query=query,
            target_features=target_features,
            tool_chain=["RemoteSensingVLMClient"],
            confidence=0.92,
            reasoning="Earth Observation visual question answering and land cover interpretation on single image.",
        )

    def _build_plan(
        self,
        task: str,
        internal: str,
        query: str,
        reasoning: str,
        confidence: float,
        target_features: Optional[List[str]] = None,
        tool_chain: Optional[List[str]] = None,
    ) -> SemanticRoutingResult:
        std_task = STANDARDIZED_TASKS.get(task, task)
        features = target_features or ["infrastructure"]
        if not tool_chain:
            if std_task == TASK_BITEMPORAL_CHANGE:
                tool_chain = ["TemporalChangeTool", "GeodesicMeasurementTool"]
            elif std_task == TASK_CROSS_MODAL:
                tool_chain = ["OpticalSARFusionTool", "GeodesicMeasurementTool"]
            elif std_task in [TASK_SINGLE_GROUNDING, TASK_VISUAL_GROUNDING]:
                tool_chain = ["WaterGroundingTool" if "water" in features else "RemoteSensingVLMClient", "GeodesicMeasurementTool"]
            elif std_task == TASK_DOMAIN_KNOWLEDGE_QA:
                tool_chain = ["RemoteSensingVLMClient"]
            else:
                tool_chain = ["RemoteSensingVLMClient"]

        return SemanticRoutingResult(
            task=std_task,
            target_features=features,
            tool_chain=tool_chain,
            confidence=confidence,
            reasoning=reasoning,
            internal_task=internal or TASK_TO_INTERNAL.get(std_task, INTERNAL_SINGLE_VQA),
        )
