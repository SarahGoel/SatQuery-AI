"""Stateful SatQueryController — parse, align, classify, execute.

Geospatial I/O stays in this layer and geospatial/*; GPU modules live
under services/models. The controller records every registry hop onto
the auditable trace.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import rasterio
from shapely.geometry import box, mapping
from shapely.geometry.base import BaseGeometry
from sqlalchemy.orm import Session

from app.core.config import settings
from app.database.models import AuditableExecutionTrace, TraceModelExecution
from app.schemas.trace import (
    AuditableTraceLogSchema,
    InputMetadataSchema,
    RegistryExecutionSchema,
)
from app.schemas.validation import TaskType
from app.services.geospatial.alignment import SpatialAligner
from app.services.geospatial.spectral import SpectralExtractor
from app.services.geospatial.vector import raster_mask_to_geojson
from app.services.models.base import LocalVisionLanguageClient
from app.services.models.change_vqa import TemporalChangeVQA
from app.services.models.fusion import OpticalSarFusion
from app.services.models.grounding import TextGuidedGrounder
from app.utils.logger import get_logger

try:
    from langgraph.graph import END, StateGraph
except ImportError:  # langgraph 0.1.x import path
    try:
        from langgraph.graph import END
        from langgraph.graph.graph import StateGraph
    except ImportError:
        END = None
        StateGraph = None

logger = get_logger(__name__)

_CHANGE_HINTS = (
    "change",
    "changed",
    "difference",
    "before after",
    "t1",
    "t2",
    "temporal",
    "bi-temporal",
    "bitemporal",
)
_GROUNDING_HINTS = (
    "highlight",
    "segment",
    "locate",
    "where is",
    "show me",
    "mask",
    "delineate",
    "footprint",
)
_FUSION_HINTS = ("sar", "risat", "radar", "cross-modal", "cross modal", "joint")


class SatQueryController:
    """Orchestrates Feature 2.1–2.5 workflows and persists traces."""

    def __init__(self, db: Session | None = None) -> None:
        self.db = db
        self.aligner = SpatialAligner()
        self.spectral = SpectralExtractor()
        self.vlm = LocalVisionLanguageClient()
        self.fusion = OpticalSarFusion()
        self.grounder = TextGuidedGrounder()
        self.change_vqa = TemporalChangeVQA()
        self.last_geojson: dict[str, Any] | None = None
        self.last_overlay_uri: str | None = None

    def parse_geotiff_metadata(
        self,
        path: Path,
        modalities: list[str],
    ) -> InputMetadataSchema:
        """Read CRS, bounds, and affine coefficients via Rasterio."""
        with rasterio.open(path) as src:
            if src.crs is None:
                raise ValueError(f"{path.name} has no CRS; cannot ingest unreferenced raster")
            crs = src.crs.to_string()
            bounds = [float(src.bounds.left), float(src.bounds.bottom), float(src.bounds.right), float(src.bounds.top)]
            transform = list(src.transform)[:6]
        return InputMetadataSchema(
            crs=crs,
            bounds=bounds,
            affine_transform=transform,
            modalities=modalities,
        )

    def validate_spatial_alignment(
        self,
        primary: InputMetadataSchema,
        others: list[InputMetadataSchema],
    ) -> None:
        """Require overlapping footprints (Shapely) before multi-raster fusion."""
        primary_geom = self._bounds_polygon(primary)
        for other in others:
            other_geom = self._bounds_polygon(other)
            if not primary_geom.intersects(other_geom):
                raise ValueError(
                    "Rasters do not intersect; refuse to fuse misaligned scenes "
                    f"(primary={primary.bounds}, other={other.bounds})"
                )
            inter = primary_geom.intersection(other_geom)
            union = primary_geom.union(other_geom)
            iou = float(inter.area / union.area) if union.area else 0.0
            if iou < settings.ALIGNMENT_IOU_THRESHOLD:
                raise ValueError(
                    f"Spatial IoU {iou:.3f} below threshold {settings.ALIGNMENT_IOU_THRESHOLD}"
                )

    def classify_query(
        self,
        query: str,
        has_t2: bool,
        has_sar: bool,
        force_task: str | None = None,
    ) -> TaskType:
        """Route NL queries onto one of four operational workflows."""
        if force_task:
            return TaskType(force_task)
        q = query.lower()
        if has_t2 or any(h in q for h in _CHANGE_HINTS):
            return TaskType.BI_TEMPORAL_CHANGE_ANALYSIS
        if has_sar or any(h in q for h in _FUSION_HINTS):
            return TaskType.CROSS_MODAL_JOINT_ANALYSIS
        if any(h in q for h in _GROUNDING_HINTS):
            return TaskType.SINGLE_IMAGE_GROUNDING
        return TaskType.SINGLE_IMAGE_VQA

    def execute_workflow(
        self,
        query: str,
        optical_path: Path,
        optical_t2_path: Path | None = None,
        sar_path: Path | None = None,
        force_task: str | None = None,
        use_mobilesam: bool = True,
    ) -> AuditableTraceLogSchema:
        """End-to-end routing: metadata → alignment → spectral → DL → trace."""
        self.last_geojson = None
        self.last_overlay_uri = None
        trace_id = uuid.uuid4().hex
        executions: list[RegistryExecutionSchema] = []

        optical_meta = self.parse_geotiff_metadata(optical_path, ["optical", "cartosat-2s"])
        extras: list[InputMetadataSchema] = []
        if optical_t2_path is not None:
            extras.append(self.parse_geotiff_metadata(optical_t2_path, ["optical", "t2"]))
        if sar_path is not None:
            extras.append(self.parse_geotiff_metadata(sar_path, ["sar", "risat"]))
        if extras:
            self.validate_spatial_alignment(optical_meta, extras)

        task = self.classify_query(
            query,
            has_t2=optical_t2_path is not None,
            has_sar=sar_path is not None,
            force_task=force_task,
        )
        if task == TaskType.BI_TEMPORAL_CHANGE_ANALYSIS and optical_t2_path is None:
            raise ValueError("bi_temporal_change_analysis requires optical_t2 GeoTIFF")
        if task == TaskType.CROSS_MODAL_JOINT_ANALYSIS and sar_path is None:
            raise ValueError("cross_modal_joint_analysis requires a SAR GeoTIFF")

        aligned_optical = optical_path
        aligned_t2 = optical_t2_path
        aligned_sar = sar_path
        if optical_t2_path is not None:
            t2_pack = self.aligner.align_pair(optical_path, optical_t2_path)
            aligned_optical = t2_pack.reference_path
            aligned_t2 = t2_pack.moving_path
            executions.append(
                RegistryExecutionSchema(
                    model="spatial-aligner",
                    params={
                        "inliers": t2_pack.inliers,
                        "crs": t2_pack.crs,
                        "pair": "optical_t2",
                    },
                )
            )
        if sar_path is not None:
            sar_pack = self.aligner.align_pair(optical_path, sar_path)
            aligned_optical = sar_pack.reference_path
            aligned_sar = sar_pack.moving_path
            executions.append(
                RegistryExecutionSchema(
                    model="spatial-aligner",
                    params={
                        "inliers": sar_pack.inliers,
                        "crs": sar_pack.crs,
                        "pair": "optical_sar",
                    },
                )
            )

        spectral_pack = self.spectral.extract(aligned_optical)
        executions.append(
            RegistryExecutionSchema(
                model="spectral-extractor",
                params={
                    "ndvi_mean": spectral_pack.ndvi_mean,
                    "ndwi_mean": spectral_pack.ndwi_mean,
                    "tensor_shape": list(spectral_pack.feature_tensor.shape),
                },
            )
        )

        output_text = ""
        confidence = 0.5
        mask = None

        if task == TaskType.SINGLE_IMAGE_VQA:
            vlm = self.vlm.generate(prompt=query, image_path=aligned_optical)
            output_text = vlm.text
            confidence = vlm.confidence
            executions.append(
                RegistryExecutionSchema(model=settings.VLM_MODEL_NAME, params=vlm.params)
            )
        elif task == TaskType.SINGLE_IMAGE_GROUNDING:
            grounded = self.grounder.ground(
                image_path=aligned_optical,
                prompt=query,
                use_mobilesam=use_mobilesam,
            )
            mask = grounded.mask
            output_text = grounded.description
            confidence = grounded.confidence
            executions.append(
                RegistryExecutionSchema(
                    model="mobilesam" if use_mobilesam else "sam-vit-b",
                    params=grounded.params,
                )
            )
        elif task == TaskType.CROSS_MODAL_JOINT_ANALYSIS:
            fused = self.fusion.fuse(optical_path=aligned_optical, sar_path=aligned_sar)  # type: ignore[arg-type]
            vlm = self.vlm.generate(
                prompt=query,
                image_path=aligned_optical,
                extra_context={"fusion_energy": fused.attention_energy},
            )
            output_text = vlm.text
            confidence = min(vlm.confidence, fused.confidence)
            mask = fused.salient_mask
            executions.append(
                RegistryExecutionSchema(model="optical-sar-fusion", params=fused.params)
            )
            executions.append(
                RegistryExecutionSchema(model=settings.VLM_MODEL_NAME, params=vlm.params)
            )
        elif task == TaskType.BI_TEMPORAL_CHANGE_ANALYSIS:
            changed = self.change_vqa.analyze(
                t1_path=aligned_optical,
                t2_path=aligned_t2,  # type: ignore[arg-type]
                query=query,
            )
            output_text = changed.answer
            confidence = changed.confidence
            mask = changed.change_mask
            self.last_overlay_uri = changed.overlay_uri
            executions.append(
                RegistryExecutionSchema(model="change-vqa", params=changed.params)
            )

        if mask is not None:
            self.last_geojson = raster_mask_to_geojson(aligned_optical, mask)

        modalities = list(optical_meta.modalities)
        if optical_t2_path is not None:
            modalities.append("optical-t2")
        if sar_path is not None:
            modalities.append("sar")
        input_metadata = InputMetadataSchema(
            crs=optical_meta.crs,
            bounds=optical_meta.bounds,
            affine_transform=optical_meta.affine_transform,
            modalities=modalities,
        )

        trace = AuditableTraceLogSchema(
            trace_id=trace_id,
            task=task.value,
            query=query,
            input_metadata=input_metadata,
            registry_execution=executions,
            confidence_score=float(max(0.0, min(1.0, confidence))),
            output=output_text,
        )
        self._persist_trace(trace, input_metadata)
        logger.info("workflow_complete", extra={"trace_id": trace_id, "task": task.value})
        return trace

    def _bounds_polygon(self, meta: InputMetadataSchema) -> BaseGeometry:
        minx, miny, maxx, maxy = meta.bounds
        return box(minx, miny, maxx, maxy)

    def _persist_trace(self, trace: AuditableTraceLogSchema, meta: InputMetadataSchema) -> None:
        if self.db is None:
            return
        geom = mapping(self._bounds_polygon(meta))
        # GeoAlchemy2 accepts WKT; keep EPSG:4326 contract from SQL schema.
        from geoalchemy2.shape import from_shape

        record = AuditableExecutionTrace(
            trace_id=trace.trace_id,
            task_type=trace.task,
            user_query=trace.query,
            crs=(meta.crs or "EPSG:4326")[:32],
            affine_transform_matrix=list(meta.affine_transform),
            bounding_box_geometry=from_shape(self._bounds_polygon(meta), srid=4326),
            overall_confidence=trace.confidence_score,
            final_output=trace.output,
        )
        self.db.add(record)
        for order, step in enumerate(trace.registry_execution, start=1):
            self.db.add(
                TraceModelExecution(
                    trace_id=trace.trace_id,
                    model_name=step.model if self._model_known(step.model) else "llava-3b",
                    parameter_configuration=step.params,
                    execution_order=order,
                )
            )
        try:
            self.db.commit()
        except Exception:
            self.db.rollback()
            logger.exception("trace_persist_failed", extra={"trace_id": trace.trace_id, "geom": geom})
            raise

    def _model_known(self, name: str) -> bool:
        known = {
            "llava-3b",
            "sam-vit-b",
            "mobilesam",
            "optical-sar-fusion",
            "change-vqa",
            "bigearthnet-encoder",
            "spatial-aligner",
            "spectral-extractor",
        }
        return name in known


def compile_satquery_graph(controller: SatQueryController):
    """LangGraph wrapper around classify → execute for team extensions."""
    if StateGraph is None:
        return None

    def classify_node(state: dict[str, Any]) -> dict[str, Any]:
        task = controller.classify_query(
            state["query"],
            has_t2=state.get("optical_t2_path") is not None,
            has_sar=state.get("sar_path") is not None,
            force_task=state.get("force_task"),
        )
        return {**state, "task": task.value}

    def execute_node(state: dict[str, Any]) -> dict[str, Any]:
        trace = controller.execute_workflow(
            query=state["query"],
            optical_path=state["optical_path"],
            optical_t2_path=state.get("optical_t2_path"),
            sar_path=state.get("sar_path"),
            force_task=state.get("task"),
            use_mobilesam=state.get("use_mobilesam", True),
        )
        return {**state, "trace": trace.model_dump()}

    graph = StateGraph(dict)
    graph.add_node("classify", classify_node)
    graph.add_node("execute", execute_node)
    graph.set_entry_point("classify")
    graph.add_edge("classify", "execute")
    graph.add_edge("execute", END)
    return graph.compile()

