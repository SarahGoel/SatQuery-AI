"""Export API Endpoints — GeoJSON and KML Vector Downloads.

Provides:
- GET /api/v1/reports/{trace_id}/export: Vector export in GeoJSON or KML format.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from sqlalchemy.orm import Session

try:
    import simplekml
except ImportError:  # pragma: no cover
    simplekml = None

from app.database.session import get_optional_db
from app.utils.report_generator import extract_execution_trace
from app.utils.trace_store import recall_trace

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/reports", tags=["export"])


def _extract_feature_collection(trace: dict[str, Any], trace_id: str) -> dict[str, Any]:
    """Extracts or normalizes a GeoJSON FeatureCollection from trace payload."""
    geojson = trace.get("geojson")
    if not geojson and "audit_summary" in trace and isinstance(trace["audit_summary"], dict):
        geojson = trace["audit_summary"].get("geojson")
    if not geojson and "analysis_data" in trace and isinstance(trace["analysis_data"], dict):
        geojson = trace["analysis_data"].get("geojson")

    if not geojson:
        # Fallback to scene bounding box if present in input_metadata
        meta = trace.get("input_metadata") or {}
        bounds = meta.get("bounds")
        if bounds and len(bounds) == 4:
            west, south, east, north = bounds
            poly_coords = [[[west, south], [east, south], [east, north], [west, north], [west, south]]]
            return {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {
                            "trace_id": trace_id,
                            "label": "Scene Extent Bounding Box",
                            "source": "SatQuery Spatial Bounding Extent",
                        },
                        "geometry": {"type": "Polygon", "coordinates": poly_coords},
                    }
                ],
            }
        return {"type": "FeatureCollection", "features": []}

    if isinstance(geojson, dict):
        if geojson.get("type") == "FeatureCollection":
            return geojson
        if geojson.get("type") == "Feature":
            return {"type": "FeatureCollection", "features": [geojson]}
        if "features" in geojson:
            return {"type": "FeatureCollection", "features": geojson["features"]}
        if "type" in geojson and "coordinates" in geojson:
            return {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"trace_id": trace_id, "label": "Detected Grounding Feature"},
                        "geometry": geojson,
                    }
                ],
            }

    return {"type": "FeatureCollection", "features": []}


def _build_kml_from_features(fc: dict[str, Any], trace_id: str) -> str:
    """Converts a GeoJSON FeatureCollection to styled KML XML."""
    if simplekml is None:
        raise HTTPException(
            status_code=503,
            detail="simplekml library is not installed for KML export.",
        )

    kml = simplekml.Kml(name=f"SatQuery AI Geospatial Export — {trace_id}")
    folder = kml.newfolder(name=f"Delineated Features ({trace_id})")

    features = fc.get("features", [])
    for idx, feat in enumerate(features):
        if not isinstance(feat, dict):
            continue

        props = feat.get("properties") or {}
        geom = feat.get("geometry") or {}
        geom_type = geom.get("type", "")
        coords = geom.get("coordinates")

        name = str(
            props.get("name")
            or props.get("label")
            or props.get("class")
            or props.get("feature_type")
            or f"Delineated Feature {idx + 1}"
        )

        desc_items = []
        for k, v in props.items():
            if k not in ["coordinates", "geometry"] and v is not None:
                desc_items.append(f"<b>{k}</b>: {v}")
        desc = "<br/>".join(desc_items) if desc_items else f"Grounding polygon from trace {trace_id}"

        # Determine styling based on class / task name
        tag = (name + " " + str(props.get("category", ""))).lower()
        if any(w in tag for w in ["water", "lake", "river", "flood", "inundat"]):
            fill_color = simplekml.Color.changealphaint(150, simplekml.Color.cyan)
            line_color = simplekml.Color.blue
        elif any(w in tag for w in ["forest", "vegetat", "crop", "tree", "green"]):
            fill_color = simplekml.Color.changealphaint(150, simplekml.Color.lime)
            line_color = simplekml.Color.darkgreen
        elif any(w in tag for w in ["urban", "built", "struct", "building", "road"]):
            fill_color = simplekml.Color.changealphaint(150, simplekml.Color.orange)
            line_color = simplekml.Color.red
        else:
            fill_color = simplekml.Color.changealphaint(130, simplekml.Color.yellow)
            line_color = simplekml.Color.gold

        if geom_type == "Polygon" and coords:
            # coords: [exterior, interior1, ...]
            outer = [(float(pt[0]), float(pt[1])) for pt in coords[0]]
            poly = folder.newpolygon(name=name, description=desc, outerboundaryis=outer)
            if len(coords) > 1:
                poly.innerboundaryis = [
                    [(float(pt[0]), float(pt[1])) for pt in ring] for ring in coords[1:]
                ]
            poly.style.polystyle.color = fill_color
            poly.style.linestyle.color = line_color
            poly.style.linestyle.width = 2.5

        elif geom_type == "MultiPolygon" and coords:
            for p_idx, poly_part in enumerate(coords):
                outer = [(float(pt[0]), float(pt[1])) for pt in poly_part[0]]
                poly = folder.newpolygon(
                    name=f"{name} (Part {p_idx + 1})",
                    description=desc,
                    outerboundaryis=outer,
                )
                if len(poly_part) > 1:
                    poly.innerboundaryis = [
                        [(float(pt[0]), float(pt[1])) for pt in ring] for ring in poly_part[1:]
                    ]
                poly.style.polystyle.color = fill_color
                poly.style.linestyle.color = line_color
                poly.style.linestyle.width = 2.5

        elif geom_type == "Point" and coords:
            pnt = folder.newpoint(name=name, description=desc, coords=[(float(coords[0]), float(coords[1]))])
            pnt.style.iconstyle.color = line_color

        elif geom_type == "LineString" and coords:
            ls = folder.newlinestring(name=name, description=desc, coords=[(float(pt[0]), float(pt[1])) for pt in coords])
            ls.style.linestyle.color = line_color
            ls.style.linestyle.width = 3.0

    return kml.kml()


@router.get("/{trace_id}/export")
async def export_vector_report(
    trace_id: str,
    format: str = Query(default="geojson", description="Export format: 'geojson' or 'kml'"),
    db: Session = Depends(get_optional_db),
):
    """Download vector delineation results as standard GeoJSON or Google Earth KML."""
    fmt = format.lower().strip()
    if fmt not in ["geojson", "kml"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported vector format '{format}'. Supported options: 'geojson', 'kml'.",
        )

    # 1. Retrieve the execution trace
    trace = None
    try:
        trace = extract_execution_trace(db=db, trace_id=trace_id)
    except KeyError:
        pass

    if not trace:
        cached = recall_trace(trace_id)
        if cached:
            trace = cached

    if not trace and db is not None:
        try:
            from app.database.models import QueryHistory

            record = db.query(QueryHistory).filter(QueryHistory.id == trace_id).first()
            if record:
                trace = record.analysis_data or {}
                trace["trace_id"] = trace_id
        except Exception as db_err:
            logger.debug("Failed QueryHistory query for export: %s", db_err)

    if not trace:
        raise HTTPException(status_code=404, detail=f"Execution trace not found for trace_id '{trace_id}'.")

    # 2. Extract FeatureCollection
    fc = _extract_feature_collection(trace, trace_id)

    # 3. Format and stream
    if fmt == "geojson":
        geojson_str = json.dumps(fc, indent=2, default=str)
        return Response(
            content=geojson_str,
            media_type="application/geo+json",
            headers={"Content-Disposition": f'attachment; filename="{trace_id}.geojson"'},
        )
    else:
        kml_str = _build_kml_from_features(fc, trace_id)
        return Response(
            content=kml_str,
            media_type="application/vnd.google-earth.kml+xml",
            headers={"Content-Disposition": f'attachment; filename="{trace_id}.kml"'},
        )
