"""SatQuery AI — Master Geospatial API Router.

Provides:
1. POST /api/v1/geospatial/composite: Multi-band/SAR visualization (RGB, CIR, SAR dB, NDVI, NDWI).
2. POST /api/v1/geospatial/inspect-point: Coordinate-to-pixel sampling with spectral indices & terrain classification.
3. GET /api/v1/reports/{trace_id}/export: Vector report downloads (GeoJSON and styled KML).
"""

from __future__ import annotations

import base64
import io
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, Response, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

try:
    import rasterio
    import rasterio.windows
    from rasterio.warp import transform
except ImportError:  # pragma: no cover
    rasterio = None

try:
    import simplekml
except ImportError:  # pragma: no cover
    simplekml = None

from app.core.config import settings
from app.database.session import get_optional_db
from app.services.geospatial.spectral import BandMapping, SpectralIndicesEngine, detect_sensor_profile
from app.utils.report_generator import extract_execution_trace
from app.utils.trace_store import recall_trace

logger = logging.getLogger(__name__)

router = APIRouter(tags=["geospatial"])


# ============================================================================
# Schemas & Helper Functions
# ============================================================================

class InspectPointRequest(BaseModel):
    """Payload for map tooltip pixel coordinate inspection."""

    trace_id: str = Field(..., description="Execution trace identifier or raster path")
    latitude: Optional[float] = Field(None, description="WGS84 Latitude")
    longitude: Optional[float] = Field(None, description="WGS84 Longitude")
    lat: Optional[float] = Field(None, description="Alias for latitude")
    lon: Optional[float] = Field(None, description="Alias for longitude")
    lng: Optional[float] = Field(None, description="Alias for longitude")

    def get_coordinates(self) -> tuple[float, float]:
        lat = self.latitude if self.latitude is not None else self.lat
        lon = self.longitude if self.longitude is not None else (self.lon if self.lon is not None else self.lng)
        if lat is None or lon is None:
            raise ValueError("Both latitude (or lat) and longitude (or lon/lng) are required.")
        return float(lat), float(lon)


def _resolve_raster_path_from_trace(trace_id: str, db: Optional[Session] = None) -> Optional[str]:
    """Locates the raster file on disk for a given trace_id."""
    if not trace_id:
        return None

    # 1. Direct path check
    direct_path = Path(trace_id)
    if direct_path.exists() and direct_path.is_file():
        return str(direct_path)

    # 2. Check in-memory trace store
    cached = recall_trace(trace_id)
    if cached and isinstance(cached, dict):
        fps = cached.get("filepaths") or cached.get("files") or cached.get("file_paths")
        if fps:
            if isinstance(fps, (list, tuple)) and len(fps) > 0:
                for candidate in fps:
                    if Path(candidate).exists():
                        return str(candidate)
            elif isinstance(fps, str) and Path(fps).exists():
                return str(fps)

        # Check input metadata
        meta = cached.get("input_metadata") or {}
        fp = meta.get("filepath")
        if fp and Path(fp).exists():
            return str(fp)

    # 3. Check PostGIS QueryHistory
    if db is not None:
        try:
            from app.database.models import QueryHistory

            record = db.query(QueryHistory).filter(QueryHistory.id == trace_id).first()
            if record and record.analysis_data:
                fps = record.analysis_data.get("filepaths")
                if fps and isinstance(fps, (list, tuple)):
                    for candidate in fps:
                        if Path(candidate).exists():
                            return str(candidate)
        except Exception as db_err:
            logger.debug("Failed to query raster from QueryHistory: %s", db_err)

    # 4. Search upload directory for matching files
    upload_dir = Path(settings.UPLOAD_DIR)
    if upload_dir.exists():
        for p in upload_dir.glob(f"**/*{trace_id}*"):
            if p.is_file() and p.suffix.lower() in [".tif", ".tiff", ".gtiff", ".png", ".jpg"]:
                return str(p)

    return None


def _stretch_channel(ch: np.ndarray) -> np.ndarray:
    """Stretches a 2D float or integer channel to uint8 [0, 255] via 2nd-98th percentiles."""
    valid = ch[np.isfinite(ch)]
    if valid.size == 0 or np.all(valid == valid[0]):
        return np.zeros_like(ch, dtype=np.uint8)
    p2, p98 = np.percentile(valid, (2, 98))
    if p98 <= p2:
        p2, p98 = float(np.min(valid)), float(np.max(valid))
    if p98 <= p2:
        return np.zeros_like(ch, dtype=np.uint8)
    clipped = np.clip(ch, p2, p98)
    scaled = ((clipped - p2) / (p98 - p2 + 1e-7)) * 255.0
    return np.clip(scaled, 0, 255).astype(np.uint8)


def _render_composite_buffer(
    arr: np.ndarray,
    band_map: BandMapping,
    composite_type: str,
    engine: SpectralIndicesEngine,
) -> io.BytesIO:
    """Renders array according to composite_type into an 8-bit PNG io.BytesIO buffer."""
    c, h, w = arr.shape
    comp = composite_type.lower().strip()

    if comp == "cir":
        cir_arr = engine.generate_cir_composite(arr, band_map=band_map)
        img = Image.fromarray(cir_arr, mode="RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return buf

    elif comp in ["sar", "sar_db"]:
        sar_ch = arr[0]
        sar_db_arr = engine.generate_sar_db(sar_ch)
        img = Image.fromarray(sar_db_arr, mode="L")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return buf

    elif comp == "ndvi":
        r_idx = band_map.red if (band_map.red is not None and band_map.red < c) else min(c - 1, 2)
        has_real_nir = band_map.nir is not None and band_map.nir < c
        if has_real_nir:
            ndvi = engine.compute_ndvi(arr[r_idx], nir=arr[band_map.nir])
        elif c >= 4:
            ndvi = engine.compute_ndvi(arr[r_idx], nir=arr[min(c - 1, 3)])
        else:
            g_idx = band_map.green if (band_map.green is not None and band_map.green < c) else min(c - 1, 1)
            ndvi = engine.compute_ndvi(arr[r_idx], nir=None, green=arr[g_idx])
        return engine.render_colormap(ndvi, colormap="RdYlGn")

    elif comp == "ndwi":
        g_idx = band_map.green if (band_map.green is not None and band_map.green < c) else min(c - 1, 1)
        has_real_nir = band_map.nir is not None and band_map.nir < c
        if has_real_nir:
            ndwi = engine.compute_ndwi(arr[g_idx], nir=arr[band_map.nir])
        elif c >= 4:
            ndwi = engine.compute_ndwi(arr[g_idx], nir=arr[min(c - 1, 3)])
        else:
            b_idx = band_map.blue if (band_map.blue is not None and band_map.blue < c) else 0
            r_idx = band_map.red if (band_map.red is not None and band_map.red < c) else min(c - 1, 2)
            ndwi = engine.compute_ndwi(arr[g_idx], nir=None, blue=arr[b_idx], red=arr[r_idx])
        return engine.render_colormap(ndwi, colormap="Blues")

    elif comp == "mndwi":
        g_idx = band_map.green if (band_map.green is not None and band_map.green < c) else min(c - 1, 1)
        swir_idx = band_map.swir if (band_map.swir is not None and band_map.swir < c) else None
        nir_idx = band_map.nir if (band_map.nir is not None and band_map.nir < c) else min(c - 1, 3 if c >= 4 else 0)
        swir_ch = arr[swir_idx] if swir_idx is not None else None
        mndwi = engine.compute_mndwi(arr[g_idx], swir=swir_ch, nir=arr[nir_idx])
        return engine.render_colormap(mndwi, colormap="Blues")

    elif comp == "ndbi":
        swir_idx = band_map.swir if (band_map.swir is not None and band_map.swir < c) else None
        nir_idx = band_map.nir if (band_map.nir is not None and band_map.nir < c) else min(c - 1, 3 if c >= 4 else 0)
        swir_ch = arr[swir_idx] if swir_idx is not None else None
        ndbi = engine.compute_ndbi(swir=swir_ch, nir=arr[nir_idx])
        return engine.render_colormap(ndbi, colormap="YlOrRd")

    else:
        # Default True Color / RGB
        r_idx = band_map.red if (band_map.red is not None and band_map.red < c) else min(c - 1, 0)
        g_idx = band_map.green if (band_map.green is not None and band_map.green < c) else min(c - 1, 1 if c > 1 else 0)
        b_idx = band_map.blue if (band_map.blue is not None and band_map.blue < c) else min(c - 1, 2 if c > 2 else 0)

        r_norm = _stretch_channel(arr[r_idx])
        g_norm = _stretch_channel(arr[g_idx])
        b_norm = _stretch_channel(arr[b_idx])
        rgb = np.stack([r_norm, g_norm, b_norm], axis=-1)
        img = Image.fromarray(rgb, mode="RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return buf


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
    """Converts a GeoJSON FeatureCollection to styled KML XML via simplekml."""
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

        # Class-based styling
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


# ============================================================================
# 1. Endpoint: POST /api/v1/geospatial/composite
# ============================================================================

@router.post("/geospatial/composite")
@router.post("/composite")
async def generate_composite(
    request: Request,
    file: Optional[UploadFile] = File(default=None),
    trace_id: Optional[str] = Form(default=None),
    composite_type: str = Form(default="rgb"),
    as_data_uri: bool = Form(default=False),
    db: Session = Depends(get_optional_db),
):
    """Generate multi-band or SAR visualization composites as streaming PNGs."""
    if rasterio is None:
        raise HTTPException(status_code=503, detail="rasterio is not installed")

    # Resilient fallback: inspect JSON body if requested as application/json
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            body = await request.json()
            if isinstance(body, dict):
                trace_id = body.get("trace_id", trace_id)
                composite_type = body.get("composite_type", composite_type)
                as_data_uri = body.get("as_data_uri", as_data_uri)
        except Exception:
            pass

    # Inspect query parameters fallback
    if not trace_id:
        trace_id = request.query_params.get("trace_id")
    if not composite_type or composite_type == "rgb":
        composite_type = request.query_params.get("composite_type", composite_type)
    if not as_data_uri:
        as_data_uri = request.query_params.get("as_data_uri", "false").lower() in ["true", "1", "yes"]

    engine = SpectralIndicesEngine()

    if file is not None:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        with rasterio.MemoryFile(content) as memfile:
            with memfile.open() as src:
                arr = src.read().astype(np.float32)
                band_map = detect_sensor_profile(src)
    elif trace_id:
        raster_path = _resolve_raster_path_from_trace(trace_id, db=db)
        if not raster_path:
            raise HTTPException(status_code=404, detail=f"No raster found for trace_id '{trace_id}'.")
        with rasterio.open(raster_path) as src:
            arr = src.read().astype(np.float32)
            band_map = detect_sensor_profile(src)
    else:
        raise HTTPException(
            status_code=400,
            detail="Either 'file' multipart upload or 'trace_id' must be provided.",
        )

    if arr.ndim == 2:
        arr = np.expand_dims(arr, 0)

    buf = _render_composite_buffer(arr, band_map, composite_type, engine)

    if as_data_uri:
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return JSONResponse({
            "status": "ok",
            "composite_type": composite_type,
            "data_uri": f"data:image/png;base64,{b64}",
            "width": int(arr.shape[2]),
            "height": int(arr.shape[1]),
        })

    return StreamingResponse(
        buf,
        media_type="image/png",
        headers={"Content-Disposition": f'inline; filename="{composite_type}_composite.png"'},
    )


# ============================================================================
# 2. Endpoint: POST /api/v1/geospatial/inspect-point
# ============================================================================

@router.post("/geospatial/inspect-point")
@router.post("/inspect-point")
async def inspect_point(
    payload: InspectPointRequest,
    db: Session = Depends(get_optional_db),
):
    """Sample localized spectral indices and classify terrain for map pixel tooltips."""
    if rasterio is None:
        raise HTTPException(status_code=503, detail="rasterio is not installed")

    try:
        lat, lon = payload.get_coordinates()
    except ValueError as val_err:
        raise HTTPException(status_code=400, detail=str(val_err)) from val_err

    raster_path = _resolve_raster_path_from_trace(payload.trace_id, db=db)
    if not raster_path:
        raise HTTPException(status_code=404, detail=f"No raster found for trace_id '{payload.trace_id}'.")

    engine = SpectralIndicesEngine()

    with rasterio.open(raster_path) as src:
        band_map = detect_sensor_profile(src)

        # Coordinate transformation: EPSG:4326 to raster CRS
        if src.crs and str(src.crs).upper() not in ["EPSG:4326", "WGS 84"]:
            try:
                xs, ys = transform("EPSG:4326", src.crs, [lon], [lat])
                target_x, target_y = xs[0], ys[0]
            except Exception as tr_err:
                logger.warning("Coordinate re-projection failed (%s); using direct coordinates", tr_err)
                target_x, target_y = lon, lat
        else:
            target_x, target_y = lon, lat

        # Transform (x, y) to pixel coordinates (row, col) via ~affine_transform
        try:
            row, col = src.index(target_x, target_y)
        except Exception:
            inv_transform = ~src.transform
            col_f, row_f = inv_transform * (target_x, target_y)
            row, col = int(round(row_f)), int(round(col_f))

        # Check bounds with graceful clamping for near-boundary clicks
        if not (-5 <= row <= src.height + 5 and -5 <= col <= src.width + 5):
            raise HTTPException(
                status_code=400,
                detail=f"Coordinate ({lat:.5f}, {lon:.5f}) is outside raster extent {src.bounds}.",
            )

        row_clamped = max(0, min(src.height - 1, row))
        col_clamped = max(0, min(src.width - 1, col))

        window = rasterio.windows.Window(col_clamped, row_clamped, 1, 1)
        raw_values = src.read(window=window)[:, 0, 0].astype(np.float32)

    c = len(raw_values)

    def _norm(v: float) -> float:
        val = float(v)
        if val > 255.0:
            return round(val / 10000.0, 4)
        elif val > 1.0:
            return round(val / 255.0, 4)
        return round(val, 4)

    has_real_nir = (band_map.nir is not None and band_map.nir < c) or (c >= 4)
    red_val = float(raw_values[band_map.red]) if (band_map.red is not None and band_map.red < c) else float(raw_values[min(c - 1, 2)])
    green_val = float(raw_values[band_map.green]) if (band_map.green is not None and band_map.green < c) else float(raw_values[min(c - 1, 1)])
    nir_val = float(raw_values[band_map.nir]) if (band_map.nir is not None and band_map.nir < c) else (float(raw_values[min(c - 1, 3)]) if c >= 4 else None)
    blue_val = float(raw_values[band_map.blue]) if (band_map.blue is not None and band_map.blue < c) else float(raw_values[0])
    swir_val = float(raw_values[band_map.swir]) if (band_map.swir is not None and band_map.swir < c) else None

    # Compute indices with 1e-7 division-by-zero protection
    if has_real_nir and nir_val is not None:
        ndvi = round(float((nir_val - red_val) / (nir_val + red_val + 1e-7)), 4)
        ndwi = round(float((green_val - nir_val) / (green_val + nir_val + 1e-7)), 4)
    else:
        # Visible-spectrum approximations for 3-band RGB imagery
        ndvi = round(float((green_val - red_val) / (green_val + red_val + 1e-7)), 4)
        ndwi = round(float((blue_val - red_val) / (blue_val + red_val + 1e-7)), 4)

    bands_dict = {
        "red": _norm(red_val),
        "green": _norm(green_val),
        "blue": _norm(blue_val),
    }
    if nir_val is not None:
        bands_dict["nir"] = _norm(nir_val)
    if swir_val is not None:
        bands_dict["swir"] = _norm(swir_val)

    indices_dict = {
        "ndvi": ndvi,
        "ndwi": ndwi,
    }

    if swir_val is not None:
        mndwi = round(float((green_val - swir_val) / (green_val + swir_val + 1e-7)), 4)
        ndbi = round(float((swir_val - nir_val) / (swir_val + nir_val + 1e-7)), 4)
        indices_dict["mndwi"] = mndwi
        indices_dict["ndbi"] = ndbi

    is_sar_raster = getattr(band_map, "is_sar", False) or ("sar" in getattr(band_map, "sensor_name", "").lower()) or c <= 2
    if is_sar_raster:
        sar_db = round(float(10.0 * np.log10(float(raw_values[0]) ** 2 + 1e-7)), 2)
        indices_dict["sar_db"] = sar_db

    # Terrain Classification
    if ndwi > 0.05 or (ndwi > -0.05 and ndvi < 0.05):
        terrain = "Water Body"
    elif ndvi >= 0.5:
        terrain = "Dense Vegetation"
    elif ndvi >= 0.2:
        terrain = "Moderate / Sparse Vegetation"
    elif ndvi >= 0.0:
        if swir_val is not None and indices_dict.get("ndbi", 0) > 0:
            terrain = "Built-up / Urban Area"
        else:
            terrain = "Low Vegetation / Shrubland"
    else:
        terrain = "Barren Land / Rock / Soil"

    available_comps = engine.get_available_composites(band_map=band_map, band_count=c)

    return {
        "coordinate": {"lat": round(lat, 6), "lon": round(lon, 6)},
        "pixel": {"row": int(row), "col": int(col)},
        "bands": bands_dict,
        "indices": indices_dict,
        "available_composites": available_comps,
        "terrain_classification": terrain,
    }


# ============================================================================
# 3. Endpoint: GET /api/v1/reports/{trace_id}/export
# ============================================================================

@router.get("/reports/{trace_id}/export")
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
        buf = io.BytesIO(kml_str.encode("utf-8"))
        return StreamingResponse(
            buf,
            media_type="application/vnd.google-earth.kml+xml",
            headers={"Content-Disposition": f'attachment; filename="{trace_id}.kml"'},
        )
