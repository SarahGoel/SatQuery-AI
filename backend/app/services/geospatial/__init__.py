"""Geospatial processing (no GPU weights)."""

from app.services.geospatial.spectral import (
    BandMapping,
    SpectralIndicesEngine,
    SpectralIndicesResult,
    compute_ndvi,
    compute_ndwi,
    compute_mndwi,
    compute_ndbi,
    generate_cir_composite,
    generate_sar_db,
    render_colormap,
    detect_sensor_profile,
)

__all__ = [
    "BandMapping",
    "SpectralIndicesEngine",
    "SpectralIndicesResult",
    "compute_ndvi",
    "compute_ndwi",
    "compute_mndwi",
    "compute_ndbi",
    "generate_cir_composite",
    "generate_sar_db",
    "render_colormap",
    "detect_sensor_profile",
]
