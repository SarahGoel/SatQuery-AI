"""Modular tool registry and remote-sensing specialist capabilities."""

from app.tools.base import BaseTool


def __getattr__(name: str):
    if name in (
        "ToolRegistry",
        "WaterGroundingTool",
        "TemporalChangeTool",
        "RemoteCLIPTemporalEncoder",
        "OpticalSARFusionTool",
        "GeodesicMeasurementTool",
        "default_tool_registry",
    ):
        import app.tools.registry as reg

        return getattr(reg, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BaseTool",
    "ToolRegistry",
    "WaterGroundingTool",
    "TemporalChangeTool",
    "RemoteCLIPTemporalEncoder",
    "OpticalSARFusionTool",
    "GeodesicMeasurementTool",
    "default_tool_registry",
]
