"""Phase 5 Step 10 — strict Pydantic auditable trace schemas."""

from __future__ import annotations

from app.schemas.trace import AuditableTraceLogSchema


def test_auditable_trace_log_schema_constructs() -> None:
    data = {
        "trace_id": "TEST-123",
        "task": "single_image_vqa",
        "query": "Is there vegetation?",
        "input_metadata": {
            "crs": "EPSG:4326",
            "bounds": [0.0, 0.0, 1.0, 1.0],
            "affine_transform": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            "modalities": ["Optical"],
        },
        "registry_execution": [{"model": "VQA-Model", "params": {}}],
        "confidence_score": 0.95,
        "output": "Yes, dense vegetation is present.",
    }
    log = AuditableTraceLogSchema(**data)
    assert log.trace_id == "TEST-123"
    assert log.task == "single_image_vqa"
    assert log.input_metadata.crs == "EPSG:4326"
    assert log.registry_execution[0].model == "VQA-Model"
    assert log.tools_executed[0].model == "VQA-Model"
    assert log.confidence_score == 0.95


def test_tools_executed_accepts_list_of_registry_execution_schema() -> None:
    from app.schemas.trace import InputMetadataSchema, RegistryExecutionSchema

    tools = [
        RegistryExecutionSchema(model="RS-Grounding-V3", params={"threshold": 0.75}),
        RegistryExecutionSchema(model="CD-VQA-Pro", params={"epoch_difference": True}),
        RegistryExecutionSchema(model="Opt-SAR-Fusion-Net", params={"cross_attention": True}),
    ]
    meta = InputMetadataSchema(
        crs="EPSG:4326",
        bounds=[77.0, 28.0, 77.2, 28.2],
        affine_transform=[0.01, 0.0, 77.0, 0.0, -0.01, 28.2],
        modalities=["RGB"],
    )
    log = AuditableTraceLogSchema(
        trace_id="TEST-TOOLS-001",
        task="cross_modal",
        query="Analyze multi-sensor scene",
        input_metadata=meta,
        tools_executed=tools,
        confidence_score=0.92,
        output="Multi-sensor scene analyzed successfully.",
    )
    assert len(log.tools_executed) == 3
    assert log.tools_executed[0].model == "RS-Grounding-V3"
    assert log.tools_executed[1].model == "CD-VQA-Pro"
    assert log.tools_executed[2].model == "Opt-SAR-Fusion-Net"
    assert len(log.registry_execution) == 3
    assert log.registry_execution[0].model == "RS-Grounding-V3"

    dumped = log.model_dump()
    assert isinstance(dumped["tools_executed"], list)
    assert len(dumped["tools_executed"]) == 3
    assert isinstance(dumped["tools_executed"][0], dict)
    assert dumped["tools_executed"][0]["model"] == "RS-Grounding-V3"

    # Reconstruct from dumped dict
    reconstructed = AuditableTraceLogSchema(**dumped)
    assert len(reconstructed.tools_executed) == 3
    assert isinstance(reconstructed.tools_executed[0], RegistryExecutionSchema)


def test_tools_executed_accepts_list_of_dicts() -> None:
    data = {
        "trace_id": "TEST-DICTS-002",
        "task": "single_grounding",
        "query": "Detect vehicles",
        "input_metadata": {
            "crs": "EPSG:4326",
            "bounds": [0.0, 0.0, 1.0, 1.0],
            "affine_transform": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            "modalities": ["Optical"],
        },
        "tools_executed": [
            {"model": "RS-Grounding-V3", "params": {"threshold": 0.8}},
            {"model": "mobilesam", "params": {}},
        ],
        "confidence_score": 0.89,
        "output": "Vehicles detected.",
    }
    log = AuditableTraceLogSchema(**data)
    assert len(log.tools_executed) == 2
    assert log.tools_executed[0].model == "RS-Grounding-V3"
    assert log.tools_executed[1].model == "mobilesam"
    assert len(log.registry_execution) == 2
    assert log.registry_execution[0].model == "RS-Grounding-V3"
