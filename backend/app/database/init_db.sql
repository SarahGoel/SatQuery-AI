-- Core Database Schema for SatQuery AI Platform
-- Applied on first cluster boot via /docker-entrypoint-initdb.d (after PostGIS init).
-- Boundaries use EPSG:4326 (GEOMETRY(Polygon, 4326)).

CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 1. Table to trace each session/execution
CREATE TABLE IF NOT EXISTS auditable_execution_traces (
    trace_id VARCHAR(64) PRIMARY KEY,
    task_type VARCHAR(64) NOT NULL,
    user_query TEXT NOT NULL,
    crs VARCHAR(32) NOT NULL,
    affine_transform_matrix DOUBLE PRECISION[] NOT NULL,
    bounding_box_geometry GEOMETRY(Polygon, 4326) NOT NULL,
    overall_confidence DOUBLE PRECISION NOT NULL,
    final_output TEXT NOT NULL,
    executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. Table to register available models
CREATE TABLE IF NOT EXISTS model_registry (
    model_name VARCHAR(64) PRIMARY KEY,
    model_version VARCHAR(16) NOT NULL,
    model_type VARCHAR(32) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    local_weights_path TEXT NOT NULL
);

-- 3. Table to map executions of models per trace log
CREATE TABLE IF NOT EXISTS trace_model_executions (
    execution_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    trace_id VARCHAR(64) REFERENCES auditable_execution_traces(trace_id) ON DELETE CASCADE,
    model_name VARCHAR(64) REFERENCES model_registry(model_name) ON DELETE RESTRICT,
    parameter_configuration JSONB NOT NULL,
    execution_order INT NOT NULL
);

-- Indexes for performance and georeferenced spatial calculations
CREATE INDEX IF NOT EXISTS idx_traces_spatial ON auditable_execution_traces USING GIST (bounding_box_geometry);
CREATE INDEX IF NOT EXISTS idx_executions_trace ON trace_model_executions (trace_id);

-- Seed the offline weight registry so trace FKs resolve on a fresh cluster.
INSERT INTO model_registry (model_name, model_version, model_type, is_active, local_weights_path)
VALUES
    ('llava-3b', '0.1.0', 'vlm', TRUE, '/app/models/vllm'),
    ('sam-vit-b', '1.0.0', 'segmenter', TRUE, '/app/models/sam/sam_vit_b.pth'),
    ('mobilesam', '1.0.0', 'segmenter', TRUE, '/app/models/sam/mobile_sam.pt'),
    ('optical-sar-fusion', '0.1.0', 'fusion', TRUE, '/app/models/fusion/cross_attention.pt'),
    ('change-vqa', '0.1.0', 'change_vqa', TRUE, '/app/models/cdvqa'),
    ('bigearthnet-encoder', '0.1.0', 'encoder', TRUE, '/app/models/bigearthnet/adapter_model.bin'),
    ('spatial-aligner', '0.1.0', 'geospatial', TRUE, '/app/app/services/geospatial/alignment.py'),
    ('spectral-extractor', '0.1.0', 'geospatial', TRUE, '/app/app/services/geospatial/spectral.py'),
    ('RS-Grounding-V3', '3.0.0', 'grounding', TRUE, '/local_models/sam/mobile_sam.pt'),
    ('SAR-Structure-Extractor', '1.0.0', 'sar_processing', TRUE, '/local_models/fusion/cross_attention.pt'),
    ('CD-VQA-Pro', '1.0.0', 'change_detection', TRUE, '/local_models/change_vqa/temporal_attn.pt'),
    ('Opt-SAR-Fusion-Net', '1.0.0', 'fusion', TRUE, '/local_models/fusion/cross_attention.pt')
ON CONFLICT (model_name) DO NOTHING;
