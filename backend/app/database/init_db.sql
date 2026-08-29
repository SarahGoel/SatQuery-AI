-- SatQuery AI — PostGIS schema (SIH26167)
-- Applied on first container boot via /docker-entrypoint-initdb.d.
-- Geometry is stored in EPSG:4326 for audit and map overlay.

CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS auditable_execution_traces (
    trace_id                VARCHAR(64) PRIMARY KEY,
    task_type               VARCHAR(64) NOT NULL,
    user_query              TEXT NOT NULL,
    crs                     VARCHAR(128),
    affine_transform_matrix  JSONB,
    bounding_box_geometry    GEOMETRY(Polygon, 4326),
    overall_confidence      DOUBLE PRECISION,
    final_output            TEXT,
    executed_at            TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_auditable_traces_executed_at
    ON auditable_execution_traces (executed_at DESC);

CREATE INDEX IF NOT EXISTS idx_auditable_traces_task_type
    ON auditable_execution_traces (task_type);

CREATE INDEX IF NOT EXISTS idx_auditable_traces_bbox
    ON auditable_execution_traces
    USING GIST (bounding_box_geometry);

CREATE TABLE IF NOT EXISTS model_registry (
    model_name          VARCHAR(128) PRIMARY KEY,
    model_version       VARCHAR(64) NOT NULL,
    model_type          VARCHAR(64) NOT NULL,
    is_active           BOOLEAN NOT NULL DEFAULT TRUE,
    local_weights_path TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS trace_model_executions (
    execution_id            UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    trace_id                VARCHAR(64) NOT NULL
        REFERENCES auditable_execution_traces (trace_id) ON DELETE CASCADE,
    model_name              VARCHAR(128) NOT NULL
        REFERENCES model_registry (model_name),
    parameter_configuration  JSONB,
    execution_order         INTEGER NOT NULL,
    CONSTRAINT uq_trace_execution_order UNIQUE (trace_id, execution_order)
);

CREATE INDEX IF NOT EXISTS idx_trace_model_executions_trace
    ON trace_model_executions (trace_id);

INSERT INTO model_registry (model_name, model_version, model_type, is_active, local_weights_path)
VALUES
    ('llava-3b', '0.1.0', 'vlm', TRUE, '/models/llava-3b'),
    ('sam-vit-b', '1.0.0', 'segmenter', TRUE, '/models/sam/sam_vit_b.pth'),
    ('mobilesam', '1.0.0', 'segmenter', TRUE, '/models/mobilesam/mobile_sam.pt'),
    ('optical-sar-fusion', '0.1.0', 'fusion', TRUE, '/models/fusion/cross_attention.pt'),
    ('change-vqa', '0.1.0', 'change_vqa', TRUE, '/models/change_vqa/temporal_attn.pt'),
    ('bigearthnet-encoder', '0.1.0', 'encoder', TRUE, '/models/bigearthnet/checkpoint.pt'),
    ('spatial-aligner', '0.1.0', 'geospatial', TRUE, '/app/app/services/geospatial/alignment.py'),
    ('spectral-extractor', '0.1.0', 'geospatial', TRUE, '/app/app/services/geospatial/spectral.py')
ON CONFLICT (model_name) DO NOTHING;
