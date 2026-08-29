# SatQuery AI

**Smart India Hackathon 2025 — Problem Statement SIH26167**

Agentic vision-language assistant for ISRO Earth Observation (EO) data analysis. The stack is designed for **sovereign, air-gapped, on-premise GPU deployments**: geospatial I/O, deep-learning inference, and the analyst UI are isolated services that never require an outbound network after weights are staged.

## What it does

Analysts submit a natural-language query plus one or more GeoTIFFs (optical Cartosat-2S, SAR RISAT, optional T1/T2 pair). A stateful controller:

1. Parses GeoTIFF CRS, affine transform, and bounds (Rasterio).
2. Validates spatial overlap (Shapely / PostGIS).
3. Classifies the query into `bi_temporal_change_analysis`, `single_image_grounding`, `cross_modal_joint_analysis`, or `single_image_vqa`.
4. Routes work through alignment, spectral indices, fusion, grounding, or Change-VQA modules.
5. Persists an auditable execution trace (parameters, models, confidence, output).

## Tech stack

| Layer | Technology |
| --- | --- |
| UI | React, OpenLayers 9, Tailwind CSS |
| API | FastAPI, Uvicorn, Pydantic v2.6.1 |
| Geospatial | GDAL 3.8+, Rasterio 1.3.9, Shapely 2.0.2, Fiona 1.9.5, PostGIS |
| Inference | PyTorch 2.2.1, Transformers 4.38.1, PEFT 0.8.2, LangGraph 0.1.1 |
| Serving | Ollama (default) or vLLM — local open-weight VLMs (e.g. LLaVA-3B) |
| Persistence | PostgreSQL 16 + PostGIS 3.4 |

## Repository layout

```
.
├── docker-compose.yml          # React, FastAPI, PostGIS, Ollama / vLLM
├── db/init-postgis.sh         # Enable PostGIS on first boot
├── backend/                  # GDAL-enabled FastAPI + agent + DL modules
├── frontend/                 # Analyst console (map, query, traces, PDF)
├── local_models/             # Offline weight drop zone (not committed)
└── scripts/                  # Weight prefetch + CLI pipeline smoke test
```

Geospatial manipulation lives under `backend/app/services/geospatial/`. Deep-learning inference lives under `backend/app/services/models/`. The UI never talks to the GPU or GDAL directly.

## Prerequisites

- Docker Engine 24+ with Compose v2
- NVIDIA Container Toolkit (on-premise GPU hosts)
- 16+ GB GPU VRAM recommended for LLaVA-class models
- For local Python without Docker: Python 3.11, GDAL 3.8+ system libraries

## Quick start (local, networked)

1. Copy environment defaults (optional):

   ```bash
   cp .env.example .env
   ```

2. Prefetch open-weight checkpoints **before** disconnecting from the internet:

   ```bash
   bash scripts/download_weights.sh
   ```

3. Start PostGIS, API, UI, and Ollama:

   ```bash
   docker compose up --build
   ```

4. Open the analyst console at [http://localhost:3000](http://localhost:3000).  
   API docs: [http://localhost:8000/docs](http://localhost:8000/docs).  
   Health: [http://localhost:8000/api/v1/health](http://localhost:8000/api/v1/health).

vLLM profile (optional):

```bash
docker compose --profile vllm up --build
```

Then set `INFERENCE_BACKEND=vllm` on the backend service.

## Air-gapped / on-premise GPU

1. On a connected machine, run `scripts/download_weights.sh` and copy `local_models/` onto the isolated host (see `local_models/README.md`).
2. Load container images from an internal registry or `docker save` / `docker load` tarballs.
3. Point `LOCAL_MODELS_DIR` at the mounted weight volume. Do not set Hugging Face tokens.
4. Confirm `INFERENCE_BACKEND` is `ollama` or `vllm` and that the serving container can read `/models`.
5. Run `docker compose up` with **no** outbound routes. Health must report `gpu_available` and `models_dir_present`.

## Backend without Docker

```bash
cd backend
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
export DATABASE_URL=postgresql://satquery:satquery@localhost:5432/satquery
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

GDAL/Rasterio must resolve native libraries (`gdal-config` on PATH).

## Frontend without Docker

```bash
cd frontend
npm install
npm run dev
```

Vite proxies `/api` to `http://localhost:8000`.

## CLI smoke test

```bash
python scripts/test_pipeline.py
```

Uses mocked GeoTIFF metadata (no GPU required) to exercise classification and workflow routing.

## Evaluation guidelines (SIH26167)

Judges and internal reviewers should exercise:

| Criterion | How to evaluate |
| --- | --- |
| Spatial correctness | Upload two overlapping GeoTIFFs; traces must record CRS, affine matrix, and EPSG:4326 bounding polygon. Misaligned pairs must fail `validate_spatial_alignment`. |
| Task routing | Queries mentioning “change / T1 / T2” → `bi_temporal_change_analysis`; “highlight / segment / locate” → `single_image_grounding`; optical+SAR → `cross_modal_joint_analysis`; otherwise VQA. |
| Spectral products | NDVI / NDWI tensors appear in the trace `registry_execution` parameters for optical workflows. |
| Auditability | Every `/api/v1/satquery/analyze` response includes `AuditableTraceLogSchema`; PostGIS `auditable_execution_traces` and `trace_model_executions` rows must match. |
| Sovereignty | With network disabled, inference still runs from `local_models/` via Ollama or vLLM. |
| UI | MapViewer overlays GeoJSON / change rasters; TraceViewer shows the live JSON log; ReportDownloader exports PDF. |

Do **not** ship proprietary ISRO scenes in this repository. Use synthetic or openly licensed GeoTIFFs for demos.

## API surface

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/v1/health` | Process, PostGIS, and GPU metrics |
| `POST` | `/api/v1/satquery/analyze` | Multipart GeoTIFF + natural-language query |

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE).
