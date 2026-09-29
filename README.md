# SatQuery AI: Autonomous Multimodal Earth Observation Intelligence Console

[![ISRO SAC Compliant](https://img.shields.io/badge/ISRO%20SAC-SIH26167-blue.svg)](https://www.isro.gov.in/)
[![Architecture](https://img.shields.io/badge/Architecture-Dual--Engine%20VLM%20%2B%20Agentic%20Router-teal.svg)](#system-architecture)
[![Evaluation](https://img.shields.io/badge/Air--Gap%20Ready-Verified-green.svg)](#air-gapped-sovereign-deployment)
[![License](https://img.shields.io/badge/License-MIT-gray.svg)](LICENSE)

SatQuery AI is an agentic, multi-modal Earth Observation (EO) platform engineered for the **Indian Space Research Organisation (ISRO) Space Applications Centre (SAC)**. Designed to analyze single, co-registered optical-SAR, and bi-temporal satellite rasters through natural-language queries, SatQuery AI translates user intent into deterministic remote-sensing pipelines, vector-grounded segmentation masks, and auditable scientific intelligence.

---

## Evaluator Quick Start (One-Step Run)

### System Prerequisites
- **Operating System:** Linux (Ubuntu 20.04+ recommended), macOS, or Windows 10/11 with WSL2.
- **Container Runtime:** [Docker Engine](https://docs.docker.com/engine/install/) (v24.0+) and [Docker Compose](https://docs.docker.com/compose/install/) (v2.20+).
- *No local Python, Node.js, GDAL, or database installations are needed.*

### 1. Launch Platform
Clone the repository and execute the master initialization script:

```bash
git clone https://github.com/SarahGoel/SatQuery-AI
cd SatQueryAI
chmod +x run.sh
./run.sh
```

### 2. Access the Applications
- **Analyst Web Console:** [http://localhost:5173](http://localhost:5173)
- **FastAPI OpenAPI Interactive Documentation:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Air-Gap Verification Diagnostic Probe:** [http://localhost:8000/api/v1/health/air-gap](http://localhost:8000/api/v1/health/air-gap)

---

## VLM Inference Configuration

SatQuery AI supports both instant cloud evaluation and sovereign air-gapped evaluation. Configure your selection in `backend/.env`:

### Option A: Cloud Inference (Fastest — 2-Second Turnaround)
1. Get a free API key from [Google AI Studio](https://aistudio.google.com/).
2. Edit `backend/.env`:
   ```env
   VLM_PROVIDER=gemini
   GEMINI_API_KEY=AIzaSyYourActualKeyHere
   ```
3. Restart the backend: `docker compose restart backend`.

### Option B: Sovereign Air-Gapped Mode (100% Offline)
Leave the API key blank or configure Ollama:
```env
VLM_PROVIDER=ollama
GEMINI_API_KEY=
OLLAMA_BASE_URL=http://satquery_ollama:11434
```
*Note: In offline CPU environments, heavy vision tensor calculations take approximately 60–90 seconds per query.*

---

## Core Evaluation Scenarios

| # | Query Archetype | Sample Evaluation Query | Expected System Action |
| :-: | :--- | :--- | :--- |
| **1** | **Scene VQA** | *"Describe the land cover and major objects visible in this image."* | Returns a concise 6-to-7 line domain-grounded summary distinguishing agricultural cropland from forest and bare soil. |
| **2** | **Feature Grounding** | *"Highlight the water bodies referred to in the query."* | Calculates NDWI anchor points, executes MobileSAM, and generates an exact vector polygon with area in $\text{km}^2$. |
| **3** | **Bi-Temporal Change** | *"What changed between these two dates and where did the change occur?"* | Calculates Siamese difference masks and outputs localized directional findings (`[INCREASED]`, `[DECREASED]`). |
| **4** | **Cross-Modal Fusion** | *"Use the optical and SAR image together to identify build-up and water-covered regions."* | SIFT-aligns both rasters, identifies water using SAR specular drop ($\sigma_0 < -18\text{ dB}$), and isolates built-up areas via optical-SAR texture fusion. |
| **5** | **Class Tracking** | *"Has the built-up area increased, decreased, or remained unchanged?"* | Computes NDBI deltas across T1 and T2 to output verified urban development metrics. |

---

## Stopping the Platform

To cleanly stop all running services and network bridges:
```bash
docker compose down
```

To remove containers and wipe associated database volumes:
```bash
docker compose down -v
```
