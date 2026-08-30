# Local model weights (air-gapped)

This directory is the **only** place the backend should load checkpoints from on an isolated GPU host. Do not commit weight files.

Host path `./local_models` is bind-mounted to **`/local_models`** in the backend container.

## Expected layout (Phase 1)

```
local_models/
  README.md
  sam/            # Segment Anything (and MobileSAM) checkpoints
  bigearthnet/    # BigEarthNet-tuned encoder
  cdvqa/          # Change-detection VQA / temporal attention
  vllm/           # Open-weight VLM snapshots consumed by vLLM / Ollama
```

Create the tree (idempotent):

```bash
bash scripts/init_env.sh
```

Populate binaries on a **connected** machine (`scripts/download_weights.sh` or manual copy), then rsync `local_models/` onto the air-gapped node. Do not set `HUGGING_FACE_HUB_TOKEN` on the isolated host.

`GET /api/v1/health` reports `air_gap_ready` only when `/local_models` exists, is readable, and the four registry subdirectories are present.
