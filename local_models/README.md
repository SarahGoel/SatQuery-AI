## Sovereign Weight Registry Guide

For secure, zero-network air-gapped deployments, manually download model weights and place them in the following directories.

Host path `./local_models` is bind-mounted into the backend container at **`/app/models`** and **`/local_models`** (read-only). Do not commit weight binaries.

1. `local_models/sam/`
   - MobileSAM visual backbone checkpoints (e.g., `mobile_sam.pt`).
2. `local_models/bigearthnet/`
   - PEFT/LoRA adapter weights for domain adaptation (`adapter_model.bin`).
3. `local_models/cdvqa/`
   - Checkpoints for temporal difference attention and Change-VQA modules.
4. `local_models/vllm/`
   - Inference weights for local LLM servers (e.g., LLaVA-3B vision-language models).

Create the tree (idempotent):

```bash
bash scripts/init_env.sh
```

Populate binaries on a **connected** machine (`scripts/download_weights.sh` or manual copy), then copy `local_models/` onto the air-gapped node. Do not set `HUGGING_FACE_HUB_TOKEN` on the isolated host.

`GET /api/v1/health` reports `air_gap_ready` only when `/local_models` exists, is readable, and the four registry subdirectories are present.

Verify orchestration and GPU:

```bash
docker compose ps
docker compose exec backend nvidia-smi
```
