## Sovereign Weight Registry Guide

For secure, zero-network air-gapped deployments, manually download model weights and place them in the following directories [2, 14, 15]:

1. `local_models/sam/`
   - MobileSAM visual backbone check points (e.g., `mobile_sam.pt`) [14, 16].
2. `local_models/bigearthnet/`
   - PEFT/LoRA adapter weights for domain adaptation (`adapter_model.bin`) [14, 16].
3. `local_models/cdvqa/`
   - Checkpoints for temporal difference attention and Change-VQA modules [14, 16].
4. `local_models/vllm/`
   - Inference weights for local LLM servers (e.g., LLaVA-3B vision-language models) [14, 16, 17].

Host path `./local_models` is bind-mounted into the backend container at `/app/models`. Do not commit weight binaries.

Create the tree (idempotent):

```bash
bash scripts/init_env.sh
```

Populate binaries on a connected machine (`scripts/download_weights.sh` or manual copy), then copy `local_models/` onto the air-gapped node.

Verify orchestration and GPU:

```bash
docker-compose ps
docker-compose exec backend nvidia-smi
```
