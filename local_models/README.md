# Local model weights (air-gapped)

This directory is the **only** place the backend and inference servers should load checkpoints from. Do not commit weight files. Copy this tree onto the isolated GPU host after prefetching on a connected machine.

## Expected layout

```
local_models/
  README.md                 # this file
  llava-3b/                 # Open-weight VLM consumed by Ollama or vLLM
  sam/
    sam_vit_b.pth           # Segment Anything ViT-B
  mobilesam/
    mobile_sam.pt
  bigearthnet/
    checkpoint.pt           # BigEarthNet-tuned encoder (team checkpoint)
  fusion/
    cross_attention.pt     # Optical–SAR fusion (Feature 2.3)
  change_vqa/
    temporal_attn.pt       # Temporal Attention Change-VQA (Feature 2.5)
```

## How to populate (connected machine)

From the repository root:

```bash
bash scripts/download_weights.sh
```

The script uses `huggingface-cli` / `huggingface_hub` and writes under this folder. After it finishes:

1. Verify checksums printed by the script (or `sha256sum` locally).
2. `docker compose` bind-mounts `./local_models` to `/models` in backend, Ollama, and vLLM.
3. Set `LOCAL_MODELS_DIR=/models` (already the compose default).
4. Disconnect the host. Do not set `HUGGING_FACE_HUB_TOKEN` on the air-gapped node.

## Ollama

Once files are present, create a local model (example):

```bash
docker compose exec ollama ollama create llava-3b -f /models/llava-3b/Modelfile
```

If you vendor a GGUF instead of a Transformers snapshot, point the Modelfile `FROM` at that GGUF.

## vLLM

Enable the compose profile and pass `--model /models/llava-3b`. The container must **not** attempt Hub downloads (`HF_HUB_OFFLINE=1` is recommended on isolated hosts).

## Environment overrides

| Variable | Default |
| --- | --- |
| `LOCAL_MODELS_DIR` | `/models` |
| `SAM_WEIGHTS_PATH` | `$LOCAL_MODELS_DIR/sam/sam_vit_b.pth` |
| `MOBILESAM_WEIGHTS_PATH` | `$LOCAL_MODELS_DIR/mobilesam/mobile_sam.pt` |
| `BIGEARTHNET_CHECKPOINT` | `$LOCAL_MODELS_DIR/bigearthnet/checkpoint.pt` |
| `FUSION_CHECKPOINT` | optional |
| `CHANGE_VQA_CHECKPOINT` | optional |

Modules load `strict=False` so you can drop official SAM / PEFT checkpoints without changing Python until the team wires exact `state_dict` keys.
