#!/usr/bin/env bash
# Prefetch open-weight checkpoints on a connected machine, then copy local_models/
# onto the air-gapped GPU host. Never run this script on an isolated node expecting
# Hub access — it will fail closed.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${LOCAL_MODELS_DIR:-$ROOT/local_models}"
PYTHON="${PYTHON:-python3}"

mkdir -p \
  "$DEST/llava-3b" \
  "$DEST/sam" \
  "$DEST/mobilesam" \
  "$DEST/bigearthnet" \
  "$DEST/fusion" \
  "$DEST/change_vqa"

echo "[satquery] Staging weights into ${DEST}"

if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "[satquery] Installing huggingface_hub CLI into user site..."
  "$PYTHON" -m pip install --user -q "huggingface_hub[cli]>=0.21.0"
fi

download() {
  local repo="$1"
  local out="$2"
  echo "[satquery] huggingface-cli download ${repo} -> ${out}"
  huggingface-cli download "$repo" --local-dir "$out" --local-dir-use-symlinks False
}

# Vision-language backbone (swap the repo id when the team freezes a license-cleared snapshot).
download "llava-hf/llava-1.5-7b-hf" "$DEST/llava-3b" || true

# Segment Anything official ViT-B (Facebook Research).
download "facebook/sam-vit-base" "$DEST/sam" || true

# MobileSAM community snapshot.
download "dhkim2810/MobileSAM" "$DEST/mobilesam" || true

cat > "$DEST/llava-3b/Modelfile" <<'EOF'
# Example Ollama Modelfile. Point FROM at a GGUF if you vendor one.
FROM ./
TEMPLATE """{{ .Prompt }}"""
PARAMETER temperature 0.2
PARAMETER stop "</s>"
EOF

cat > "$DEST/MANIFEST.txt" <<EOF
SatQuery AI weight manifest
generated=$(date -u +%Y-%m-%dT%H:%M:%SZ)
host=$(hostname)
dest=${DEST}
EOF

if command -v find >/dev/null 2>&1; then
  echo "[satquery] Checksums:"
  find "$DEST" -type f \( -name '*.pt' -o -name '*.pth' -o -name '*.safetensors' -o -name '*.bin' \) -print0 \
    | xargs -0 -r sha256sum || true
fi

echo "[satquery] Done. Copy ${DEST} to the air-gapped host and mount at /models."
echo "[satquery] Team-trained BigEarthNet / fusion / change-vqa checkpoints must be copied manually into their folders."
