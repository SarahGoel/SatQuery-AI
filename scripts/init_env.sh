#!/usr/bin/env bash
# SatQuery AI — create air-gapped weight registries and Phase 1 directories.
# Safe to re-run. Does not download checkpoints (use download_weights.sh on a
# connected host, then copy local_models/ onto the isolated GPU node).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_HOST="${LOCAL_MODELS_HOST_DIR:-$ROOT/local_models}"

echo "[satquery] Repository root: ${ROOT}"
echo "[satquery] Host weights directory: ${MODELS_HOST}"

mkdir -p \
  "${MODELS_HOST}/sam" \
  "${MODELS_HOST}/bigearthnet" \
  "${MODELS_HOST}/cdvqa" \
  "${MODELS_HOST}/vllm" \
  "${ROOT}/models/inference" \
  "${ROOT}/models/weights" \
  "${ROOT}/models/configs" \
  "${ROOT}/data/raw" \
  "${ROOT}/data/processed" \
  "${ROOT}/backend/api" \
  "${ROOT}/backend/validation" \
  "${ROOT}/backend/query_understanding" \
  "${ROOT}/backend/agent" \
  "${ROOT}/backend/preprocessing" \
  "${ROOT}/backend/fusion" \
  "${ROOT}/backend/evidence" \
  "${ROOT}/backend/confidence" \
  "${ROOT}/backend/storage" \
  "${ROOT}/tests"

# Keep empty registries visible in `ls` without committing weight binaries.
for sub in sam bigearthnet cdvqa vllm; do
  keep="${MODELS_HOST}/${sub}/.keep"
  if [[ ! -e "${keep}" ]]; then
    printf 'SatQuery AI offline registry: %s\nDrop checkpoints here. Do not commit binaries.\n' "${sub}" > "${keep}"
  fi
done

for keep_dir in \
  "${ROOT}/models/inference" \
  "${ROOT}/models/weights" \
  "${ROOT}/models/configs" \
  "${ROOT}/data/raw" \
  "${ROOT}/data/processed"
do
  if [[ ! -e "${keep_dir}/.gitkeep" ]]; then
    : > "${keep_dir}/.gitkeep"
  fi
done

if [[ ! -f "${ROOT}/.env" && -f "${ROOT}/.env.example" ]]; then
  cp "${ROOT}/.env.example" "${ROOT}/.env"
  echo "[satquery] Wrote ${ROOT}/.env from .env.example (edit credentials locally)."
fi

echo "[satquery] Host bind maps ${MODELS_HOST} -> /local_models and /app/models inside the backend container."
echo "[satquery] Registries:"
find "${MODELS_HOST}" -mindepth 1 -maxdepth 1 -type d | sort
echo "[satquery] Done. Next: docker compose up --build"
