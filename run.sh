#!/usr/bin/env bash
# SatQuery AI — master orchestrator for sovereign, air-gapped stack bring-up.
# Usage:
#   ./run.sh --setup   Full provisioning on a fresh host
#   ./run.sh --start   Fast start (default) when images and weights already exist
#   ./run.sh           Same as --start
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

RUNTIME_DIR="${ROOT}/.satquery"
RUNTIME_COMPOSE="${RUNTIME_DIR}/docker-compose.runtime.yml"
CPU_MARKER="${RUNTIME_DIR}/cpu-mode"
COMPOSE_FILE="${ROOT}/docker-compose.yml"
GPU_ENABLED="Disabled"

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

DB_CONTAINER="satquery_db"
BACKEND_CONTAINER="satquery_backend"
FRONTEND_CONTAINER="satquery_frontend"
POSTGRES_USER="${POSTGRES_USER:-satquery_admin}"
POSTGRES_DB="${POSTGRES_DB:-satquery_gis}"
POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-isro_secure_db}"

PORTS=(5432 8000 3000)
PORT_ROLES=(PostGIS FastAPI React)

log_info()  { echo -e "${GREEN}[INFO] $(date +'%Y-%m-%d %H:%M:%S')${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[WARN] $(date +'%Y-%m-%d %H:%M:%S')${NC} $*"; }
log_error() { echo -e "${RED}[ERROR] $(date +'%Y-%m-%d %H:%M:%S')${NC} $*" >&2; exit 1; }

on_err() {
  trap - ERR
  echo -e "${RED}[ERROR] $(date +'%Y-%m-%d %H:%M:%S')${NC} Unhandled failure at line ${BASH_LINENO[0]}: ${BASH_COMMAND}" >&2
  exit 1
}
trap on_err ERR

usage() {
  cat <<'EOF'
SatQuery AI master launcher

  ./run.sh --setup    End-to-end init: diagnostics, weights, builds, PostGIS, health
  ./run.sh --start    Fast boot of existing containers (default)
  ./run.sh            Same as --start
  ./run.sh --help     Show this help

Required host tools: bash, Docker Engine, Docker Compose (v1 binary or v2 plugin).
GPU hosts additionally need nvidia-smi and the NVIDIA Container Toolkit.
EOF
}

MODE="start"
if [[ $# -gt 0 ]]; then
  for arg in "$@"; do
    case "${arg}" in
      --setup ) MODE="setup" ;;
      --start ) MODE="start" ;;
      -h|--help ) usage; exit 0 ;;
      * ) log_error "Unknown argument '${arg}'. Run ./run.sh --help" ;;
    esac
  done
fi

mkdir -p "$RUNTIME_DIR"

# ---------------------------------------------------------------------------
# Compose alias (docker compose V2 plugin vs docker-compose V1)
# ---------------------------------------------------------------------------
COMPOSE_BIN=()
detect_compose() {
  if ! command -v docker >/dev/null 2>&1; then
    log_error "Docker is not installed or not on PATH. Install Docker Engine 24+ and retry."
  fi
  if ! docker info >/dev/null 2>&1; then
    log_error "Docker daemon is not reachable. Start Docker Desktop / dockerd, then retry."
  fi
  if docker compose version >/dev/null 2>&1; then
    COMPOSE_BIN=(docker compose)
    log_info "Compose engine: docker compose (V2 plugin) — $(docker compose version --short 2>/dev/null || docker compose version | head -n1)"
  elif command -v docker-compose >/dev/null 2>&1; then
    COMPOSE_BIN=(docker-compose)
    log_info "Compose engine: docker-compose (V1) — $(docker-compose version --short 2>/dev/null || docker-compose --version)"
  else
    log_error "Neither 'docker compose' nor 'docker-compose' is installed. Install Compose V2 and retry."
  fi
}

compose() {
  "${COMPOSE_BIN[@]}" --project-directory "$ROOT" -p satquery-ai -f "$COMPOSE_FILE" "$@"
}

# ---------------------------------------------------------------------------
# Port diagnostics
# ---------------------------------------------------------------------------
port_listener_pids() {
  local port="$1"
  local pids=""
  if command -v lsof >/dev/null 2>&1; then
    pids="$(lsof -nP -iTCP:"${port}" -sTCP:LISTEN -t 2>/dev/null || true)"
  fi
  if [[ -z "${pids}" ]] && command -v ss >/dev/null 2>&1; then
    pids="$(ss -lptn "sport = :${port}" 2>/dev/null | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' | sort -u || true)"
  fi
  if [[ -z "${pids}" ]] && command -v netstat >/dev/null 2>&1; then
    pids="$(netstat -ano 2>/dev/null | awk -v needle=":${port}" '
      $0 ~ needle && /LISTEN/ { print $NF }
    ' | sort -u || true)"
  fi
  echo "${pids}" | tr '\n' ' ' | xargs 2>/dev/null || true
}

our_stack_owns_port() {
  local port="$1"
  docker ps --filter "name=satquery_" --format '{{.Names}} {{.Ports}}' 2>/dev/null \
    | grep -E ":${port}->" >/dev/null 2>&1
}

check_ports() {
  local idx port role pids cmd
  log_info "Checking host bind availability for PostGIS (5432), FastAPI (8000), React (3000)..."
  idx=0
  for port in "${PORTS[@]}"; do
    role="${PORT_ROLES[$idx]}"
    idx=$((idx + 1))
    pids="$(port_listener_pids "${port}")"
    if [[ -z "${pids}" ]]; then
      log_info "Port ${port} (${role}) is free."
      continue
    fi
    if our_stack_owns_port "${port}"; then
      log_info "Port ${port} (${role}) is already owned by the SatQuery stack — will reuse."
      continue
    fi
    cmd=""
    for pid in ${pids}; do
      if [[ "${pid}" =~ ^[0-9]+$ ]] && [[ -r "/proc/${pid}/comm" ]]; then
        cmd+=" ${pid}:$(tr -d '\0' < "/proc/${pid}/comm")"
      else
        cmd+=" ${pid}"
      fi
    done
    echo -e "${RED}Port ${port} (${role}) is bound by foreign PID(s):${cmd}${NC}"
    echo "Action plan:"
    echo "  1. Identify the process:  lsof -nP -iTCP:${port} -sTCP:LISTEN   (or  netstat -ano | grep ${port})"
    echo "  2. Stop it if it is safe, or change SATQUERY published ports in docker-compose.yml / .env"
    echo "  3. If this is a leftover SatQuery container:  ${COMPOSE_BIN[*]:-docker compose} -f docker-compose.yml down"
    echo "  4. Re-run:  ./run.sh --setup"
    log_error "Refusing to start while port ${port} is occupied by a non-SatQuery process."
  done
}

# ---------------------------------------------------------------------------
# GPU / NVIDIA container runtime
# ---------------------------------------------------------------------------
strip_gpu_from_compose() {
  mkdir -p "$RUNTIME_DIR"
  awk '
    BEGIN { skip=0; indent=-1 }
    /^[[:space:]]*deploy:[[:space:]]*$/ {
      match($0, /^[[:space:]]*/)
      indent = RLENGTH
      skip = 1
      next
    }
    skip == 1 {
      if ($0 ~ /^[[:space:]]*$/) next
      match($0, /^[[:space:]]*/)
      if (RLENGTH > indent) next
      skip = 0
    }
    { print }
  ' "${ROOT}/docker-compose.yml" > "${RUNTIME_COMPOSE}"
  COMPOSE_FILE="${RUNTIME_COMPOSE}"
  echo "cpu" > "${CPU_MARKER}"
  log_warn "Wrote CPU-fallback Compose file at ${RUNTIME_COMPOSE} (NVIDIA device reservations removed)."
}

detect_gpu() {
  GPU_ENABLED="Disabled"
  local have_smi=0 have_runtime=0
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    have_smi=1
    log_info "nvidia-smi detected: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -n1 || echo present)"
  else
    log_warn "nvidia-smi is missing or failed. CUDA devices will not be used."
  fi

  local runtimes=""
  runtimes="$(docker info --format '{{range $k,$v := .Runtimes}}{{$k}} {{end}}' 2>/dev/null || true)"
  if echo "${runtimes}" | grep -qi nvidia; then
    have_runtime=1
    log_info "Docker NVIDIA runtime is registered (${runtimes})."
  elif docker info 2>/dev/null | grep -qi 'nvidia'; then
    have_runtime=1
    log_info "Docker reports NVIDIA hooks in daemon info."
  elif [[ -f /usr/bin/nvidia-container-runtime ]] || [[ -f /usr/bin/nvidia-container-toolkit ]]; then
    have_runtime=1
    log_info "NVIDIA container toolkit binaries are present on the host."
  else
    log_warn "NVIDIA Container Toolkit / nvidia Docker runtime is not registered."
  fi

  if [[ "${have_smi}" -eq 1 && "${have_runtime}" -eq 1 ]]; then
    GPU_ENABLED="Enabled"
    rm -f "${CPU_MARKER}"
    COMPOSE_FILE="${ROOT}/docker-compose.yml"
    log_info "Sovereign GPU path: NVIDIA reservations in docker-compose.yml will be honoured."
    return 0
  fi

  log_warn "Falling back to CPU execution so developer laptops can still boot the stack."
  strip_gpu_from_compose
}

select_compose_file() {
  if [[ "${GPU_ENABLED}" == "Enabled" ]]; then
    COMPOSE_FILE="${ROOT}/docker-compose.yml"
    return
  fi
  if [[ -f "${RUNTIME_COMPOSE}" ]]; then
    COMPOSE_FILE="${RUNTIME_COMPOSE}"
  elif [[ -f "${CPU_MARKER}" ]]; then
    strip_gpu_from_compose
  else
    COMPOSE_FILE="${ROOT}/docker-compose.yml"
  fi
}

# ---------------------------------------------------------------------------
# Connectivity + weight cache
# ---------------------------------------------------------------------------
host_is_online() {
  if command -v curl >/dev/null 2>&1; then
    curl -fsS --max-time 4 -I https://huggingface.co >/dev/null 2>&1 && return 0
    curl -fsS --max-time 4 -I https://github.com >/dev/null 2>&1 && return 0
  fi
  if command -v ping >/dev/null 2>&1; then
    ping -c 1 -W 2 1.1.1.1 >/dev/null 2>&1 && return 0
    ping -n 1 -w 2000 1.1.1.1 >/dev/null 2>&1 && return 0
  fi
  return 1
}

weight_hits() {
  find "${ROOT}/local_models" \
    \( -name '*.pt' -o -name '*.pth' -o -name '*.bin' -o -name '*.safetensors' -o -name '*.gguf' \) \
    -type f 2>/dev/null | wc -l | tr -d ' '
}

verify_weight_cache() {
  local sam_ok=0 ben_ok=0 vlm_ok=0
  shopt -s nullglob
  local f
  for f in \
    "${ROOT}/local_models/sam/"*.pt \
    "${ROOT}/local_models/sam/"*.pth \
    "${ROOT}/local_models/mobilesam/"*.pt \
    "${ROOT}/local_models/mobilesam/"*.pth
  do
    [[ -s "$f" ]] && sam_ok=1 && break
  done
  for f in \
    "${ROOT}/local_models/bigearthnet/"*.bin \
    "${ROOT}/local_models/bigearthnet/"*.safetensors \
    "${ROOT}/local_models/bigearthnet/"*.pt
  do
    [[ -s "$f" ]] && ben_ok=1 && break
  done
  for f in \
    "${ROOT}/local_models/vllm/"*.bin \
    "${ROOT}/local_models/vllm/"*.safetensors \
    "${ROOT}/local_models/vllm/"*.gguf \
    "${ROOT}/local_models/llava-3b/"*.bin \
    "${ROOT}/local_models/llava-3b/"*.safetensors \
    "${ROOT}/local_models/llava-3b/"*.gguf
  do
    [[ -s "$f" ]] && vlm_ok=1 && break
  done
  shopt -u nullglob

  log_info "Weight registry scan (see local_models/README.md): MobileSAM=$([[ $sam_ok -eq 1 ]] && echo present || echo missing)  BigEarthNet=$([[ $ben_ok -eq 1 ]] && echo present || echo missing)  LLaVA/VLM=$([[ $vlm_ok -eq 1 ]] && echo present || echo missing)  files=$(weight_hits)"

  if [[ "${sam_ok}" -eq 1 && "${ben_ok}" -eq 1 && "${vlm_ok}" -eq 1 ]]; then
    return 0
  fi
  return 1
}

provision_weights() {
  if [[ -x "${ROOT}/scripts/init_env.sh" ]]; then
    bash "${ROOT}/scripts/init_env.sh"
  else
    mkdir -p \
      "${ROOT}/local_models/sam" \
      "${ROOT}/local_models/bigearthnet" \
      "${ROOT}/local_models/cdvqa" \
      "${ROOT}/local_models/vllm" \
      "${ROOT}/data"
    if [[ ! -f "${ROOT}/.env" && -f "${ROOT}/.env.example" ]]; then
      cp "${ROOT}/.env.example" "${ROOT}/.env"
      log_info "Wrote .env from .env.example"
    fi
  fi

  if host_is_online; then
    log_info "Network available — prefetching LLaVA-3B, MobileSAM, and adapter weights via scripts/download_weights.sh"
    if [[ -x "${ROOT}/scripts/download_weights.sh" ]] || [[ -f "${ROOT}/scripts/download_weights.sh" ]]; then
      bash "${ROOT}/scripts/download_weights.sh" || log_warn "download_weights.sh exited non-zero; continuing with whatever is already cached."
    else
      log_warn "scripts/download_weights.sh is missing; skip Hub prefetch."
    fi
  else
    log_warn "Host appears offline / air-gapped — skipping Hub downloads and validating local_models/"
  fi

  if verify_weight_cache; then
    log_info "Required .pt / .bin weight families are present under local_models/"
  else
    log_warn "One or more checkpoint families are missing. Inference will be degraded until MobileSAM (.pt), BigEarthNet adapter (.bin), and LLaVA/VLM weights are copied into local_models/ (see local_models/README.md)."
    if ! host_is_online; then
      log_error "Offline setup cannot proceed without cached weights. Stage local_models/ from a connected machine and re-run ./run.sh --setup"
    fi
  fi
}

# ---------------------------------------------------------------------------
# Database wait + schema
# ---------------------------------------------------------------------------
tcp_open() {
  local host="$1" port="$2"
  if command -v timeout >/dev/null 2>&1; then
    timeout 1 bash -c "echo >/dev/tcp/${host}/${port}" >/dev/null 2>&1
    return $?
  fi
  bash -c "exec 3<>/dev/tcp/${host}/${port}" >/dev/null 2>&1
}

wait_for_postgres() {
  local timeout_s="${1:-180}"
  local start now
  start="$(date +%s)"
  log_info "Waiting for PostgreSQL to accept TCP connections on 127.0.0.1:5432 (no fixed sleep)..."
  while true; do
    if tcp_open 127.0.0.1 5432; then
      log_info "TCP 5432 is accepting sockets."
      break
    fi
    now="$(date +%s)"
    if (( now - start >= timeout_s )); then
      log_error "Timed out after ${timeout_s}s waiting for PostgreSQL on port 5432. Inspect: docker logs ${DB_CONTAINER}"
    fi
    sleep 1
  done

  log_info "Waiting for pg_isready inside ${DB_CONTAINER}..."
  start="$(date +%s)"
  while true; do
    if docker exec "${DB_CONTAINER}" pg_isready -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" >/dev/null 2>&1; then
      log_info "PostgreSQL is ready (${POSTGRES_DB})."
      return 0
    fi
    now="$(date +%s)"
    if (( now - start >= timeout_s )); then
      log_error "pg_isready never succeeded in ${DB_CONTAINER}."
    fi
    sleep 1
  done
}

bootstrap_postgis() {
  log_info "Activating PostGIS spatial extensions via db/init-postgis.sh inside ${DB_CONTAINER}..."
  docker exec \
    -e POSTGRES_USER="${POSTGRES_USER}" \
    -e POSTGRES_DB="${POSTGRES_DB}" \
    "${DB_CONTAINER}" \
    bash /docker-entrypoint-initdb.d/init-postgis.sh

  log_info "Applying geographic schema backend/app/database/init_db.sql (auditable_execution_traces, model_registry, trace_model_executions)..."
  docker exec -i \
    "${DB_CONTAINER}" \
    psql -v ON_ERROR_STOP=1 -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" \
    -f /docker-entrypoint-initdb.d/02-init_db.sql

  docker exec "${DB_CONTAINER}" \
    psql -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" -tAc \
    "SELECT extname || ' ' || extversion FROM pg_extension WHERE extname IN ('postgis','uuid-ossp');" \
    | sed '/^$/d' | while read -r line; do log_info "Extension: ${line}"; done
}

# ---------------------------------------------------------------------------
# Health + dashboard
# ---------------------------------------------------------------------------
HEALTH_JSON=""
SPATIAL_DRIVER="unknown"
CUDA_STATUS="Disabled"

parse_health_json() {
  local json="$1"
  if command -v python3 >/dev/null 2>&1 || command -v python >/dev/null 2>&1; then
    local py="python3"
    command -v python3 >/dev/null 2>&1 || py="python"
    local parsed
    parsed="$($py - "$json" <<'PY'
import json, sys
raw = sys.argv[1]
try:
    d = json.loads(raw)
except Exception:
    print("unknown")
    print("Disabled")
    sys.exit(0)
spatial = d.get("spatial_drivers")
if not spatial:
    gv = d.get("gdal_version") or {}
    gdal = gv.get("gdal") or "unknown"
    rio = gv.get("rasterio") or "unknown"
    spatial = f"GDAL {gdal} / Rasterio {rio}"
gpu = d.get("gpu_acceleration") or {}
cuda = gpu.get("cuda_available")
if cuda is None:
    tv = d.get("torch_version") or {}
    cuda = tv.get("cuda_available")
if cuda is None:
    cuda = d.get("gpu_available")
status = "Enabled" if cuda else "Disabled"
device = gpu.get("device") or ""
if device:
    status = f"{status} ({device})"
print(spatial)
print(status)
PY
)"
    SPATIAL_DRIVER="$(printf '%s\n' "${parsed}" | sed -n '1p')"
    CUDA_STATUS="$(printf '%s\n' "${parsed}" | sed -n '2p')"
    return
  fi
  if echo "${json}" | grep -qi 'cuda_available.: true'; then
    CUDA_STATUS="Enabled"
  else
    CUDA_STATUS="Disabled"
  fi
  SPATIAL_DRIVER="$(echo "${json}" | sed -n 's/.*"spatial_drivers"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -n1)"
  [[ -z "${SPATIAL_DRIVER}" ]] && SPATIAL_DRIVER="GDAL/Rasterio (see /health)"
}

probe_health_async() {
  local timeout_s="${1:-180}"
  local start now tmp
  start="$(date +%s)"
  tmp="$(mktemp "${TMPDIR:-/tmp}/satquery-health.XXXXXX")"
  log_info "Asynchronous health probe against http://localhost:8000/health ..."

  (
    while true; do
      if command -v curl >/dev/null 2>&1; then
        if curl -fsS --max-time 3 http://127.0.0.1:8000/health >"${tmp}.part" 2>/dev/null; then
          mv "${tmp}.part" "${tmp}"
          exit 0
        fi
      elif command -v wget >/dev/null 2>&1; then
        if wget -q -T 3 -O "${tmp}.part" http://127.0.0.1:8000/health 2>/dev/null; then
          mv "${tmp}.part" "${tmp}"
          exit 0
        fi
      else
        if command -v python3 >/dev/null 2>&1; then
          python3 - <<'PY' >"${tmp}.part" 2>/dev/null && mv "${tmp}.part" "${tmp}" && exit 0
import urllib.request
print(urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=3).read().decode())
PY
        fi
      fi
      sleep 2
    done
  ) &
  local probe_pid=$!

  while kill -0 "${probe_pid}" 2>/dev/null; do
    now="$(date +%s)"
    if (( now - start >= timeout_s )); then
      kill "${probe_pid}" 2>/dev/null || true
      wait "${probe_pid}" 2>/dev/null || true
      rm -f "${tmp}" "${tmp}.part"
      log_error "API health check timed out after ${timeout_s}s. Logs: docker logs ${BACKEND_CONTAINER}"
    fi
    sleep 1
  done
  wait "${probe_pid}" || true

  if [[ ! -s "${tmp}" ]]; then
    rm -f "${tmp}"
    log_error "Health endpoint returned an empty body."
  fi
  HEALTH_JSON="$(cat "${tmp}")"
  rm -f "${tmp}"
  parse_health_json "${HEALTH_JSON}"
  log_info "Gateway health verified."
}

print_dashboard() {
  local gdal_line cuda_line
  gdal_line="${SPATIAL_DRIVER}"
  cuda_line="${CUDA_STATUS}"
  if [[ "${GPU_ENABLED}" == "Disabled" && "${cuda_line}" != Disabled* ]]; then
    cuda_line="${cuda_line} (host NVIDIA runtime was unavailable at launch — Compose ran CPU-side)"
  fi
  if [[ "${GPU_ENABLED}" == "Enabled" && "${cuda_line}" == Disabled* ]]; then
    cuda_line="Disabled inside container (driver present on host; check NVIDIA Container Toolkit / CUDA image)"
  fi

  echo
  echo -e "${CYAN}${BOLD}╔══════════════════════════════════════════════════════════════════════╗${NC}"
  echo -e "${CYAN}${BOLD}║              SatQuery AI  —  operational dashboard                   ║${NC}"
  echo -e "${CYAN}${BOLD}╠══════════════════════════════════════════════════════════════════════╣${NC}"
  printf "${CYAN}${BOLD}║${NC}  %-68s ${CYAN}${BOLD}║${NC}\n" "Gateway API     http://localhost:8000"
  printf "${CYAN}${BOLD}║${NC}  %-68s ${CYAN}${BOLD}║${NC}\n" "API docs        http://localhost:8000/docs"
  printf "${CYAN}${BOLD}║${NC}  %-68s ${CYAN}${BOLD}║${NC}\n" "Health          http://localhost:8000/health"
  printf "${CYAN}${BOLD}║${NC}  %-68s ${CYAN}${BOLD}║${NC}\n" "Frontend Web UI http://localhost:3000"
  printf "${CYAN}${BOLD}║${NC}  %-68s ${CYAN}${BOLD}║${NC}\n" "Spatial engine  ${gdal_line}"
  printf "${CYAN}${BOLD}║${NC}  %-68s ${CYAN}${BOLD}║${NC}\n" "CUDA GPU        ${cuda_line}"
  echo -e "${CYAN}${BOLD}╚══════════════════════════════════════════════════════════════════════╝${NC}"
  echo
}

# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
ensure_env_file() {
  if [[ ! -f "${ROOT}/.env" && -f "${ROOT}/.env.example" ]]; then
    cp "${ROOT}/.env.example" "${ROOT}/.env"
    log_info "Created ${ROOT}/.env from .env.example"
  fi
}

start_db() {
  log_info "Starting PostGIS container (${DB_CONTAINER})..."
  compose up -d db
  wait_for_postgres 180
  bootstrap_postgis
}

start_app_services() {
  if [[ "${MODE}" == "setup" ]]; then
    log_info "Launching FastAPI and React containers from images built in this setup pass..."
  else
    log_info "Starting backend and frontend from existing images (no rebuild)..."
  fi
  compose up -d --no-build backend frontend
}

cmd_setup() {
  log_info "=== SatQuery AI --setup (full provisioning) ==="
  detect_compose
  check_ports
  detect_gpu
  ensure_env_file
  provision_weights
  log_info "Container ingestion: compose build for backend + frontend (requirements.txt + libgdal)..."
  compose build backend frontend
  start_db
  start_app_services
  probe_health_async 240
  print_dashboard
}

cmd_start() {
  log_info "=== SatQuery AI --start (fast localized boot) ==="
  detect_compose
  check_ports
  if [[ -f "${CPU_MARKER}" ]]; then
    GPU_ENABLED="Disabled"
    log_info "Previous setup selected CPU fallback (${CPU_MARKER})."
  else
    detect_gpu
  fi
  select_compose_file
  ensure_env_file
  if ! compose images 2>/dev/null | awk 'NR>1 {print}' | grep -qi backend; then
    log_error "No backend image is present. Run ./run.sh --setup on this machine once before --start."
  fi
  start_db
  start_app_services
  probe_health_async 180
  print_dashboard
}

case "${MODE}" in
  setup ) cmd_setup ;;
  start ) cmd_start ;;
  * ) log_error "Internal error: unknown mode ${MODE}" ;;
esac
