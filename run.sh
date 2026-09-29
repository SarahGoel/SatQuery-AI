#!/usr/bin/env bash

# ==============================================================================
# SatQuery AI — Autonomous Earth Observation Intelligence Console
# ISRO Space Applications Centre (SAC) / SIH Evaluation Launcher
# ==============================================================================

set -eo pipefail

# Visual formatting
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m' # No Color

echo -e "${CYAN}${BOLD}"
echo "======================================================================"
echo "      ____        __  ____                            ___     ____    "
echo "     / __/____ _ / /_/ __ \__  __ ___   _____ __  __ /   |   /  _/    "
echo "    _\ \ / _ \`// __// / / // / / // _ \ / ___// / / // /| |   / /      "
echo "   /___/ \_,_/ \__/ \___\_\\__,_// .__//_/    \__, // ___ | _/ /       "
echo "                                /_/          /____//_/  |_|/___/       "
echo "   ISRO SAC Earth Observation Intelligence Platform (Air-Gap Ready)   "
echo "======================================================================"
echo -e "${NC}"

# 1. System Prerequisite Checks
echo -e "${BLUE}[+] Step 1/6: Verifying system host prerequisites...${NC}"
if ! command -v docker &> /dev/null; then
    echo -e "${RED}[ERROR] Docker engine is not installed or not in PATH.${NC}"
    echo "Please install Docker Desktop or Docker Engine to evaluate SatQuery AI."
    exit 1
fi

if docker compose version &> /dev/null; then
    DOCKER_COMPOSE_CMD="docker compose"
elif command -v docker-compose &> /dev/null; then
    DOCKER_COMPOSE_CMD="docker-compose"
else
    echo -e "${RED}[ERROR] Docker Compose plugin is not detected.${NC}"
    exit 1
fi

if ! docker info &> /dev/null; then
    echo -e "${RED}[ERROR] Docker daemon is not running.${NC}"
    exit 1
fi
echo -e "${GREEN}[✓] Docker Engine and Compose verified.${NC}"

# 2. Automated Environment Configuration
echo -e "\n${BLUE}[+] Step 2/6: Configuring environment variables...${NC}"
if [ ! -f "backend/.env" ]; then
    if [ -f "backend/.env.example" ]; then
        echo -e "${YELLOW}[!] 'backend/.env' not found. Creating from 'backend/.env.example'...${NC}"
        cp backend/.env.example backend/.env
    else
        echo -e "${YELLOW}[!] Creating default air-gapped 'backend/.env'...${NC}"
        cat <<EOF> backend/.env
VLM_PROVIDER=ollama
GEMINI_API_KEY=
OLLAMA_BASE_URL=http://satquery_ollama:11434
POSTGRES_DB=satquery_db
POSTGRES_USER=satquery_user
POSTGRES_PASSWORD=satquery_secret
POSTGRES_HOST=satquery_postgres
POSTGRES_PORT=5432
EOF
    fi
fi

if grep -q "^VLM_PROVIDER=gemini" backend/.env 2>/dev/null && grep -qv "^GEMINI_API_KEY=\s*$" backend/.env 2>/dev/null; then
    echo -e "${GREEN}[✓] Engine Mode: HYBRID CLOUD (Google Gemini Active — ~2s Inference)${NC}"
else
    echo -e "${YELLOW}[!] Engine Mode: AIR-GAPPED LOCAL (Ollama Local Weights / Offline Mode)${NC}"
fi

# 3. Model Weight Directory
echo -e "\n${BLUE}[+] Step 3/6: Preparing local model weight directories...${NC}"
mkdir -p local_models/sam local_models/vlm_lora local_models/bigearthnet
chmod -R 775 local_models 2>/dev/null || true
echo -e "${GREEN}[✓] Local model cache mount directories initialized.${NC}"

# 4. Container Launch
echo -e "\n${BLUE}[+] Step 4/6: Building and bringing up stack containers...${NC}"
$DOCKER_COMPOSE_CMD up -d --build

# 5. Health Check
echo -e "\n${BLUE}[+] Step 5/6: Polling system health probes...${NC}"
BACKEND_HEALTH_URL="http://localhost:8000/api/v1/health"
MAX_ATTEMPTS=45
ATTEMPT=1
BACKEND_READY=false

echo -n "Waiting for backend services to stabilize"
while [ $ATTEMPT -le $MAX_ATTEMPTS ]; do
    if curl -s -f "$BACKEND_HEALTH_URL" &> /dev/null || curl -s -f "http://localhost:8000/docs" &> /dev/null; then
        BACKEND_READY=true
        break
    fi
    echo -n "."
    sleep 2
    ATTEMPT=$((ATTEMPT + 1))
done
echo ""

if [ "$BACKEND_READY" = false ]; then
    echo -e "${RED}[ERROR] Backend failed to report healthy status.${NC}"
    $DOCKER_COMPOSE_CMD logs --tail=40 backend
    exit 1
fi
echo -e "${GREEN}[✓] Backend API gateway and spatial services are online.${NC}"

# 6. Success Output
echo -e "\n${GREEN}${BOLD}======================================================================${NC}"
echo -e "${GREEN}${BOLD}   SatQuery AI Platform Successfully Deployed & Ready for Evaluation  ${NC}"
echo -e "${GREEN}${BOLD}======================================================================${NC}"
echo -e "  ${BOLD}Interactive Web Console:${NC}   http://localhost:5173  (or http://localhost:3000)"
echo -e "  ${BOLD}FastAPI Interactive Docs:${NC}  http://localhost:8000/docs"
echo -e "  ${BOLD}Air-Gap Verification:${NC}     http://localhost:8000/api/v1/health/air-gap"
echo -e "======================================================================\n"
