# ==============================================================================
# SatQuery AI — Native PowerShell Automation Script (run.ps1)
# ==============================================================================
param (
    [switch]$NoLogs,
    [switch]$Help
)

if ($Help) {
    Write-Host @"
SatQuery AI: Windows PowerShell Bring-Up Script

Usage:
  .\run.ps1            Complete startup: containers, health check, log stream
  .\run.ps1 -NoLogs    Startup without streaming logs
  .\run.ps1 -Help      Show this documentation

Services Launched:
  - Database:         PostgreSQL 16 + PostGIS 3.4 (localhost:5432)
  - Vision LLM:       Ollama serving llava:latest (localhost:11434)
  - Backend Gateway:  FastAPI + PyTorch + GDAL (localhost:8000)
  - Frontend Web UI:  React + Vite + Tailwind (localhost:3000)
  - Swagger Docs:     http://localhost:8000/docs
"@
    exit 0
}

$ErrorActionPreference = "Stop"
Write-Host "==> Step 1/4: Launching SatQuery AI stack with Docker Compose..." -ForegroundColor Cyan
docker compose up -d --build

Write-Host "==> Step 2/4: Waiting for backend to become healthy on http://localhost:8000/health..." -ForegroundColor Cyan
$maxAttempts = 30
$attempt = 0
$healthy = $false

while ($attempt -lt $maxAttempts -and -not $healthy) {
    Start-Sleep -Seconds 2
    $attempt++
    try {
        $res = Invoke-WebRequest -Uri "http://localhost:8000/health" -UseBasicParsing -TimeoutSec 3 -ErrorAction SilentlyContinue
        if ($res.StatusCode -eq 200) {
            $healthy = $true
        }
    } catch {
        Write-Host -NoNewline "."
    }
}
Write-Host ""

if (-not $healthy) {
    Write-Host "[ERROR] Backend did not respond with 200 OK within 60s." -ForegroundColor Red
    docker compose logs --tail 50 backend
    exit 1
}

Write-Host "[SUCCESS] Backend API is fully operational and healthy (HTTP 200 OK)." -ForegroundColor Green

Write-Host @"

================================================================================
                         SATQUERY AI SYSTEM ONLINE                             
================================================================================
  Frontend Web UI     : http://localhost:3000
  Backend API Gateway : http://localhost:8000
  Swagger API Docs    : http://localhost:8000/docs
  Health Check        : http://localhost:8000/health
  Ollama VLM Service  : http://localhost:11434
================================================================================

"@ -ForegroundColor Green

if (-not $NoLogs) {
    Write-Host "Streaming live container logs (Backend & Frontend). Press Ctrl+C to exit log stream..." -ForegroundColor Yellow
    docker compose logs -f backend frontend
}
