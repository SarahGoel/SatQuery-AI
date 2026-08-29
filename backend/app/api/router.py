"""Consolidated API gateway for SatQuery AI."""

from fastapi import APIRouter

from app.api.endpoints import analyze, health

api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
api_router.include_router(analyze.router, prefix="/satquery", tags=["satquery"])
