"""System health, readiness, and privileged GPU warmup endpoints."""
from __future__ import annotations

import logging
from urllib.parse import urlsplit, urlunsplit

import httpx
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from backend.app.auth.dependencies import CurrentPrincipal, require_roles, require_scope
from backend.app.config import settings
from backend.app.database import get_db
from backend.app.middleware.rate_limit import rate_limiter
from backend.app.services.rate_limiter import RateLimiterUnavailable

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/system", tags=["system"])


def _worker_url(path: str) -> str:
    parsed = urlsplit(settings.vision_inference_endpoint.rstrip("/"))
    base_path = parsed.path.rstrip("/")
    return urlunsplit((parsed.scheme, parsed.netloc, f"{base_path}{path}", "", ""))


def _database_status(db: Session) -> str:
    try:
        db.execute(text("SELECT 1")).scalar_one()
        return "ok"
    except SQLAlchemyError:
        db.rollback()
        return "unavailable"


async def _worker_health() -> dict[str, object]:
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(3.0, connect=1.0)) as client:
            response = await client.get(_worker_url("/health"))
            response.raise_for_status()
            payload = response.json()
        if not isinstance(payload, dict):
            return {"status": "unavailable", "models_loaded": False}
        loaded = payload.get("models_loaded") is True
        return {
            "status": "ready" if loaded else "warming",
            "models_loaded": loaded,
        }
    except (httpx.HTTPError, ValueError, TypeError, AttributeError):
        return {"status": "unavailable", "models_loaded": False}


async def _health_components(db: Session) -> dict[str, object]:
    database_status = await run_in_threadpool(_database_status, db)
    try:
        rate_status = await rate_limiter.health()
    except RateLimiterUnavailable:
        rate_status = {"status": "unavailable", "mode": "redis"}
    worker_status = await _worker_health()
    return {
        "database": {"status": database_status},
        "rate_limit_store": rate_status,
        "vision_worker": worker_status,
    }


@router.get("/health")
async def health(db: Session = Depends(get_db)):
    components = await _health_components(db)
    status = "ok" if components["database"]["status"] == "ok" else "degraded"
    return {"status": status, "components": components}


@router.get("/ready")
async def readiness(db: Session = Depends(get_db)):
    components = await _health_components(db)
    rate_status = components["rate_limit_store"]["status"]
    worker_status = components["vision_worker"]["status"]
    ready = (
        components["database"]["status"] == "ok"
        and rate_status in {"ok", "local"}
        and worker_status == "ready"
    )
    body = {"status": "ready" if ready else "not_ready", "components": components}
    return body if ready else JSONResponse(status_code=503, content=body)


async def _perform_worker_warmup() -> dict[str, object]:
    if not settings.vision_api_token:
        raise RuntimeError("Vision worker authentication is not configured.")
    timeout = httpx.Timeout(settings.vision_warmup_timeout_seconds, connect=5.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            _worker_url("/warmup"),
            headers={"X-API-Token": settings.vision_api_token},
        )
        response.raise_for_status()
        payload = response.json()
    if not isinstance(payload, dict) or payload.get("models_loaded") is not True:
        raise RuntimeError("Vision worker did not report ready models.")
    return {"status": "ready", "models_loaded": True}


@router.post("/warmup")
async def warmup(
    _principal: CurrentPrincipal = Depends(require_roles("owner", "admin")),
    _scope: CurrentPrincipal = Depends(require_scope("system:warmup")),
):
    try:
        return await _perform_worker_warmup()
    except (httpx.HTTPError, RuntimeError, ValueError, TypeError, AttributeError):
        logger.exception("Privileged vision worker warmup failed")
        raise HTTPException(
            status_code=503,
            detail="Vision worker warmup did not complete successfully.",
        )
