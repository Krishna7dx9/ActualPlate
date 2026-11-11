from __future__ import annotations

import asyncio
import json
import logging
import uuid

from fastapi import APIRouter, Depends, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

from backend.app.auth.dependencies import CurrentPrincipal, require_scope
from backend.app.config import settings
from backend.app.database import get_db
from backend.app.services.pipeline import process_image, process_image_stream
from backend.app.services.upload_validation import validate_image_upload
from backend.app.services.usage_metering import (
    DailyQuotaExceeded,
    OrganizationUnavailable,
    finalize_usage,
    reserve_scan_usage,
)

logger = logging.getLogger(__name__)
router = APIRouter()


def _validate_declared_content_type(file: UploadFile) -> JSONResponse | None:
    """Apply a cheap declared-type filter before inspecting the payload."""
    if not file.content_type or not file.content_type.startswith("image/"):
        return JSONResponse(
            status_code=400,
            content={"error": "Only image files allowed"},
        )
    return None


async def _validate_upload(file: UploadFile) -> JSONResponse | None:
    validation_error = _validate_declared_content_type(file)
    if validation_error is not None:
        return validation_error

    await validate_image_upload(
        file,
        max_bytes=settings.max_upload_bytes,
        max_pixels=settings.max_image_pixels,
    )
    return None


async def _metered_stream(file: UploadFile, usage_id: str):
    status_code = 200
    try:
        async for event in process_image_stream(file):
            text = event.decode("utf-8", errors="replace") if isinstance(event, bytes) else str(event)
            for line in text.splitlines():
                if not line.startswith("data:"):
                    continue
                try:
                    payload = json.loads(line[5:].strip())
                except json.JSONDecodeError:
                    continue
                if payload.get("step") == "error":
                    status_code = 500
            yield event
    except asyncio.CancelledError:
        status_code = 499
        raise
    except Exception:
        status_code = 500
        logger.exception("Versioned scan stream failed")
        raise
    finally:
        try:
            await run_in_threadpool(finalize_usage, usage_id, status_code)
        except Exception:
            logger.exception("Failed to finalize scan usage record")


@router.post("/v1/scans")
async def create_scan(
    file: UploadFile = File(...),
    principal: CurrentPrincipal = Depends(require_scope("scans:write")),
    db: Session = Depends(get_db),
):
    validation_error = await _validate_upload(file)
    if validation_error is not None:
        return validation_error

    request_id = uuid.uuid4().hex
    try:
        usage_id, usage_limit, usage_remaining = reserve_scan_usage(
            db,
            organization_id=principal.organization_id,
            api_key_id=principal.api_key_id,
            request_id=request_id,
            endpoint="/v1/scans",
        )
    except DailyQuotaExceeded as exc:
        return JSONResponse(
            status_code=429,
            content={
                "detail": "Daily organization request limit exceeded.",
                "used": exc.used,
                "limit": exc.limit,
                "resets_at": exc.resets_at.isoformat(),
            },
            headers={
                "Retry-After": str(exc.retry_after_seconds),
                "X-Usage-Limit": str(exc.limit),
                "X-Usage-Remaining": "0",
            },
        )
    except OrganizationUnavailable:
        return JSONResponse(
            status_code=403,
            content={"detail": "Organization access is disabled."},
        )

    return StreamingResponse(
        _metered_stream(file, usage_id),
        media_type="text/event-stream",
        headers={
            "X-Accel-Buffering": "no",
            "X-Request-ID": request_id,
            "X-Usage-Limit": str(usage_limit),
            "X-Usage-Remaining": str(usage_remaining),
        },
    )


@router.post("/upload-image-stream")
async def upload_image_stream(file: UploadFile = File(...)):
    validation_error = await _validate_upload(file)
    if validation_error is not None:
        return validation_error

    return StreamingResponse(
        process_image_stream(file),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no"},
    )


@router.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    validation_error = await _validate_upload(file)
    if validation_error is not None:
        return validation_error

    return await process_image(file)
