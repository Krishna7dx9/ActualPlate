from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from backend.app.config import settings
from backend.app.services.pipeline import process_image, process_image_stream
from backend.app.services.upload_validation import validate_image_upload


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
