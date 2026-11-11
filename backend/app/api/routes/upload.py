from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from backend.app.services.pipeline import process_image, process_image_stream


router = APIRouter()


def _validate_declared_content_type(file: UploadFile) -> JSONResponse | None:
    """Reject uploads that do not declare an image media type.

    This is only an early filter. Actual image decoding and bounded body
    validation must also be performed before inference.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        return JSONResponse(
            status_code=400,
            content={"error": "Only image files allowed"},
        )
    return None


@router.post("/upload-image-stream")
async def upload_image_stream(file: UploadFile = File(...)):
    validation_error = _validate_declared_content_type(file)
    if validation_error is not None:
        return validation_error

    return StreamingResponse(
        process_image_stream(file),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no"},
    )


@router.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    validation_error = _validate_declared_content_type(file)
    if validation_error is not None:
        return validation_error

    return await process_image(file)
