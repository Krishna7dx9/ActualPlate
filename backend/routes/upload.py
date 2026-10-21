from fastapi import APIRouter, UploadFile, File
from backend.services.pipeline import process_image_stream
from fastapi.responses import StreamingResponse

router = APIRouter()

@router.post("/upload-image-stream")
async def upload_image_stream(file: UploadFile = File(...)):
    return StreamingResponse(process_image_stream(file), media_type="text/event-stream")

@router.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        return {"error": "Only image files allowed"}
    return await process_image(file)