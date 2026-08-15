from fastapi import APIRouter, UploadFile, File
from backend.services.pipeline import process_image

router = APIRouter()

@router.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        return {"error": "Only image files allowed"}
    return await process_image(file)