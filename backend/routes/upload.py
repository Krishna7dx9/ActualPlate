from fastapi import APIRouter, UploadFile, File, Response
from fastapi.responses import StreamingResponse
from backend.services.pipeline import process_image, process_image_stream

router = APIRouter()

@router.options("/upload-image")
async def options_upload():
    return Response(headers={
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "*",
    })

@router.options("/upload-image-stream")
async def options_stream():
    return Response(headers={
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "*",
    })

@router.post("/upload-image-stream")
async def upload_image_stream(file: UploadFile = File(...)):
    # Explicit CORS headers + X-Accel-Buffering to prevent Cloudflare from buffering the stream
    return StreamingResponse(
        process_image_stream(file),
        media_type="text/event-stream",
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "*",
            "X-Accel-Buffering": "no"
        }
    )

@router.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    return await process_image(file)