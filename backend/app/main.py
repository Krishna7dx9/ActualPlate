import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.routes.nutrition import router as nutrition_router
from backend.app.api.routes.upload import router as upload_router
from backend.app.config import settings
from backend.app.middleware.request_size_limit import RequestSizeLimitMiddleware


logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    RequestSizeLimitMiddleware,
    max_bytes=settings.max_request_body_bytes,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_allowed_origins),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-API-Token"],
)

app.include_router(nutrition_router)
app.include_router(upload_router)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Return a stable public error while retaining diagnostics in server logs."""
    logger.exception(
        "Unhandled application exception",
        extra={
            "request_method": request.method,
            "request_path": request.url.path,
        },
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error"},
    )
