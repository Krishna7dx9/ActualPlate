import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.routes.nutrition import router as nutrition_router
from backend.app.api.routes.upload import router as upload_router
from backend.app.auth.routes import router as auth_router
from backend.app.auth.api_keys import router as api_keys_router
from backend.app.config import settings
from backend.app.middleware.request_size_limit import RequestSizeLimitMiddleware
from backend.app.middleware.rate_limit import RateLimitMiddleware, close_rate_limiter
from backend.app.api.routes.usage import router as usage_router
from backend.app.api.routes.system import router as system_router
from backend.app.middleware.security_headers import SecurityHeadersMiddleware


logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await close_rate_limiter()


app = FastAPI(
    lifespan=lifespan,
    docs_url=None if settings.environment == "production" else "/docs",
    redoc_url=None if settings.environment == "production" else "/redoc",
    openapi_url=None if settings.environment == "production" else "/openapi.json",
)

app.add_middleware(RateLimitMiddleware)
app.add_middleware(
    RequestSizeLimitMiddleware,
    max_bytes=settings.max_request_body_bytes,
)

app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_allowed_origins),
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-API-Token", "X-API-Key"],
)

app.include_router(nutrition_router)
app.include_router(upload_router)
app.include_router(auth_router)
app.include_router(api_keys_router)
app.include_router(usage_router)
app.include_router(system_router)


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
