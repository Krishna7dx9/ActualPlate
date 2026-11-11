"""ASGI rate limiting that preserves streamed SSE responses."""
from __future__ import annotations

import hashlib
import logging
import math
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse

from backend.app.config import settings
from backend.app.services.rate_limiter import RateLimiter, RateLimiterUnavailable

logger = logging.getLogger(__name__)
rate_limiter = RateLimiter(
    redis_url=settings.redis_url,
    limit=settings.rate_limit_requests_per_minute,
)


async def close_rate_limiter() -> None:
    await rate_limiter.close()


class RateLimitMiddleware:
    """Apply per-IP and per-credential sliding windows without buffering response bodies."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive, send) -> None:
        if scope["type"] != "http" or scope.get("method") == "OPTIONS":
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive)
        client_ip = request.client.host if request.client else "unknown"
        identities = [f"ip:{client_ip}"]
        raw_api_key = request.headers.get("x-api-key")
        authorization = request.headers.get("authorization")
        if raw_api_key:
            identities.append("api-key:" + hashlib.sha256(raw_api_key.encode("utf-8")).hexdigest())
        elif authorization:
            identities.append("authorization:" + hashlib.sha256(authorization.encode("utf-8")).hexdigest())

        remaining = settings.rate_limit_requests_per_minute
        retry_after = 0
        try:
            for identity in identities:
                allowed, bucket_remaining, bucket_retry = await rate_limiter.check(identity)
                remaining = min(remaining, bucket_remaining)
                if not allowed:
                    retry_after = max(retry_after, bucket_retry)
                    response = JSONResponse(
                        status_code=429,
                        content={"detail": "Rate limit exceeded. Retry after the indicated interval."},
                        headers={
                            "Retry-After": str(max(1, math.ceil(retry_after))),
                            "X-RateLimit-Limit": str(settings.rate_limit_requests_per_minute),
                            "X-RateLimit-Remaining": "0",
                        },
                    )
                    await response(scope, receive, send)
                    return
        except RateLimiterUnavailable:
            logger.exception("Rate-limit backend unavailable")
            if settings.environment == "production":
                response = JSONResponse(
                    status_code=503,
                    content={"detail": "Request protection is temporarily unavailable."},
                )
                await response(scope, receive, send)
                return
            remaining = settings.rate_limit_requests_per_minute

        async def send_with_rate_headers(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.extend([
                    (b"x-ratelimit-limit", str(settings.rate_limit_requests_per_minute).encode("ascii")),
                    (b"x-ratelimit-remaining", str(max(0, remaining)).encode("ascii")),
                ])
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_with_rate_headers)
