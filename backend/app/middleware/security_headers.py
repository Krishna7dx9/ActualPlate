"""Baseline security headers for API responses without buffering streams."""
from __future__ import annotations

from typing import Any

from backend.app.config import settings


class SecurityHeadersMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_security_headers(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                existing = {key.lower() for key, _value in headers}
                defaults = [
                    (b"x-content-type-options", b"nosniff"),
                    (b"x-frame-options", b"DENY"),
                    (b"referrer-policy", b"strict-origin-when-cross-origin"),
                    (b"permissions-policy", b"camera=(), microphone=(), geolocation=()"),
                    (b"cross-origin-opener-policy", b"same-origin"),
                ]
                if settings.environment == "production":
                    defaults.append((
                        b"content-security-policy",
                        b"default-src 'none'; frame-ancestors 'none'; base-uri 'none'",
                    ))
                    if scope.get("scheme") == "https":
                        defaults.append((
                            b"strict-transport-security",
                            b"max-age=31536000",
                        ))
                for key, value in defaults:
                    if key not in existing:
                        headers.append((key, value))
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_with_security_headers)
