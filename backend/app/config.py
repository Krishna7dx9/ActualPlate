"""Validated backend runtime configuration.

Deployment-specific values are read from the process environment. No
credentials, hostnames, or resource limits are embedded in application code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _required_positive_integer(name: str) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None:
        raise ValueError(f"{name} must be configured.")

    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc

    if value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


@dataclass(frozen=True)
class Settings:
    """Validated runtime settings required by the backend application."""

    environment: str
    cors_allowed_origins: tuple[str, ...]
    max_upload_bytes: int
    max_request_body_bytes: int
    max_image_pixels: int

    @classmethod
    def from_environment(cls) -> "Settings":
        environment = os.environ.get("APP_ENV", "development").strip().lower()
        if not environment:
            raise ValueError("APP_ENV must not be empty.")

        raw_origins = os.environ.get("CORS_ALLOWED_ORIGINS", "")
        origins = tuple(
            origin.strip().rstrip("/")
            for origin in raw_origins.split(",")
            if origin.strip()
        )

        if environment == "production" and "*" in origins:
            raise ValueError(
                "Wildcard CORS origins are not allowed in production."
            )

        max_upload_bytes = _required_positive_integer("MAX_UPLOAD_BYTES")
        max_request_body_bytes = _required_positive_integer(
            "MAX_REQUEST_BODY_BYTES"
        )
        if max_request_body_bytes < max_upload_bytes:
            raise ValueError(
                "MAX_REQUEST_BODY_BYTES must be greater than or equal to "
                "MAX_UPLOAD_BYTES."
            )

        return cls(
            environment=environment,
            cors_allowed_origins=origins,
            max_upload_bytes=max_upload_bytes,
            max_request_body_bytes=max_request_body_bytes,
            max_image_pixels=_required_positive_integer("MAX_IMAGE_PIXELS"),
        )


settings = Settings.from_environment()
