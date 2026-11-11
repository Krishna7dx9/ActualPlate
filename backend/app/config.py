"""Validated backend runtime configuration.

Deployment-specific values are read from the process environment. No
credentials, hostnames, or resource limits are embedded in application code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import urlparse

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
    vision_inference_endpoint: str
    vision_api_token: str | None
    max_upload_bytes: int
    max_request_body_bytes: int
    max_image_pixels: int

    @classmethod
    def from_environment(cls) -> "Settings":
        environment = os.environ.get("APP_ENV", "development").strip().lower()
        allowed_environments = {"development", "test", "production"}
        if environment not in allowed_environments:
            allowed = ", ".join(sorted(allowed_environments))
            raise ValueError(f"APP_ENV must be one of: {allowed}.")

        raw_origins = os.environ.get("CORS_ALLOWED_ORIGINS", "")
        origins = tuple(
            origin.strip().rstrip("/")
            for origin in raw_origins.split(",")
            if origin.strip()
        )

        raw_endpoint = os.environ.get("VISION_INFERENCE_ENDPOINT", "").strip()
        parsed_endpoint = urlparse(raw_endpoint)
        if (
            parsed_endpoint.scheme not in {"http", "https"}
            or not parsed_endpoint.netloc
        ):
            raise ValueError(
                "VISION_INFERENCE_ENDPOINT must be configured as an absolute HTTP(S) URL."
            )

        vision_api_token = os.environ.get("VISION_API_TOKEN")
        vision_api_token = (
            vision_api_token.strip()
            if vision_api_token and vision_api_token.strip()
            else None
        )
        if environment == "production" and not vision_api_token:
            raise ValueError(
                "VISION_API_TOKEN must be configured when APP_ENV=production."
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
            vision_inference_endpoint=raw_endpoint.rstrip("/"),
            vision_api_token=vision_api_token,
            max_upload_bytes=max_upload_bytes,
            max_request_body_bytes=max_request_body_bytes,
            max_image_pixels=_required_positive_integer("MAX_IMAGE_PIXELS"),
        )


settings = Settings.from_environment()
