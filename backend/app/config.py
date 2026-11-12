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
    database_url: str
    jwt_secret_key: str | None
    api_key_hash_secret: str | None
    access_token_ttl_seconds: int
    redis_url: str | None
    rate_limit_requests_per_minute: int

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


        database_url = os.environ.get(
            "DATABASE_URL",
            "sqlite:///:memory:" if environment == "test" else "sqlite:///./actualplate.db",
        ).strip()
        parsed_database_url = urlparse(database_url)
        if parsed_database_url.scheme not in {"sqlite", "postgresql", "postgresql+psycopg"}:
            raise ValueError("DATABASE_URL must use SQLite or PostgreSQL.")
        if environment == "production" and parsed_database_url.scheme == "sqlite":
            raise ValueError("DATABASE_URL must use PostgreSQL when APP_ENV=production.")

        jwt_secret_key = os.environ.get("JWT_SECRET_KEY", "").strip() or None
        api_key_hash_secret = os.environ.get("API_KEY_HASH_SECRET", "").strip() or None
        if environment == "production":
            if not jwt_secret_key or len(jwt_secret_key.encode("utf-8")) < 32:
                raise ValueError("JWT_SECRET_KEY must contain at least 32 bytes in production.")
            if not api_key_hash_secret or len(api_key_hash_secret.encode("utf-8")) < 32:
                raise ValueError("API_KEY_HASH_SECRET must contain at least 32 bytes in production.")

        try:
            access_token_ttl_seconds = int(os.environ.get("ACCESS_TOKEN_TTL_SECONDS", "900"))
        except ValueError as exc:
            raise ValueError("ACCESS_TOKEN_TTL_SECONDS must be a positive integer.") from exc
        if not 60 <= access_token_ttl_seconds <= 3600:
            raise ValueError("ACCESS_TOKEN_TTL_SECONDS must be between 60 and 3600.")


        redis_url = os.environ.get("REDIS_URL", "").strip() or None
        if redis_url:
            parsed_redis_url = urlparse(redis_url)
            if parsed_redis_url.scheme not in {"redis", "rediss"} or not parsed_redis_url.netloc:
                raise ValueError("REDIS_URL must be an absolute redis:// or rediss:// URL.")
        if environment == "production" and not redis_url:
            raise ValueError("REDIS_URL must be configured when APP_ENV=production.")

        try:
            rate_limit_requests_per_minute = int(
                os.environ.get("RATE_LIMIT_REQUESTS_PER_MINUTE", "60")
            )
        except ValueError as exc:
            raise ValueError("RATE_LIMIT_REQUESTS_PER_MINUTE must be a positive integer.") from exc
        if not 1 <= rate_limit_requests_per_minute <= 100000:
            raise ValueError("RATE_LIMIT_REQUESTS_PER_MINUTE must be between 1 and 100000.")

        return cls(
            environment=environment,
            cors_allowed_origins=origins,
            vision_inference_endpoint=raw_endpoint.rstrip("/"),
            vision_api_token=vision_api_token,
            max_upload_bytes=max_upload_bytes,
            max_request_body_bytes=max_request_body_bytes,
            max_image_pixels=_required_positive_integer("MAX_IMAGE_PIXELS"),
            database_url=database_url,
            jwt_secret_key=jwt_secret_key,
            api_key_hash_secret=api_key_hash_secret,
            access_token_ttl_seconds=access_token_ttl_seconds,
            redis_url=redis_url,
            rate_limit_requests_per_minute=rate_limit_requests_per_minute,
        )


settings = Settings.from_environment()
