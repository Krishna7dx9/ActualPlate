"""Validated backend runtime configuration.

Configuration is read from the process environment once at application
initialization. Keep deployment-specific origins out of source control.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """Runtime settings required by the backend application."""

    environment: str
    cors_allowed_origins: tuple[str, ...]

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

        return cls(
            environment=environment,
            cors_allowed_origins=origins,
        )


settings = Settings.from_environment()
