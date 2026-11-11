"""Centralized, validated runtime settings for the vision service."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from urllib.parse import urlparse


def _positive_int(name: str, default: str) -> int:
    raw = os.environ.get(name, default).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _positive_float(name: str, default: str) -> float:
    raw = os.environ.get(name, default).strip()
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive number.") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number.")
    return value


def _threshold(name: str, default: str) -> float:
    raw = os.environ.get(name, default).strip()
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be between 0 and 1.") from exc
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1.")
    return value


def _boolean(name: str, default: str) -> bool:
    raw = os.environ.get(name, default).strip().lower()
    if raw in {"true", "1", "yes"}:
        return True
    if raw in {"false", "0", "no"}:
        return False
    raise ValueError(f"{name} must be a boolean value.")


def _required_text(name: str, default: str) -> str:
    value = os.environ.get(name, default).strip()
    if not value:
        raise ValueError(f"{name} must not be empty.")
    return value


@dataclass(frozen=True)
class VisionSettings:
    """Immutable application settings; secrets are never logged."""

    app_env: str
    api_token: str | None
    inference_endpoint: str | None
    request_timeout_seconds: float
    openrouter_api_key: str | None
    openrouter_base_url: str
    openrouter_model: str
    openrouter_max_tokens: int
    grounding_dino_model_id: str
    sam2_model_id: str
    moge_model_id: str
    box_threshold: float
    text_threshold: float
    show_visualizations: bool
    max_array_side: int

    @classmethod
    def from_environment(cls) -> "VisionSettings":
        app_env = _required_text("APP_ENV", "development").lower()
        api_token = os.environ.get("VISION_API_TOKEN")
        api_token = api_token.strip() if api_token and api_token.strip() else None

        endpoint = os.environ.get("VISION_INFERENCE_ENDPOINT")
        endpoint = endpoint.strip().rstrip("/") if endpoint and endpoint.strip() else None
        if endpoint is not None:
            parsed_endpoint = urlparse(endpoint)
            if parsed_endpoint.scheme not in {"http", "https"} or not parsed_endpoint.netloc:
                raise ValueError(
                    "VISION_INFERENCE_ENDPOINT must be an absolute HTTP(S) URL."
                )

        base_url = _required_text(
            "OPENROUTER_BASE_URL",
            "https://openrouter.ai/api/v1",
        ).rstrip("/")
        parsed_base_url = urlparse(base_url)
        if parsed_base_url.scheme not in {"http", "https"} or not parsed_base_url.netloc:
            raise ValueError("OPENROUTER_BASE_URL must be an absolute HTTP(S) URL.")

        openrouter_key = os.environ.get("OPENROUTER_API_KEY")
        openrouter_key = openrouter_key.strip() if openrouter_key and openrouter_key.strip() else None

        return cls(
            app_env=app_env,
            api_token=api_token,
            inference_endpoint=endpoint,
            request_timeout_seconds=_positive_float(
                "VISION_REQUEST_TIMEOUT_SECONDS",
                "300",
            ),
            openrouter_api_key=openrouter_key,
            openrouter_base_url=base_url,
            openrouter_model=_required_text(
                "OPENROUTER_MODEL",
                "google/gemma-3-27b-it",
            ),
            openrouter_max_tokens=_positive_int(
                "OPENROUTER_MAX_TOKENS",
                "800",
            ),
            grounding_dino_model_id=_required_text(
                "GROUNDING_DINO_MODEL_ID",
                "IDEA-Research/grounding-dino-base",
            ),
            sam2_model_id=_required_text(
                "SAM2_MODEL_ID",
                "facebook/sam2-hiera-large",
            ),
            moge_model_id=_required_text(
                "MOGE_MODEL_ID",
                "Ruicheng/moge-2-vitl-normal",
            ),
            box_threshold=_threshold("BOX_THRESHOLD", "0.35"),
            text_threshold=_threshold("TEXT_THRESHOLD", "0.25"),
            show_visualizations=_boolean("SHOW_VISUALIZATIONS", "false"),
            max_array_side=_positive_int("VISION_MAX_ARRAY_SIDE", "512"),
        )


settings = VisionSettings.from_environment()
