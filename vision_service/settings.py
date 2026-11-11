"""Centralized, validated runtime settings for the vision service."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from urllib.parse import urlparse

from dotenv import load_dotenv

load_dotenv()


def _positive_int(name: str) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        raise ValueError(f"{name} must be configured as a positive integer.")
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _positive_float(name: str) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        raise ValueError(f"{name} must be configured as a positive number.")
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive number.") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number.")
    return value


def _threshold(name: str) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        raise ValueError(f"{name} must be configured between 0 and 1.")
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be between 0 and 1.") from exc
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1.")
    return value


def _boolean(name: str) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        raise ValueError(f"{name} must be configured as a boolean value.")
    if raw in {"true", "1", "yes"}:
        return True
    if raw in {"false", "0", "no"}:
        return False
    raise ValueError(f"{name} must be a boolean value.")


def _required_text(name: str) -> str:
    value = os.environ.get(name, "").strip()
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
        app_env = _required_text("APP_ENV").lower()
        allowed_environments = {"development", "test", "production"}
        if app_env not in allowed_environments:
            allowed = ", ".join(sorted(allowed_environments))
            raise ValueError(f"APP_ENV must be one of: {allowed}.")
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
        ).rstrip("/")
        parsed_base_url = urlparse(base_url)
        if parsed_base_url.scheme not in {"http", "https"} or not parsed_base_url.netloc:
            raise ValueError("OPENROUTER_BASE_URL must be an absolute HTTP(S) URL.")

        openrouter_key = os.environ.get("OPENROUTER_API_KEY")
        openrouter_key = openrouter_key.strip() if openrouter_key and openrouter_key.strip() else None
        if app_env == "production" and not openrouter_key:
            raise ValueError(
                "OPENROUTER_API_KEY must be configured when APP_ENV=production."
            )

        return cls(
            app_env=app_env,
            api_token=api_token,
            inference_endpoint=endpoint,
            request_timeout_seconds=_positive_float("VISION_REQUEST_TIMEOUT_SECONDS"),
            openrouter_api_key=openrouter_key,
            openrouter_base_url=base_url,
            openrouter_model=_required_text("OPENROUTER_MODEL"),
            openrouter_max_tokens=_positive_int("OPENROUTER_MAX_TOKENS"),
            grounding_dino_model_id=_required_text("GROUNDING_DINO_MODEL_ID"),
            sam2_model_id=_required_text("SAM2_MODEL_ID"),
            moge_model_id=_required_text("MOGE_MODEL_ID"),
            box_threshold=_threshold("BOX_THRESHOLD"),
            text_threshold=_threshold("TEXT_THRESHOLD"),
            show_visualizations=_boolean("SHOW_VISUALIZATIONS"),
            max_array_side=_positive_int("VISION_MAX_ARRAY_SIDE"),
        )


settings = VisionSettings.from_environment()
