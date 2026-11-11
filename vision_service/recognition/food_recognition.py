"""Food recognition provider client and response validation."""

from __future__ import annotations

import base64
import json
import math
import re
from functools import lru_cache
from io import BytesIO
from pathlib import Path

from openai import OpenAI
from PIL import Image

from vision_service.settings import settings


_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "food_recognition.txt"
try:
    SYSTEM_PROMPT = _PROMPT_PATH.read_text(encoding="utf-8").strip()
except OSError as exc:
    raise RuntimeError("Unable to load food recognition prompt asset.") from exc


@lru_cache(maxsize=1)
def _get_client() -> OpenAI:
    """Create the provider client lazily using centralized timeout and credentials."""
    if not settings.openrouter_api_key:
        raise RuntimeError("OPENROUTER_API_KEY must be configured for food recognition.")
    return OpenAI(
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        timeout=settings.request_timeout_seconds,
    )


def _strip_json_fences(text: str) -> str:
    cleaned = re.sub(r"^\x60\x60\x60(?:json)?\s*", "", text.strip())
    return re.sub(r"\s*\x60\x60\x60$", "", cleaned)


def _positive_finite_number(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric_value = float(value)
    if not math.isfinite(numeric_value) or numeric_value <= 0:
        return None
    return numeric_value


def _validate_items(items) -> list[dict]:
    if not isinstance(items, list):
        return []

    valid = []
    for item in items:
        if not isinstance(item, dict):
            continue

        query = item.get("search_query")
        density = _positive_finite_number(item.get("density_g_cm3"))
        if not isinstance(query, str) or not query.strip() or density is None:
            continue

        valid.append(
            {
                "search_query": query.strip(),
                "density_g_cm3": density,
                "estimated_diameter_cm": _positive_finite_number(
                    item.get("estimated_diameter_cm")
                ),
                "estimated_thickness_cm": _positive_finite_number(
                    item.get("estimated_thickness_cm")
                ),
            }
        )
    return valid


def detect_food(image_bytes: bytes) -> list[dict]:
    with Image.open(BytesIO(image_bytes)) as image:
        image_format = image.format
    image_mime = Image.MIME.get(image_format)
    if image_mime not in {"image/jpeg", "image/png", "image/webp"}:
        raise ValueError("Food recognition received an unsupported image format.")

    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    response = _get_client().chat.completions.create(
        model=settings.openrouter_model,
        max_tokens=settings.openrouter_max_tokens,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": SYSTEM_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{image_mime};base64,{image_base64}"
                        },
                    },
                ],
            }
        ],
    )

    if not response.choices:
        return []
    content = response.choices[0].message.content
    if not content:
        return []

    try:
        parsed = json.loads(_strip_json_fences(content))
    except json.JSONDecodeError:
        return []

    return _validate_items(parsed)
