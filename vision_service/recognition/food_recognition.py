SYSTEM_PROMPT = """
You are an expert food scientist analyzing a photo.
Return ONLY a JSON array of objects. Do not include markdown.

Extract exactly four fields:
1. "search_query": The simplest name of the food.
2. "density_g_cm3": Estimated density.
3. "estimated_diameter_cm": The maximum physical width/diameter of the food in cm.
4. "estimated_thickness_cm": The physical height/thickness of the food in cm.

Example Output:
[{"search_query": "bagel", "density_g_cm3": 0.8, "estimated_diameter_cm": 10.0, "estimated_thickness_cm": 4.0}]
"""

import base64
import json
import math
import re
from functools import lru_cache
from io import BytesIO

from openai import OpenAI
from PIL import Image

from vision_service.settings import settings


@lru_cache(maxsize=1)
def _get_client() -> OpenAI:
    """Create the provider client lazily so missing secrets fail at use time."""
    if not settings.openrouter_api_key:
        raise RuntimeError("OPENROUTER_API_KEY must be configured for food recognition.")
    return OpenAI(
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
    )

def _strip_json_fences(text: str) -> str:
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned

def _validate_items(items) -> list:
    if not isinstance(items, list):
        return []
    valid = []
    for item in items:
        if not isinstance(item, dict): continue
        query = item.get("search_query")
        density = item.get("density_g_cm3")
        if not isinstance(query, str) or not query.strip(): continue
        if not isinstance(density, (int, float)) or density <= 0: continue
        
        # Extract new fields
        diam = item.get("estimated_diameter_cm", 10.0)
        thick = item.get("estimated_thickness_cm", 3.0)
        
        valid.append({
            "search_query": query.strip(), 
            "density_g_cm3": float(density),
            "estimated_diameter_cm": float(diam),
            "estimated_thickness_cm": float(thick)
        })
    return valid

def detect_food(image_bytes: bytes) -> list:
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
                    {"type": "image_url", "image_url": {"url": f"data:{image_mime};base64,{image_base64}"}},
                ],
            }
        ],
    )
    content = response.choices[0].message.content
    if not content: return []
    
    cleaned = _strip_json_fences(content)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return []
        
    return _validate_items(parsed)