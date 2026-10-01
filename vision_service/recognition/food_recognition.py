from openai import OpenAI
import base64
import json
import re
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)


SYSTEM_PROMPT = """
You are an expert food scientist and clinical nutritionist. Your task is to analyze the provided image of a meal and extract precise data for a physical volume-to-mass calculation and a nutrition database search.

You MUST return your analysis strictly as a JSON array of objects. Do not include markdown blocks, explanations, or conversational text. Return ONLY the JSON.

For each distinct food item on the plate, extract exactly two fields:

1. "search_query": A highly specific, concise string combining the cooking method, preparation state, and exact food name, optimized for a nutrition database API.
   - Good: "grilled skinless chicken breast", "steamed white rice", "creamy tomato soup", "raw spinach"
   - Bad (Too vague): "chicken", "soup", "a bowl of rice"

2. "density_g_cm3": The estimated effective physical density of this specific food in grams per cubic centimeter (g/cm³).
   - Do not use a static lookup; dynamically estimate based on the state of matter visible in the image:
   - Fluids and Solid Masses (e.g., soups, sauces, meats, cheese): Use true density. Most water-based liquids or dense proteins sit closely between 1.0 and 1.05 g/cm³.
   - Particulates and Aggregates (e.g., rice, salads, popcorn, cereals): Use bulk density, accounting for the macroscopic air gaps between pieces. Cooked grains pack tighter (~0.7 to 0.85 g/cm³) than raw leafy greens (~0.1 to 0.2 g/cm³).
   - Aerated/Porous Solids (e.g., breads, pastries, sponge cakes): Account for internal microscopic air pockets (~0.3 to 0.5 g/cm³).

Example Expected Output:
[
  {"search_query": "creamy tomato soup", "density_g_cm3": 1.02},
  {"search_query": "grilled skinless chicken breast", "density_g_cm3": 1.04},
  {"search_query": "steamed jasmine rice", "density_g_cm3": 0.78},
  {"search_query": "fresh mixed green salad", "density_g_cm3": 0.15}
]
"""


def _strip_json_fences(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` wrappers if present."""
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned


def _validate_items(items) -> list:
    """Ensure each item has the required fields with correct types."""
    if not isinstance(items, list):
        return []
    valid = []
    for item in items:
        if not isinstance(item, dict):
            continue
        query = item.get("search_query")
        density = item.get("density_g_cm3")
        if not isinstance(query, str) or not query.strip():
            continue
        if not isinstance(density, (int, float)) or density <= 0:
            continue
        valid.append({
            "search_query": query.strip(),
            "density_g_cm3": float(density),
        })
    return valid


def detect_food(image_bytes: bytes) -> list:
    """Send image to Gemma. Return list of {search_query, density_g_cm3}."""

    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    response = client.chat.completions.create(
        model="google/gemma-3-27b-it",
        max_tokens=800,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": SYSTEM_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        },
                    },
                ],
            }
        ],
    )

    content = response.choices[0].message.content
    if not content:
        return []

    cleaned = _strip_json_fences(content)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        print(f"[detect_food] JSON parse failed: {exc}")
        print(f"[detect_food] Raw content: {content!r}")
        return []

    return _validate_items(parsed)