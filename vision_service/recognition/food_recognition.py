SYSTEM_PROMPT = """
You are an expert food scientist analyzing a photo.
Return ONLY a JSON array of objects. Do not include markdown or explanations.

For each food item, extract exactly three fields:
1. "search_query": The simplest name of the food (e.g., "apple", "bagel").
2. "density_g_cm3": Estimated density in g/cm3.
3. "estimated_height_cm": Your best visual guess of the maximum physical height of the food in centimeters (e.g., 6.5).

Example Output:
[
  {"search_query": "apple", "density_g_cm3": 0.85, "estimated_height_cm": 6.5}
]
"""

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
        
        # Extract height, default to 0.0 if missing
        height = item.get("estimated_height_cm", 0.0)
        
        valid.append({
            "search_query": query.strip(), 
            "density_g_cm3": float(density),
            "estimated_height_cm": float(height)
        })
    return valid

def detect_food(image_bytes: bytes) -> list:
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    
    # Note: response_format was removed so it returns a list, not a dict.
    response = client.chat.completions.create(
        model="google/gemma-3-27b-it",
        max_tokens=800,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": SYSTEM_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
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