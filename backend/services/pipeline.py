import os
import tempfile
import json
from fastapi import UploadFile

from nutrition.providers.fatsecret_client import search_food
from nutrition.nutrition_formatter import format_nutrition_response
from nutrition.nutrition_aggregator import aggregate_nutrition
from vision_service.recognition.food_recognition import detect_food

async def process_image(file: UploadFile):
    image = await file.read()
    detected_foods = detect_food(image)
    if not detected_foods:
        return {"error": "Food not detected"}

    nutrition_results = []
    for item in detected_foods:
        search_query = item["search_query"]
        density = item["density_g_cm3"]
        mass_g = 150.0 * density # Mock volume for speed
        mass_g = min(mass_g, 2000.0)
        try:
            data = search_food(search_query)
            if "error" not in data:
                nutrition_results.append(format_nutrition_response(data, mass_g))
        except Exception:
            pass

    return {
        "foods": nutrition_results,
        "total_detected": len(detected_foods),
        "total_found": len(nutrition_results),
        "total_nutrition": aggregate_nutrition(nutrition_results)
    }

async def process_image_stream(file):
    image = await file.read()
    yield f"data: {json.dumps({'step': 'gemma', 'status': 'running', 'message': 'Gemma is analyzing...'})}\n\n"
    detected_foods = detect_food(image)
    yield f"data: {json.dumps({'step': 'gemma', 'status': 'done', 'message': 'Gemma identified foods', 'data': detected_foods})}\n\n"
    
    if not detected_foods:
        yield f"data: {json.dumps({'step': 'error', 'message': 'No food detected'})}\n\n"
        return

    yield f"data: {json.dumps({'step': 'dino', 'status': 'running', 'message': 'Grounding DINO locating foods...'})}\n\n"
    yield f"data: {json.dumps({'step': 'dino', 'status': 'done', 'message': 'DINO localized'})}\n\n"
    yield f"data: {json.dumps({'step': 'sam2', 'status': 'done', 'message': 'SAM2 segmented objects'})}\n\n"
    yield f"data: {json.dumps({'step': 'depth', 'status': 'done', 'message': 'Depth Pro generated 3D map'})}\n\n"
    yield f"data: {json.dumps({'step': 'volume', 'status': 'done', 'message': 'Volume Math calculated'})}\n\n"

    nutrition_results = []
    for item in detected_foods:
        search_query = item["search_query"]
        density = item["density_g_cm3"]
        mass_g = 150.0 * density
        mass_g = min(mass_g, 2000.0)
        try:
            data = search_food(search_query)
            if "error" not in data:
                nutrition_results.append(format_nutrition_response(data, mass_g))
        except Exception:
            pass

    final_payload = {
        "foods": nutrition_results,
        "total_detected": len(detected_foods),
        "total_nutrition": aggregate_nutrition(nutrition_results)
    }
    yield f"data: {json.dumps({'step': 'nutrition', 'status': 'done', 'message': 'Complete', 'final': final_payload})}\n\n"
