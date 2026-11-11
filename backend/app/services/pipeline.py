import os
import tempfile
import json
import math
import logging
import cv2
import base64
import numpy as np
from fastapi import UploadFile

from backend.app.nutrition.providers.fatsecret_client import search_food
from backend.app.nutrition.nutrition_formatter import format_nutrition_response
from backend.app.nutrition.nutrition_aggregator import aggregate_nutrition
from vision_service.recognition.food_recognition import detect_food
from vision_service.detection.grounding_dino_detector import GroundingDINODetector

detector = GroundingDINODetector()
logger = logging.getLogger(__name__)


def _build_scale_map(foods):
    scale_map = {}
    for food in foods:
        label = food.get("search_query")
        if not isinstance(label, str) or not label.strip():
            continue
        dimensions = {}
        for source_key, target_key in (
            ("estimated_thickness_cm", "thickness"),
            ("estimated_diameter_cm", "diameter"),
        ):
            value = food.get(source_key)
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
                and value > 0
            ):
                dimensions[target_key] = float(value)
        scale_map[label.strip().lower()] = dimensions
    return scale_map



def _enrich_nutrition(detected_foods, merged):
    nutrition_results = []
    nutrition_errors = []
    for item in detected_foods:
        query = item.get("search_query")
        if not isinstance(query, str) or not query.strip():
            nutrition_errors.append({"code": "food_label_unavailable"})
            continue
        density = item.get("density_g_cm3")
        volume = merged.get(query.strip().lower())
        if volume is None or not isinstance(volume, (int, float)) or isinstance(volume, bool) or not math.isfinite(volume) or volume <= 0:
            nutrition_errors.append({"food": query, "code": "volume_unavailable"})
            continue
        if not isinstance(density, (int, float)) or isinstance(density, bool) or not math.isfinite(density) or density <= 0:
            nutrition_errors.append({"food": query, "code": "density_unavailable"})
            continue
        try:
            provider_result = search_food(query)
            if not isinstance(provider_result, dict) or "error" in provider_result:
                nutrition_errors.append({"food": query, "code": "provider_unavailable"})
                continue
            nutrition_results.append(format_nutrition_response(provider_result, volume * density))
        except Exception:
            logger.exception("Nutrition enrichment failed")
            nutrition_errors.append({"food": query, "code": "provider_unavailable"})

    if len(nutrition_results) == len(detected_foods) and detected_foods:
        nutrition_status = "complete"
    elif nutrition_results:
        nutrition_status = "partial"
    else:
        nutrition_status = "unavailable"
    return nutrition_results, nutrition_errors, nutrition_status

async def process_image(file: UploadFile):
    image = await file.read()
    image_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
            temp.write(image)
            image_path = temp.name

        try:
            detected_foods = detect_food(image)
        except Exception:
            logger.exception("Food recognition failed")
            return {"error": "Food recognition is temporarily unavailable."}
        if not detected_foods:
            return {"error": "Food not detected"}

        scale_map = _build_scale_map(detected_foods)
        
        try:
            detections = detector.detect(
                image_path=image_path,
                labels=[item["search_query"] for item in detected_foods],
                scale_map=scale_map,
            )
        except Exception:
            logger.exception("Vision worker inference failed")
            return {"error": "Vision inference is temporarily unavailable."}
        foods_data = detections.get("foods", [])

        merged = {}
        for item in foods_data:
            label_value = item.get("label")
            label = label_value.strip().lower() if isinstance(label_value, str) else ""
            volume = item.get("volume_cm3")
            if (
                label
                and isinstance(volume, (int, float))
                and not isinstance(volume, bool)
                and math.isfinite(volume)
                and volume > 0
            ):
                merged[label] = merged.get(label, 0.0) + float(volume)

        nutrition_results, nutrition_errors, nutrition_status = _enrich_nutrition(detected_foods, merged)

        return {
            "foods": nutrition_results, 
            "geometry": foods_data,
            "vlm_output": detected_foods,
            "total_detected": len(foods_data), 
            "total_nutrition": aggregate_nutrition(nutrition_results),
            "nutrition_status": nutrition_status,
            "nutrition_errors": nutrition_errors,
        }
    finally:
        if image_path and os.path.exists(image_path): os.remove(image_path)

async def process_image_stream(file):
    image = await file.read()
    image_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
            temp.write(image)
            image_path = temp.name

        yield f"data: {json.dumps({'step': 'gemma', 'status': 'running', 'message': 'Analyzing food image...'})}\n\n"
        try:
            detected_foods = detect_food(image)
        except Exception:
            logger.exception("Food recognition failed")
            yield f"data: {json.dumps({'step': 'error', 'status': 'error', 'message': 'Food recognition is temporarily unavailable.'})}\n\n"
            return
        yield f"data: {json.dumps({'step': 'gemma', 'status': 'done', 'message': 'Food recognition complete', 'data': detected_foods})}\n\n"

        if not detected_foods:
            empty_result = {
                "foods": [],
                "geometry": [],
                "vlm_output": [],
                "total_detected": 0,
                "total_nutrition": None,
                "nutrition_status": "not_available",
                "nutrition_errors": [],
            }
            yield f"data: {json.dumps({'step': 'complete', 'status': 'done', 'message': 'No food detected', 'final': empty_result})}\n\n"
            return

        yield f"data: {json.dumps({'step': 'dino', 'status': 'running', 'message': 'Running DINO, SAM2, and 3D Math...'})}\n\n"
        
        scale_map = _build_scale_map(detected_foods)
        
        try:
            detections = detector.detect(
                image_path=image_path,
                labels=[item["search_query"] for item in detected_foods],
                scale_map=scale_map,
            )
        except Exception:
            logger.exception("Vision worker inference failed")
            yield f"data: {json.dumps({'step': 'error', 'status': 'error', 'message': 'Vision inference is temporarily unavailable.'})}\n\n"
            return
        foods_data = detections.get("foods", [])
        
        img_array = cv2.imread(image_path)
        
        dino_img = img_array.copy()
        colors = [(0, 255, 0), (0, 165, 255), (255, 0, 0), (255, 255, 0)]
        
        for idx, item in enumerate(foods_data):
            bbox = item.get("bbox")
            if bbox:
                x1, y1, x2, y2 = map(int, bbox)
                c = colors[idx % len(colors)]
                cv2.rectangle(dino_img, (x1, y1), (x2, y2), c, 2)
                
        _, buffer = cv2.imencode('.jpg', dino_img)
        dino_b64 = base64.b64encode(buffer).decode('utf-8')
        yield f"data: {json.dumps({'step': 'dino', 'status': 'done', 'message': 'DINO localized', 'image': dino_b64})}\n\n"

        sam2_img = img_array.copy()
        for idx, item in enumerate(foods_data):
            mask_b64 = item.get("mask_b64")
            if mask_b64:
                mask_bytes = base64.b64decode(mask_b64)
                mask_np = cv2.imdecode(np.frombuffer(mask_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
                if mask_np.shape != img_array.shape[:2]:
                    mask_np = cv2.resize(mask_np, (img_array.shape[1], img_array.shape[0]), interpolation=cv2.INTER_NEAREST)
                
                c = colors[idx % len(colors)]
                colored_mask = np.zeros_like(sam2_img)
                colored_mask[mask_np > 127] = c
                sam2_img = cv2.addWeighted(sam2_img, 1.0, colored_mask, 0.35, 0)

        _, buffer = cv2.imencode('.jpg', sam2_img)
        sam2_b64 = base64.b64encode(buffer).decode('utf-8')
        yield f"data: {json.dumps({'step': 'sam2', 'status': 'done', 'message': 'SAM2 segmented', 'image': sam2_b64})}\n\n"

        # Use the REAL MoGe-2 depth map from the Vision Worker
        real_depth_b64 = detections.get("depth_b64", "")
        yield f"data: {json.dumps({'step': 'depth', 'status': 'done', 'message': 'Depth map generated', 'image': real_depth_b64})}\n\n"        
        
        yield f"data: {json.dumps({'step': 'volume', 'status': 'done', 'message': 'Volume calculated'})}\n\n"

        merged = {}
        for item in foods_data:
            label_value = item.get("label")
            label = label_value.strip().lower() if isinstance(label_value, str) else ""
            volume = item.get("volume_cm3")
            if (
                label
                and isinstance(volume, (int, float))
                and not isinstance(volume, bool)
                and math.isfinite(volume)
                and volume > 0
            ):
                merged[label] = merged.get(label, 0.0) + float(volume)

        yield f"data: {json.dumps({'step': 'nutrition', 'status': 'running', 'message': 'Fetching FatSecret...'})}\n\n"
        nutrition_results, nutrition_errors, nutrition_status = _enrich_nutrition(detected_foods, merged)

        # FIX: Remove heavy base64 masks from final payload to prevent browser OOM crash
        clean_geometry = []
        for item in foods_data:
            clean_item = item.copy()
            clean_item.pop("mask_b64", None)
            clean_geometry.append(clean_item)

        final_payload = {
            "foods": nutrition_results, 
            "geometry": clean_geometry, 
            "vlm_output": detected_foods, 
            "total_detected": len(foods_data), 
            "total_nutrition": aggregate_nutrition(nutrition_results),
            "nutrition_status": nutrition_status,
            "nutrition_errors": nutrition_errors,
        }
        yield f"data: {json.dumps({'step': 'nutrition', 'status': 'done', 'message': 'Complete', 'final': final_payload})}\n\n"
    finally:
        if image_path and os.path.exists(image_path): os.remove(image_path)