import os
import tempfile
import json
import cv2
import base64
import numpy as np
from fastapi import UploadFile

from nutrition.providers.fatsecret_client import search_food
from nutrition.nutrition_formatter import format_nutrition_response
from nutrition.nutrition_aggregator import aggregate_nutrition
from vision_service.recognition.food_recognition import detect_food
from vision_service.detection.grounding_dino_detector import GroundingDINODetector

detector = GroundingDINODetector()

async def process_image(file: UploadFile):
    image = await file.read()
    image_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
            temp.write(image)
            image_path = temp.name

        detected_foods = detect_food(image)
        if not detected_foods: return {"error": "Food not detected"}

        detections = detector.detect(image_path=image_path, labels=[item["search_query"] for item in detected_foods])
        foods_data = detections.get("foods", [])

        merged = {}
        for item in foods_data:
            label = item.get("label", "").strip().lower()
            vol = item.get("volume_cm3")
            if label and vol: merged[label] = merged.get(label, 0) + vol

        nutrition_results = []
        for item in detected_foods:
            q = item["search_query"]
            d = item["density_g_cm3"]
            match = merged.get(q.strip().lower())
            if match:
                mass = match * d  # REAL MATH, NO CLAMP
                try:
                    data = search_food(q)
                    if "error" not in data:
                        nutrition_results.append(format_nutrition_response(data, mass))
                except Exception: pass

        return {"foods": nutrition_results, "total_detected": len(foods_data), "total_nutrition": aggregate_nutrition(nutrition_results)}
    finally:
        if image_path and os.path.exists(image_path): os.remove(image_path)

async def process_image_stream(file):
    image = await file.read()
    image_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
            temp.write(image)
            image_path = temp.name

        yield f"data: {json.dumps({'step': 'gemma', 'status': 'running', 'message': 'Gemma is analyzing...'})}\n\n"
        detected_foods = detect_food(image)
        yield f"data: {json.dumps({'step': 'gemma', 'status': 'done', 'message': 'Gemma identified foods', 'data': detected_foods})}\n\n"
        
        if not detected_foods: return

        yield f"data: {json.dumps({'step': 'dino', 'status': 'running', 'message': 'Running DINO, SAM2, and Depth...'})}\n\n"
        detections = detector.detect(image_path=image_path, labels=[item["search_query"] for item in detected_foods])
        foods_data = detections.get("foods", [])
        
        img_array = cv2.imread(image_path)
        
        # DINO
        dino_img = img_array.copy()
        for item in foods_data:
            bbox = item.get("bbox")
            if bbox:
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(dino_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        _, buffer = cv2.imencode('.jpg', dino_img)
        dino_b64 = base64.b64encode(buffer).decode('utf-8')
        yield f"data: {json.dumps({'step': 'dino', 'status': 'done', 'message': 'DINO localized', 'image': dino_b64})}\n\n"

        # SAM2
        sam2_img = img_array.copy()
        for item in foods_data:
            bbox = item.get("bbox")
            if bbox:
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(sam2_img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        _, buffer = cv2.imencode('.jpg', sam2_img)
        sam2_b64 = base64.b64encode(buffer).decode('utf-8')
        yield f"data: {json.dumps({'step': 'sam2', 'status': 'done', 'message': 'SAM2 segmented', 'image': sam2_b64})}\n\n"

        # Depth
        depth_img = cv2.applyColorMap(cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_JET)
        _, buffer = cv2.imencode('.jpg', depth_img)
        depth_b64 = base64.b64encode(buffer).decode('utf-8')
        yield f"data: {json.dumps({'step': 'depth', 'status': 'done', 'message': 'Depth map generated', 'image': depth_b64})}\n\n"

        # Volume
        yield f"data: {json.dumps({'step': 'volume', 'status': 'done', 'message': 'Volume calculated'})}\n\n"
        merged = {}
        for item in foods_data:
            label = item.get("label", "").strip().lower()
            vol = item.get("volume_cm3")
            if label and vol: merged[label] = merged.get(label, 0) + vol

        # Nutrition
        yield f"data: {json.dumps({'step': 'nutrition', 'status': 'running', 'message': 'Fetching FatSecret...'})}\n\n"
        nutrition_results = []
        for item in detected_foods:
            q = item["search_query"]
            d = item["density_g_cm3"]
            match = merged.get(q.strip().lower())
            if match:
                mass = match * d  # REAL MATH
                try:
                    data = search_food(q)
                    if "error" not in data:
                        nutrition_results.append(format_nutrition_response(data, mass))
                except Exception: pass

        final_payload = {"foods": nutrition_results, "total_detected": len(foods_data), "total_nutrition": aggregate_nutrition(nutrition_results)}
        yield f"data: {json.dumps({'step': 'nutrition', 'status': 'done', 'message': 'Complete', 'final': final_payload})}\n\n"
    finally:
        if image_path and os.path.exists(image_path): os.remove(image_path)
