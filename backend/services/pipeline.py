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
        
        # DINO Boxes (Enterprise Clean)
        dino_img = img_array.copy()
        colors = [(0, 255, 0), (0, 165, 255), (255, 0, 0), (255, 255, 0)] # Green, Orange, Blue, Yellow
        
        for idx, item in enumerate(foods_data):
            bbox = item.get("bbox")
            if bbox:
                x1, y1, x2, y2 = map(int, bbox)
                c = colors[idx % len(colors)]
                # Thin, crisp box
                cv2.rectangle(dino_img, (x1, y1), (x2, y2), c, 2)
                
                # Clean text with a subtle shadow
                label = item.get("label", "food")
                cv2.putText(dino_img, label, (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3) # Shadow
                cv2.putText(dino_img, label, (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 1) # Text
                
        _, buffer = cv2.imencode('.jpg', dino_img)
        dino_b64 = base64.b64encode(buffer).decode('utf-8')
        yield f"data: {json.dumps({'step': 'dino', 'status': 'done', 'message': 'DINO localized', 'image': dino_b64})}\n\n"

        # SAM2 REAL Masks (Enterprise Clean)
        sam2_img = img_array.copy()
        for idx, item in enumerate(foods_data):
            mask_b64 = item.get("mask_b64")
            if mask_b64:
                mask_bytes = base64.b64decode(mask_b64)
                mask_np = cv2.imdecode(np.frombuffer(mask_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
                if mask_np.shape != img_array.shape[:2]:
                    mask_np = cv2.resize(mask_np, (img_array.shape[1], img_array.shape[0]), interpolation=cv2.INTER_NEAREST)
                
                c = colors[idx % len(colors)]
                
                # Pure translucent fill, no thick contour lines
                colored_mask = np.zeros_like(sam2_img)
                colored_mask[mask_np > 127] = c
                sam2_img = cv2.addWeighted(sam2_img, 1.0, colored_mask, 0.35, 0)

        _, buffer = cv2.imencode('.jpg', sam2_img)
        sam2_b64 = base64.b64encode(buffer).decode('utf-8')
        yield f"data: {json.dumps({'step': 'sam2', 'status': 'done', 'message': 'SAM2 segmented', 'image': sam2_b64})}\n\n"

        # Depth (REAL DEPTH MAP)
        real_depth_b64 = detections.get("depth_b64")
        if real_depth_b64:
            depth_bytes = base64.b64decode(real_depth_b64)
            depth_np = cv2.imdecode(np.frombuffer(depth_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
            depth_img = cv2.applyColorMap(depth_np, cv2.COLORMAP_JET)
        else:
            # Fallback if missing
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
                mass = match * d
                try:
                    data = search_food(q)
                    if "error" not in data:
                        nutrition_results.append(format_nutrition_response(data, mass))
                except Exception: pass

        final_payload = {"foods": nutrition_results, "total_detected": len(foods_data), "total_nutrition": aggregate_nutrition(nutrition_results)}
        yield f"data: {json.dumps({'step': 'nutrition', 'status': 'done', 'message': 'Complete', 'final': final_payload})}\n\n"
    finally:
        if image_path and os.path.exists(image_path): os.remove(image_path)
