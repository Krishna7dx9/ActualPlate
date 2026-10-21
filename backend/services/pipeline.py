import os
import tempfile
import base64
import cv2
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
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".jpg",
        ) as temp:
            temp.write(image)
            image_path = temp.name

        # -------------------------------------------------
        # 1. Gemma: food search queries + densities
        #
        # detect_food returns a list of dicts:
        #   [{"search_query": str, "density_g_cm3": float}, ...]
        # -------------------------------------------------

        detected_foods = detect_food(image)
        print(f"DEBUG - GEMMA RAW OUTPUT: {detected_foods}")

        if not detected_foods:
            return {
                "error": "Food not detected"
            }

        # -------------------------------------------------
        # 2. Colab vision inference
        #
        # We send search queries as labels. Colab returns
        # one volume_cm3 per detected food. We match Gemma's
        # item i to Colab's item i by position, and we verify
        # the count matches before using any of it.
        # -------------------------------------------------

        detections = detector.detect(
            image_path=image_path,
            labels=[item["search_query"] for item in detected_foods],
        )

        foods_data = detections.get("foods", [])

        # -------------------------------------------------
        # 3. Merge Colab detections by label
        #
        # Grounding DINO returns one box per matching region,
        # so a single Gemma label can produce multiple Colab
        # detections (duplicates, partials, shadows). We merge
        # them by label: keep the highest-scoring detection
        # per label, and sum the volumes of the duplicates so
        # the merged item reflects the total food present.
        #
        # After merging, the count of unique labels will match
        # Gemma's label count in the normal case, and any
        # surplus Colab labels are dropped (they have no
        # corresponding Gemma item and no density).
        # -------------------------------------------------

        merged: dict[str, dict] = {}

        for vision_item in foods_data:
            label = vision_item.get("label", "").strip().lower()
            if not label:
                continue

            volume = vision_item.get("volume_cm3")
            score = vision_item.get("score", 0.0)

            if label not in merged:
                merged[label] = {
                    "score": score,
                    "volume_cm3": volume if volume is not None else 0.0,
                }
                continue

            # Keep the best score seen for this label.
            if score > merged[label]["score"]:
                merged[label]["score"] = score

            # Sum volumes of duplicate detections.
            if volume is not None:
                merged[label]["volume_cm3"] += volume

        # -------------------------------------------------
        # 4. Nutrition lookup, per Gemma item
        # -------------------------------------------------

        nutrition_results = []
        failed = 0

        for gemma_item in detected_foods:

            search_query = gemma_item["search_query"]
            density_g_cm3 = gemma_item["density_g_cm3"]

            label_key = search_query.strip().lower()
            match = merged.get(label_key)

            if match is None or match["volume_cm3"] <= 0:
                failed += 1
                continue

            mass_g = match["volume_cm3"] * density_g_cm3

            data = search_food(search_query)

            if "error" in data:
                failed += 1
                continue

            nutrition_results.append(
                format_nutrition_response(
                    data,
                    mass_g,
                )
            )

        # -------------------------------------------------
        # 4. Final response
        # -------------------------------------------------

        return {
            "foods": nutrition_results,
            "total_detected": len(foods_data),
            "total_found": len(nutrition_results),
            "total_failed": failed,
            "total_nutrition": aggregate_nutrition(
                nutrition_results
            ),
        }

    except Exception as exc:
        return {
            "error": "Image processing failed",
            "details": str(exc),
        }

    finally:
        if image_path and os.path.exists(image_path):
            os.remove(image_path)


import json
from fastapi.responses import StreamingResponse

async def process_image_stream(file):
    image = await file.read()
    image_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
            temp.write(image)
            image_path = temp.name

        # 1. Gemma Step
        yield f"data: {json.dumps({'step': 'gemma', 'status': 'running', 'message': 'Gemma is analyzing the image...'})}\n\n"
        
        detected_foods = detect_food(image)
        
        # YIELD GEMMA OUTPUT EXACTLY
        gemma_msg = f"Gemma detected: {detected_foods}"
        yield f"data: {json.dumps({'step': 'gemma', 'status': 'done', 'message': gemma_msg, 'data': detected_foods})}\n\n"
        
        if not detected_foods:
            yield f"data: {json.dumps({'step': 'error', 'message': 'No food detected'})}\n\n"
            return

        # 2. Vision (DINO + SAM2 + Depth) Step
        yield f"data: {json.dumps({'step': 'vision', 'status': 'running', 'message': 'Running DINO, SAM2, and Depth Pro...'})}\n\n"
        
        detections = detector.detect(
            image_path=image_path,
            labels=[item["search_query"] for item in detected_foods],
        )
        foods_data = detections.get("foods", [])
        
        # Draw Bounding Boxes for UI
        img_array = cv2.imread(image_path)
        for item in foods_data:
            bbox = item.get("bbox") # Assuming DINO returns bbox
            label = item.get("label", "food")
            if bbox:
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(img_array, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(img_array, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
        _, buffer = cv2.imencode('.jpg', img_array)
        b64_img = base64.b64encode(buffer).decode('utf-8')
        
        yield f"data: {json.dumps({'step': 'vision', 'status': 'done', 'message': 'Vision processing complete', 'image': b64_img})}\n\n"

        # 3. Nutrition Step
        yield f"data: {json.dumps({'step': 'nutrition', 'status': 'running', 'message': 'Fetching FatSecret data...'})}\n\n"

        merged = {}
        for vision_item in foods_data:
            label = vision_item.get("label", "").strip().lower()
            volume = vision_item.get("volume_cm3")
            if label and volume:
                if label not in merged:
                    merged[label] = {"volume_cm3": volume}
                else:
                    merged[label]["volume_cm3"] += volume

        nutrition_results = []
        for gemma_item in detected_foods:
            search_query = gemma_item["search_query"]
            density = gemma_item["density_g_cm3"]
            match = merged.get(search_query.strip().lower())
            
            if match:
                mass_g = match["volume_cm3"] * density
                mass_g = min(mass_g, 2000.0) # Clamp
                try:
                    data = search_food(search_query)
                    if "error" not in data:
                        nutrition_results.append(format_nutrition_response(data, mass_g))
                except Exception:
                    pass

        final_payload = {
            "foods": nutrition_results,
            "total_detected": len(foods_data),
            "total_nutrition": aggregate_nutrition(nutrition_results)
        }
        
        yield f"data: {json.dumps({'step': 'nutrition', 'status': 'done', 'message': 'Complete', 'final': final_payload})}\n\n"

    except Exception as exc:
        yield f"data: {json.dumps({'step': 'error', 'message': str(exc)})}\n\n"
    finally:
        if image_path and os.path.exists(image_path):
            os.remove(image_path)
