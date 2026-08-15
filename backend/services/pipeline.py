import os
import tempfile
from fastapi import UploadFile
from nutrition.providers.fatsecret_client import search_food
from nutrition.nutrition_formatter import format_nutrition_response
from nutrition.nutrition_aggregator import aggregate_nutrition
from vision_service.recognition.food_recognition import detect_food
from vision_service.portion.portion_estimator import estimate_portions
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
        if not detected_foods:
            return {"error": "Food not detected"}

        portions = estimate_portions(image, detected_foods)
        food_list = [f.strip() for f in detected_foods.split(",") if f.strip()]
        
        detections = detector.detect(image_path=image_path, labels=food_list)
        print(detections)

        nutrition_results = []
        failed = 0
        for food, portion in zip(food_list, portions):
            data = search_food(food)
            if "error" in data:
                failed += 1
                continue
            nutrition_results.append(format_nutrition_response(data, portion))

        return {
            "foods": nutrition_results,
            "total_detected": len(food_list),
            "total_found": len(nutrition_results),
            "total_failed": failed,
            "total_nutrition": aggregate_nutrition(nutrition_results)
        }
    except Exception as exc:
        return {"error": "Image processing failed", "details": str(exc)}
    finally:
        if image_path and os.path.exists(image_path):
            os.remove(image_path)