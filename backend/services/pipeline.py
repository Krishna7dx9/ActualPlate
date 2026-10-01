import os
import tempfile

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

        if len(foods_data) != len(detected_foods):
            return {
                "error": "Detection count mismatch between Gemma and Colab",
                "gemma_count": len(detected_foods),
                "colab_count": len(foods_data),
            }

        # -------------------------------------------------
        # 3. Nutrition lookup, per Gemma item
        # -------------------------------------------------

        nutrition_results = []
        failed = 0

        for gemma_item, vision_item in zip(detected_foods, foods_data):

            search_query = gemma_item["search_query"]
            density_g_cm3 = gemma_item["density_g_cm3"]
            volume_cm3 = vision_item.get("volume_cm3")

            if volume_cm3 is None:
                failed += 1
                continue

            mass_g = volume_cm3 * density_g_cm3

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