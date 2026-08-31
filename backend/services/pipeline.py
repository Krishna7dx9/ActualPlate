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
        # 1. Recognize food labels
        # -------------------------------------------------

        detected_foods = detect_food(image)

        if not detected_foods:
            return {
                "error": "Food not detected"
            }

        food_list = [
            food.strip()
            for food in detected_foods.split(",")
            if food.strip()
        ]

        # -------------------------------------------------
        # 2. Run vision inference
        #
        # Colab owns:
        # - Grounding DINO
        # - SAM2
        # - Depth Anything V2
        # - Plane fitting
        # - Volume estimation
        #
        # Expected response:
        # {
        #     "foods": [
        #         {
        #             "label": "rice",
        #             "score": 0.86,
        #             "volume_relative": 399959.69
        #         }
        #     ]
        # }
        # -------------------------------------------------

        detections = detector.detect(
            image_path=image_path,
            labels=food_list,
        )

        print(detections)

        foods_data = detections.get("foods", [])

        nutrition_results = []
        failed = 0

        # -------------------------------------------------
        # 3. Nutrition lookup
        # -------------------------------------------------

        for food_item in foods_data:
            food_name = food_item["label"]

            volume_relative = food_item.get(
                "volume_relative",
                0,
            )

            data = search_food(food_name)

            if "error" in data:
                failed += 1
                continue

            # Temporary contract transition.
            #
            # volume_relative is NOT grams.
            # The density -> weight calculation will be
            # implemented in the next step.
            nutrition_results.append(
                format_nutrition_response(
                    data,
                    volume_relative,
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