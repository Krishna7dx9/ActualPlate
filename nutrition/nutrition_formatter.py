def format_nutrition_response(nutrition_data, portion=None):
    reference = nutrition_data["reference_serving"]

    return {
        "food_name": nutrition_data["food_name"],
        "food_id": nutrition_data["food_id"],
        "portion": portion,
        "nutrition": {
            "calories": reference.get("calories", 0),
            "protein": reference.get("protein", 0),
            "fat": reference.get("fat", 0),
            "carbohydrate": reference.get("carbohydrate", 0),
            "fiber": reference.get("fiber", 0),
            "sugar": reference.get("sugar", 0),
            "sodium": reference.get("sodium", 0),
        }
    }