def format_nutrition_response(nutrition_data, mass_g=None):
    """
    Scale FatSecret reference nutrition to the actual mass of the food.

    Requires nutrition_data['reference_serving']['grams'] to be present
    and positive. If missing, returns an error dict rather than producing
    a fabricated scale.
    """

    reference = nutrition_data["reference_serving"]
    reference_grams = reference.get("grams")

    if not isinstance(reference_grams, (int, float)) or reference_grams <= 0:
        return {
            "error": "reference_serving.grams missing or invalid",
            "food_name": nutrition_data.get("food_name"),
            "food_id": nutrition_data.get("food_id"),
            "reference_serving": {
                "description": reference.get("description"),
                "grams": reference_grams,
                "unit": reference.get("unit"),
            },
        }

    if mass_g is None or mass_g <= 0:
        return {
            "error": "mass_g missing or invalid",
            "food_name": nutrition_data.get("food_name"),
            "food_id": nutrition_data.get("food_id"),
            "mass_g": mass_g,
        }

    scale = mass_g / reference_grams

    def scaled(key):
        value = reference.get(key, 0) or 0
        return round(value * scale, 2)

    return {
        "food_name": nutrition_data["food_name"],
        "food_id": nutrition_data["food_id"],
        "mass_g": round(mass_g, 2),
        "scale_factor": round(scale, 4),
        "reference_serving": {
            "description": reference.get("description"),
            "grams": reference_grams,
            "unit": reference.get("unit"),
        },
        "nutrition": {
            "calories": scaled("calories"),
            "protein": scaled("protein"),
            "fat": scaled("fat"),
            "carbohydrate": scaled("carbohydrate"),
            "fiber": scaled("fiber"),
            "sugar": scaled("sugar"),
            "sodium": scaled("sodium"),
        },
    }