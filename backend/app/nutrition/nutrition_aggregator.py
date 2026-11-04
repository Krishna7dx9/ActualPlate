def aggregate_nutrition(nutrition_results):
    total_calories = 0
    total_protein = 0
    total_fat = 0
    total_carbs = 0
    total_sugar = 0
    total_fiber = 0

    for food in nutrition_results:
        nutrition = food["nutrition"]

        total_calories += nutrition.get("calories", 0)
        total_protein += nutrition.get("protein", 0)
        total_fat += nutrition.get("fat", 0)
        total_carbs += nutrition.get("carbohydrate", 0)
        total_sugar += nutrition.get("sugar", 0)
        total_fiber += nutrition.get("fiber", 0)

    return {
        "calories": round(total_calories, 2),
        "protein": round(total_protein, 2),
        "fat": round(total_fat, 2),
        "carbohydrate": round(total_carbs, 2),
        "sugar": round(total_sugar, 2),
        "fiber": round(total_fiber, 2)
    }