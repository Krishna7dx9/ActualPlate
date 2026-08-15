from fastapi import APIRouter, Query
from nutrition.providers.fatsecret_client import search_food
from nutrition.nutrition_formatter import format_nutrition_response

router = APIRouter()

@router.get("/nutrition")
def nutrition(food: str = Query(..., description="Food name to search")):
    nutrition_data = search_food(food)
    if "error" in nutrition_data:
        return nutrition_data
    return format_nutrition_response(nutrition_data)