from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from backend.app.nutrition.nutrition_formatter import format_nutrition_response
from backend.app.nutrition.providers.fatsecret_client import search_food

router = APIRouter()


@router.get("/nutrition")
def nutrition(
    food: str = Query(
        ...,
        min_length=1,
        max_length=120,
        description="Food name to search in the nutrition database",
    ),
    mass_g: float | None = Query(default=None, gt=0, le=100000, description="Optional estimated food mass in grams"),
):
    query = food.strip()
    if not query:
        return JSONResponse(
            status_code=422,
            content={"detail": "Food search query must not be empty."},
        )

    nutrition_data = search_food(query)
    if "error" in nutrition_data:
        message = nutrition_data["error"]
        if message == "Food not found.":
            status_code = 404
        elif message == "FatSecret is temporarily unavailable.":
            status_code = 503
        else:
            status_code = 502
        return JSONResponse(
            status_code=status_code,
            content={"detail": message, "provider": "fatsecret"},
        )

    if mass_g is None:
        return nutrition_data

    try:
        result = format_nutrition_response(nutrition_data, mass_g)
    except (KeyError, TypeError, ValueError, OverflowError):
        return JSONResponse(
            status_code=502,
            content={
                "detail": "Nutrition provider returned an unusable record.",
                "provider": "fatsecret",
            },
        )

    if "error" in result:
        return JSONResponse(
            status_code=422,
            content={"detail": result["error"]},
        )
    return result
