import json
from unittest.mock import patch

from fastapi.responses import JSONResponse

from backend.app.api.routes.nutrition import nutrition


def test_nutrition_route_returns_reference_record_when_mass_is_omitted():
    record = {
        "food_name": "Rice",
        "food_id": "123",
        "reference_serving": {"grams": 100.0, "calories": 130.0},
    }
    with patch("backend.app.api.routes.nutrition.search_food", return_value=record):
        assert nutrition(food="rice", mass_g=None) == record


def test_nutrition_route_scales_only_when_mass_is_supplied():
    record = {"food_name": "Rice", "reference_serving": {"grams": 100.0}}
    formatted = {"food_name": "Rice", "mass_g": 150.0, "nutrition": {"calories": 195.0}}
    with (
        patch("backend.app.api.routes.nutrition.search_food", return_value=record),
        patch("backend.app.api.routes.nutrition.format_nutrition_response", return_value=formatted) as formatter,
    ):
        assert nutrition(food="rice", mass_g=150.0) == formatted
        formatter.assert_called_once_with(record, 150.0)


def test_provider_failure_returns_safe_gateway_response():
    with patch(
        "backend.app.api.routes.nutrition.search_food",
        return_value={"error": "FatSecret is temporarily unavailable."},
    ):
        response = nutrition(food="rice", mass_g=None)

    assert isinstance(response, JSONResponse)
    assert response.status_code == 503
    body = json.loads(response.body)
    assert body == {
        "detail": "FatSecret is temporarily unavailable.",
        "provider": "fatsecret",
    }


def test_not_found_uses_404():
    with patch(
        "backend.app.api.routes.nutrition.search_food",
        return_value={"error": "Food not found."},
    ):
        response = nutrition(food="unlisted food", mass_g=None)
    assert response.status_code == 404


def test_blank_query_is_rejected():
    response = nutrition(food="   ", mass_g=None)
    assert response.status_code == 422
    assert json.loads(response.body)["detail"] == "Food search query must not be empty."
