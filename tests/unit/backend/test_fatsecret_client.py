from unittest.mock import MagicMock, patch

from backend.app.nutrition.providers import fatsecret_client as client
from backend.app.nutrition.providers.fatsecret_client import (
    get_access_token,
    normalize_food_response,
    search_food,
)


def _food_details(calories="200"):
    return {
        "food": {
            "food_name": "White Rice",
            "food_id": "123",
            "food_type": "Generic",
            "food_url": "https://example.invalid/food",
            "servings": {
                "serving": [
                    {
                        "metric_serving_unit": "g",
                        "metric_serving_amount": "100",
                        "serving_description": "100 g",
                        "calories": calories,
                        "protein": "4",
                        "fat": "1",
                        "carbohydrate": "45",
                    }
                ]
            },
        }
    }


def _response(status_code=200, payload=None, text=""):
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = payload if payload is not None else {}
    response.text = text
    return response


@patch("backend.app.nutrition.providers.fatsecret_client.requests.post")
def test_get_access_token_validates_and_caches_token(mock_post, monkeypatch):
    monkeypatch.setenv("FATSECRET_CLIENT_ID", "test-id")
    monkeypatch.setenv("FATSECRET_CLIENT_SECRET", "test-secret")
    monkeypatch.delenv("FATSECRET_TIMEOUT_SECONDS", raising=False)
    monkeypatch.setattr(client, "_access_token", None)
    monkeypatch.setattr(client, "_token_expiry", 0.0)
    mock_post.return_value = _response(
        payload={"access_token": "fake_token_12345", "expires_in": 3600}
    )

    assert get_access_token() == "fake_token_12345"
    assert get_access_token() == "fake_token_12345"
    mock_post.assert_called_once()


@patch("backend.app.nutrition.providers.fatsecret_client.get_access_token")
@patch("backend.app.nutrition.providers.fatsecret_client.requests.get")
def test_search_food_normalizes_list_servings(mock_get, mock_get_token):
    mock_get_token.return_value = "fake_token"
    mock_get.side_effect = [
        _response(payload={"foods": {"food": [{"food_id": "123"}]}}),
        _response(payload=_food_details()),
    ]

    result = search_food("rice")

    assert result["food_name"] == "White Rice"
    assert result["reference_serving"]["grams"] == 100.0
    assert result["reference_serving"]["calories"] == 200.0
    assert result["reference_serving"]["carbohydrate"] == 45.0
    assert mock_get.call_count == 2


def test_normalize_food_response_accepts_single_serving_object():
    payload = _food_details()
    payload["food"]["servings"]["serving"] = payload["food"]["servings"]["serving"][0]

    result = normalize_food_response(payload)

    assert result["food_id"] == "123"
    assert result["reference_serving"]["grams"] == 100.0


def test_normalize_food_response_rejects_non_gram_reference():
    payload = _food_details()
    serving = payload["food"]["servings"]["serving"][0]
    serving["metric_serving_unit"] = "ml"
    result = normalize_food_response(payload)
    assert result["error"] == "FatSecret food record has no valid gram-based serving."


def test_normalize_food_response_rejects_non_finite_nutrients():
    result = normalize_food_response(_food_details(calories="nan"))
    assert result["error"] == "FatSecret serving contains invalid nutrition values."


@patch("backend.app.nutrition.providers.fatsecret_client.get_access_token")
@patch("backend.app.nutrition.providers.fatsecret_client.requests.get")
def test_provider_errors_do_not_leak_upstream_body(mock_get, mock_get_token):
    mock_get_token.return_value = "fake_token"
    mock_get.return_value = _response(
        status_code=503,
        text="internal upstream diagnostic secret",
    )

    result = search_food("rice")

    assert result == {"error": "FatSecret is temporarily unavailable."}
    assert "secret" not in str(result)
    assert "details" not in result


def test_search_food_rejects_empty_or_oversized_query_without_network():
    assert search_food("   ")["error"] == "Food search query must contain 1 to 120 characters."
    assert search_food("x" * 121)["error"] == "Food search query must contain 1 to 120 characters."


def test_missing_provider_configuration_is_a_safe_error(monkeypatch):
    monkeypatch.setenv("FATSECRET_CLIENT_ID", "")
    monkeypatch.setenv("FATSECRET_CLIENT_SECRET", "")
    monkeypatch.setattr(client, "_access_token", None)
    monkeypatch.setattr(client, "_token_expiry", 0.0)

    result = search_food("rice")

    assert result == {"error": "FatSecret integration is not configured."}


@patch("backend.app.nutrition.providers.fatsecret_client.get_access_token")
@patch("backend.app.nutrition.providers.fatsecret_client.requests.get")
def test_expired_bearer_token_is_refreshed_once(mock_get, mock_get_token):
    mock_get_token.side_effect = ["expired-token", "fresh-token", "fresh-token"]
    mock_get.side_effect = [
        _response(status_code=401),
        _response(payload={"foods": {"food": [{"food_id": "123"}]}}),
        _response(payload=_food_details()),
    ]

    result = search_food("rice")

    assert result["food_name"] == "White Rice"
    assert mock_get.call_count == 3
    assert mock_get.call_args_list[0].kwargs["headers"]["Authorization"] == "Bearer expired-token"
    assert mock_get.call_args_list[1].kwargs["headers"]["Authorization"] == "Bearer fresh-token"

def test_normalize_food_response_rejects_boolean_nutrient():
    result = normalize_food_response(_food_details(calories=True))
    assert result["error"] == "FatSecret serving contains invalid nutrition values."
