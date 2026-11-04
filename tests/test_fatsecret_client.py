from backend.app.nutrition.providers.fatsecret_client import get_access_token, search_food

def test_get_access_token():
    token = get_access_token()

    assert token is not None
    assert len(token) > 100

def test_search_food():
    result = search_food("rice")

    assert result["food_name"] == "White Rice"

    assert "reference_serving" in result
    assert "servings" in result

    assert "calories" in result["reference_serving"]
    assert "protein" in result["reference_serving"]
    assert "carbohydrate" in result["reference_serving"]
