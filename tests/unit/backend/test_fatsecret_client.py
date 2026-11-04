from unittest.mock import patch, MagicMock
from backend.app.nutrition.providers.fatsecret_client import get_access_token, search_food

@patch("backend.app.nutrition.providers.fatsecret_client.requests.post")
def test_get_access_token(mock_post):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"access_token": "fake_token_12345", "expires_in": 3600}
    mock_post.return_value = mock_response

    token = get_access_token()
    assert token == "fake_token_12345"

@patch("backend.app.nutrition.providers.fatsecret_client.get_access_token")
@patch("backend.app.nutrition.providers.fatsecret_client.requests.get")
def test_search_food(mock_get, mock_get_token):
    # 1. Mock the token function to avoid the global cache bug
    mock_get_token.return_value = "fake_token"
    
    # 2. Mock the two GET requests (foods.search, then food.get)
    mock_search_response = MagicMock()
    mock_search_response.status_code = 200
    mock_search_response.json.return_value = {
        "foods": {"food": [{"food_id": "123"}]}
    }
    
    mock_details_response = MagicMock()
    mock_details_response.status_code = 200
    mock_details_response.json.return_value = {
        "food": {
            "food_name": "White Rice",
            "food_id": "123",
            "food_type": "Generic",
            "food_url": "http://fake.url",
            "servings": {
                "serving": [
                    {
                        "metric_serving_unit": "g",
                        "metric_serving_amount": "100",
                        "serving_description": "100 g",
                        "calories": "200",
                        "protein": "4",
                        "fat": "1",
                        "carbohydrate": "45"
                    }
                ]
            }
        }
    }
    
    # Set up the mock to return search first, details second
    mock_get.side_effect = [mock_search_response, mock_details_response]

    # 3. Act
    result = search_food("rice")
    
    # 4. Assert
    assert result["food_name"] == "White Rice"
    assert result["reference_serving"]["calories"] == 200.0
    assert result["reference_serving"]["carbohydrate"] == 45.0