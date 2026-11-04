from backend.app.nutrition.nutrition_formatter import format_nutrition_response

def test_format_nutrition_perfect_scale():
    # GIVEN: A reference serving of 100g, but the actual mass is 200g (scale = 2.0)
    mock_data = {
        "food_name": "Apple",
        "food_id": "123",
        "reference_serving": {
            "grams": 100.0,
            "description": "1 medium",
            "unit": "g",
            "calories": 52.0,
            "protein": 0.3,
            "fat": 0.2,
            "carbohydrate": 14.0,
            "fiber": 2.4,
            "sugar": 10.0,
            "sodium": 1.0
        }
    }
    
    # WHEN: We format it for 200g
    result = format_nutrition_response(mock_data, mass_g=200.0)
    
    # THEN: The scale should be 2.0, and all macros should be doubled
    assert "error" not in result
    assert result["scale_factor"] == 2.0
    assert result["nutrition"]["calories"] == 104.0
    assert result["nutrition"]["protein"] == 0.6
    assert result["nutrition"]["carbohydrate"] == 28.0

def test_format_missing_reference_grams():
    # GIVEN: Reference serving is missing grams (Invalid)
    mock_data = {
        "food_name": "Apple",
        "food_id": "123",
        "reference_serving": {"calories": 52.0} # No grams!
    }
    
    # WHEN: We format it
    result = format_nutrition_response(mock_data, mass_g=200.0)
    
    # THEN: It should safely return an error, not crash
    assert "error" in result
    assert result["error"] == "reference_serving.grams missing or invalid"

def test_format_invalid_mass():
    # GIVEN: A valid reference serving, but mass_g is 0 (Invalid)
    mock_data = {
        "food_name": "Apple",
        "food_id": "123",
        "reference_serving": {"grams": 100.0, "calories": 52.0}
    }
    
    # WHEN: We format it with 0g mass
    result = format_nutrition_response(mock_data, mass_g=0)
    
    # THEN: It should safely return an error
    assert "error" in result
    assert result["error"] == "mass_g missing or invalid"