from backend.app.nutrition.nutrition_aggregator import aggregate_nutrition

def test_aggregate_multiple_foods():
    # GIVEN: Two foods with known nutritional values
    results = [
        {"nutrition": {"calories": 100.5, "protein": 10, "fat": 5, "carbohydrate": 20, "sugar": 2, "fiber": 1}},
        {"nutrition": {"calories": 200.0, "protein": 20, "fat": 10, "carbohydrate": 30, "sugar": 4, "fiber": 2}}
    ]
    
    # WHEN: We aggregate them
    total = aggregate_nutrition(results)
    
    # THEN: The math should add up perfectly and round to 2 decimals
    assert total["calories"] == 300.5
    assert total["protein"] == 30.0
    assert total["fat"] == 15.0
    assert total["carbohydrate"] == 50.0
    assert total["sugar"] == 6.0
    assert total["fiber"] == 3.0

def test_aggregate_missing_keys():
    # GIVEN: Foods with missing nutritional keys
    results = [
        {"nutrition": {"calories": 100}}, # Missing protein, fat, etc.
        {"nutrition": {}}                 # Empty nutrition dict
    ]
    
    # WHEN: We aggregate them
    total = aggregate_nutrition(results)
    
    # THEN: Missing keys should default to 0 and not crash
    assert total["calories"] == 100.0
    assert total["protein"] == 0.0
    assert total["fat"] == 0.0

def test_aggregate_empty_list():
    # GIVEN: No foods detected
    results = []
    
    # WHEN: We aggregate an empty list
    total = aggregate_nutrition(results)
    
    # THEN: Everything should be 0
    assert total["calories"] == 0
    assert total["protein"] == 0