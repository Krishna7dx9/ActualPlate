from unittest.mock import patch
from vision_service.recognition.food_recognition import detect_food, _validate_items

class _FakeResponse:
    def __init__(self, content):
        self.choices = [type("Choice", (), {"message": type("Message", (), {"content": content})})]

def test_detect_food_uses_supported_model_slug():
    # Provide JSON that matches the dictionary format
    fake_json = '[{"search_query": "pizza", "density_g_cm3": 1.0, "estimated_thickness_cm": 2.0, "estimated_diameter_cm": 10.0}]'
    
    with patch(
        "vision_service.recognition.food_recognition.client.chat.completions.create",
        return_value=_FakeResponse(fake_json),
    ) as mock_create:
        result = detect_food(b"fake-image-bytes")

        # Assert on the dictionary, not the string
        assert isinstance(result, list)
        assert result[0]["search_query"] == "pizza"
        assert result[0]["estimated_thickness_cm"] == 2.0


def test_missing_physical_dimensions_are_not_filled_with_defaults():
    result = _validate_items(
        [{"search_query": "pizza", "density_g_cm3": 1.0}]
    )

    assert result[0]["estimated_diameter_cm"] is None
    assert result[0]["estimated_thickness_cm"] is None


def test_non_finite_physical_dimensions_are_rejected():
    result = _validate_items(
        [
            {
                "search_query": "pizza",
                "density_g_cm3": 1.0,
                "estimated_diameter_cm": float("nan"),
                "estimated_thickness_cm": float("inf"),
            }
        ]
    )

    assert result[0]["estimated_diameter_cm"] is None
    assert result[0]["estimated_thickness_cm"] is None
