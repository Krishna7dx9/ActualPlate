from unittest.mock import patch

from backend.app.services.pipeline import _build_scale_map, _enrich_nutrition


def test_scale_map_omits_missing_or_invalid_dimensions():
    scale_map = _build_scale_map([
        {
            "search_query": "leafy greens",
            "estimated_thickness_cm": None,
            "estimated_diameter_cm": float("nan"),
        },
        {
            "search_query": "bread",
            "estimated_thickness_cm": 2.5,
            "estimated_diameter_cm": True,
        },
    ])
    assert scale_map == {
        "leafy greens": {},
        "bread": {"thickness": 2.5},
    }


def test_nutrition_is_not_requested_without_valid_density_or_volume():
    foods = [
        {"search_query": "broccoli", "density_g_cm3": None},
        {"search_query": "bread", "density_g_cm3": 0.8},
    ]
    with patch("backend.app.services.pipeline.search_food") as search_food:
        results, errors, status = _enrich_nutrition(foods, {"broccoli": 10.0, "bread": None})

    assert results == []
    assert status == "unavailable"
    assert [item["code"] for item in errors] == ["density_unavailable", "volume_unavailable"]
    search_food.assert_not_called()


def test_nutrition_provider_failure_is_reported_not_silently_swallowed():
    foods = [{"search_query": "bread", "density_g_cm3": 0.8}]
    with patch(
        "backend.app.services.pipeline.search_food",
        side_effect=RuntimeError("private provider detail"),
    ):
        results, errors, status = _enrich_nutrition(foods, {"bread": 12.0})

    assert results == []
    assert errors == [{"food": "bread", "code": "provider_unavailable"}]
    assert status == "unavailable"
