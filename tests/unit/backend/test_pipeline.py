import requests
import pytest

from backend.app.services import pipeline


def test_merge_volumes_ignores_missing_and_nonpositive_values():
    merged = pipeline._merge_volumes(
        [
            {"label": "Pizza", "volume_cm3": 12},
            {"label": "pizza", "volume_cm3": 3.5},
            {"label": "fries", "volume_cm3": None},
            {"label": "soup", "volume_cm3": 0},
            {"volume_cm3": 9},
        ]
    )

    assert merged == {"pizza": 15.5}


def test_nutrition_lookup_maps_provider_error_to_stable_code(monkeypatch):
    monkeypatch.setattr(
        pipeline,
        "search_food",
        lambda food_name: {"error": "upstream details must not be exposed"},
    )

    result, failure_code = pipeline._lookup_nutrition("pizza", 150.0)

    assert result is None
    assert failure_code == "provider_rejected_request"


def test_nutrition_lookup_maps_transport_failure(monkeypatch):
    def raise_timeout(food_name):
        raise requests.Timeout("sensitive transport details")

    monkeypatch.setattr(pipeline, "search_food", raise_timeout)

    result, failure_code = pipeline._lookup_nutrition("pizza", 150.0)

    assert result is None
    assert failure_code == "provider_unavailable"


@pytest.mark.asyncio
async def test_nutrition_enrichment_reports_partial_success(monkeypatch):
    outcomes = iter(
        [
            ({"food_name": "pizza"}, None),
            (None, "provider_unavailable"),
        ]
    )
    monkeypatch.setattr(
        pipeline,
        "_lookup_nutrition",
        lambda food_name, mass: next(outcomes),
    )

    detected_foods = [
        {"search_query": "pizza", "density_g_cm3": 1.0},
        {"search_query": "fries", "density_g_cm3": 0.5},
    ]
    volumes = {"pizza": 100.0, "fries": 50.0}

    results, errors, status = await pipeline._enrich_nutrition(
        detected_foods,
        volumes,
    )

    assert results == [{"food_name": "pizza"}]
    assert errors == [{"food": "fries", "code": "provider_unavailable"}]
    assert status == "partial"
