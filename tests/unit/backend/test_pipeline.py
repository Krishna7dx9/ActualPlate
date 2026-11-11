import math

import requests
import pytest

from backend.app.services import pipeline


def test_merge_volumes_ignores_missing_nonpositive_and_nonfinite_values():
    merged = pipeline._merge_volumes(
        [
            {"label": "Pizza", "volume_cm3": 12},
            {"label": "pizza", "volume_cm3": 3.5},
            {"label": "fries", "volume_cm3": None},
            {"label": "soup", "volume_cm3": 0},
            {"label": "water", "volume_cm3": float("inf")},
            {"label": "oil", "volume_cm3": float("nan")},
            {"label": "bool", "volume_cm3": True},
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


@pytest.mark.asyncio
async def test_nutrition_enrichment_skips_nonfinite_and_boolean_measurements(
    monkeypatch,
):
    def unexpected_lookup(*args):
        pytest.fail("Invalid measurements must not reach the nutrition provider.")

    monkeypatch.setattr(pipeline, "_lookup_nutrition", unexpected_lookup)

    detected_foods = [
        {"search_query": "nan", "density_g_cm3": float("nan")},
        {"search_query": "infinite", "density_g_cm3": float("inf")},
        {"search_query": "boolean", "density_g_cm3": True},
        {"search_query": "negative", "density_g_cm3": -1.0},
        {"search_query": "valid-density", "density_g_cm3": 1.0},
    ]
    volumes = {
        "nan": 10.0,
        "infinite": 10.0,
        "boolean": 10.0,
        "negative": 10.0,
        "valid-density": float("inf"),
    }

    results, errors, status = await pipeline._enrich_nutrition(
        detected_foods,
        volumes,
    )

    assert results == []
    assert errors == []
    assert status == "not_available"
