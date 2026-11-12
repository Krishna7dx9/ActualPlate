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



def test_stream_render_error_uses_valid_sse_frame(monkeypatch):
    import asyncio
    import json
    from io import BytesIO

    from fastapi import UploadFile

    from backend.app.services import pipeline

    monkeypatch.setattr(pipeline, "detect_food", lambda _image: [{"search_query": "bread"}])
    monkeypatch.setattr(pipeline.detector, "detect", lambda **_kwargs: {"foods": []})
    monkeypatch.setattr(pipeline.cv2, "imread", lambda _path: None)

    async def collect():
        upload = UploadFile(filename="image.jpg", file=BytesIO(b"image"))
        return [event async for event in pipeline.process_image_stream(upload)]

    events = asyncio.run(collect())
    error_event = events[-1]
    assert error_event.endswith("\n\n")
    payload = json.loads(error_event.removeprefix("data: ").strip())
    assert payload == {
        "step": "error",
        "status": "error",
        "message": "Image rendering is temporarily unavailable.",
    }


def test_public_geometry_removes_all_encoded_masks():
    from backend.app.services.pipeline import _clean_geometry_payload

    geometry = [{
        "label": "bread",
        "mask_b64": "png-data",
        "mask_array_b64": "npz-data",
        "volume_cm3": None,
    }]
    assert _clean_geometry_payload(geometry) == [{
        "label": "bread",
        "volume_cm3": None,
    }]
