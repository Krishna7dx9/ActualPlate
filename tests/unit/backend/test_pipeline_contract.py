import json
from io import BytesIO
from types import SimpleNamespace

import pytest
from fastapi import UploadFile

from backend.app.services import pipeline


def test_scale_map_omits_missing_or_invalid_dimensions():
    scale_map = pipeline._build_scale_map(
        [
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
        ]
    )
    assert scale_map == {
        "leafy greens": {},
        "bread": {"thickness": 2.5},
    }


@pytest.mark.asyncio
async def test_nutrition_is_not_requested_without_valid_density_or_volume(monkeypatch):
    def unexpected_lookup(*_args):
        pytest.fail("Unvalidated or missing physical measurements must not reach nutrition lookup.")

    monkeypatch.setattr(pipeline, "_lookup_nutrition", unexpected_lookup)
    foods = [
        {"search_query": "broccoli", "density_g_cm3": None},
        {"search_query": "bread", "density_g_cm3": 0.8},
    ]

    results, errors, status = await pipeline._enrich_nutrition(
        foods,
        {"broccoli": 10.0, "bread": None},
    )

    assert results == []
    assert errors == []
    assert status == "not_available"


@pytest.mark.asyncio
async def test_nutrition_provider_failure_is_reported_not_silently_swallowed(monkeypatch):
    monkeypatch.setattr(
        pipeline,
        "_lookup_nutrition",
        lambda _food, _mass: (None, "provider_unavailable"),
    )
    results, errors, status = await pipeline._enrich_nutrition(
        [{"search_query": "bread", "density_g_cm3": 0.8}],
        {"bread": 12.0},
    )

    assert results == []
    assert errors == [{"food": "bread", "code": "provider_unavailable"}]
    assert status == "unavailable"


@pytest.mark.asyncio
async def test_stream_render_error_uses_valid_sse_frame(monkeypatch):
    monkeypatch.setattr(
        pipeline,
        "detect_food",
        lambda _image: [{"search_query": "bread", "density_g_cm3": 0.8}],
    )
    monkeypatch.setattr(
        pipeline,
        "GroundingDINODetector",
        lambda: SimpleNamespace(detect=lambda **_kwargs: {"foods": []}),
    )
    monkeypatch.setattr(pipeline.cv2, "imread", lambda _path: None)

    upload = UploadFile(filename="image.jpg", file=BytesIO(b"image"))
    events = [event async for event in pipeline.process_image_stream(upload)]

    error_event = events[-1]
    assert error_event.endswith("\\n\\n")
    payload = json.loads(error_event.removeprefix("data: ").strip())
    assert payload == {
        "step": "error",
        "status": "error",
        "message": "Image rendering is temporarily unavailable.",
    }


def test_public_geometry_removes_all_encoded_masks():
    geometry = [
        {
            "label": "bread",
            "mask_b64": "png-data",
            "mask_array_b64": "npz-data",
            "volume_cm3": None,
        }
    ]
    assert pipeline._clean_geometry_payload(geometry) == [
        {"label": "bread", "volume_cm3": None}
    ]
