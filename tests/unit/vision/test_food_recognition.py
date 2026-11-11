from io import BytesIO
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from PIL import Image

from vision_service.recognition.food_recognition import (
    _get_client,
    _validate_items,
    detect_food,
)
from vision_service.settings import settings


class _FakeResponse:
    def __init__(self, content):
        self.choices = [
            type(
                "Choice",
                (),
                {"message": type("Message", (), {"content": content})},
            )
        ]


def _valid_png_bytes():
    buffer = BytesIO()
    Image.new("RGB", (8, 8)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_detect_food_uses_configured_model_and_decodable_image():
    fake_json = (
        '[{"search_query": "pizza", "density_g_cm3": 1.0, '
        '"estimated_thickness_cm": 2.0, "estimated_diameter_cm": 10.0}]'
    )
    create = MagicMock(return_value=_FakeResponse(fake_json))
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create),
        )
    )

    with patch(
        "vision_service.recognition.food_recognition._get_client",
        return_value=fake_client,
    ):
        result = detect_food(_valid_png_bytes())

    assert isinstance(result, list)
    assert result[0]["search_query"] == "pizza"
    assert result[0]["estimated_thickness_cm"] == 2.0
    assert create.call_args.kwargs["model"] == settings.openrouter_model
    assert create.call_args.kwargs["max_tokens"] == settings.openrouter_max_tokens
    image_url = create.call_args.kwargs["messages"][0]["content"][1]["image_url"]["url"]
    assert image_url.startswith("data:image/png;base64,")


def test_provider_client_uses_configured_request_timeout():
    _get_client.cache_clear()
    try:
        with patch("vision_service.recognition.food_recognition.OpenAI") as openai:
            _get_client()
        assert openai.call_args.kwargs["timeout"] == settings.request_timeout_seconds
    finally:
        _get_client.cache_clear()


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


def test_boolean_measurements_are_not_accepted_as_numeric_values():
    result = _validate_items(
        [
            {"search_query": "invalid density", "density_g_cm3": True},
            {
                "search_query": "pizza",
                "density_g_cm3": 1.0,
                "estimated_diameter_cm": True,
                "estimated_thickness_cm": False,
            },
        ]
    )

    assert len(result) == 1
    assert result[0]["density_g_cm3"] == 1.0
    assert result[0]["estimated_diameter_cm"] is None
    assert result[0]["estimated_thickness_cm"] is None
