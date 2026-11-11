from io import BytesIO
from unittest.mock import MagicMock, patch

from PIL import Image

from vision_service.recognition import food_recognition


class _FakeResponse:
    def __init__(self, content):
        self.choices = [
            type("Choice", (), {
                "message": type("Message", (), {"content": content})()
            })()
        ]


def _png_bytes():
    buffer = BytesIO()
    Image.new("RGB", (8, 8), color=(200, 100, 50)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_detect_food_uses_configured_model_and_actual_image_mime_type(monkeypatch):
    monkeypatch.setenv("FOOD_RECOGNITION_MODEL", "test/provider-model")
    monkeypatch.setenv("FOOD_RECOGNITION_MAX_TOKENS", "128")
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = _FakeResponse(
        '[{"search_query":"pizza","density_g_cm3":1.0,"estimated_thickness_cm":2.0,"estimated_diameter_cm":10.0}]'
    )

    with patch(
        "vision_service.recognition.food_recognition._get_client",
        return_value=fake_client,
    ):
        result = food_recognition.detect_food(_png_bytes())

    assert result[0]["search_query"] == "pizza"
    assert result[0]["estimated_thickness_cm"] == 2.0
    call = fake_client.chat.completions.create.call_args.kwargs
    assert call["model"] == "test/provider-model"
    assert call["messages"][0]["role"] == "system"
    assert "untrusted user-provided data" in call["messages"][0]["content"]
    assert call["messages"][1]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_invalid_or_missing_visual_measurements_remain_null():
    result = food_recognition._validate_items([
        {
            "search_query": "broccoli",
            "density_g_cm3": True,
            "estimated_diameter_cm": float("nan"),
            "estimated_thickness_cm": None,
        }
    ])

    assert result == [{
        "search_query": "broccoli",
        "density_g_cm3": None,
        "estimated_diameter_cm": None,
        "estimated_thickness_cm": None,
    }]


def test_recognition_client_is_lazy_and_requires_provider_credentials(monkeypatch):
    monkeypatch.setattr(food_recognition, "_CLIENT", None)
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    with patch("vision_service.recognition.food_recognition.OpenAI") as mock_openai:
        try:
            food_recognition._get_client()
        except food_recognition.RecognitionConfigurationError as error:
            assert "credentials are not configured" in str(error)
        else:
            raise AssertionError("Missing credentials should fail when a provider call is requested.")
        mock_openai.assert_not_called()


def test_malformed_provider_json_is_not_misreported_as_no_food(monkeypatch):
    monkeypatch.setenv("FOOD_RECOGNITION_MODEL", "test/provider-model")
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = _FakeResponse("not-json")
    with patch(
        "vision_service.recognition.food_recognition._get_client",
        return_value=fake_client,
    ):
        try:
            food_recognition.detect_food(_png_bytes())
        except food_recognition.RecognitionOutputError:
            pass
        else:
            raise AssertionError("Malformed provider output must be surfaced as an error.")
