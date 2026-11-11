import pytest

from vision_service.settings import VisionSettings


def test_vision_settings_parse_explicit_thresholds(monkeypatch):
    monkeypatch.setenv("BOX_THRESHOLD", "0.4")
    monkeypatch.setenv("TEXT_THRESHOLD", "0.3")
    monkeypatch.setenv("VISION_MAX_ARRAY_SIDE", "256")

    settings = VisionSettings.from_environment()

    assert settings.box_threshold == 0.4
    assert settings.text_threshold == 0.3
    assert settings.max_array_side == 256


@pytest.mark.parametrize("value", ["-0.1", "1.1", "nan", "inf", "not-a-number"])
def test_vision_settings_reject_invalid_threshold(value, monkeypatch):
    monkeypatch.setenv("BOX_THRESHOLD", value)

    with pytest.raises(ValueError, match="BOX_THRESHOLD"):
        VisionSettings.from_environment()


@pytest.mark.parametrize("value", ["0", "-1", "nan", "invalid"])
def test_vision_settings_reject_invalid_request_timeout(value, monkeypatch):
    monkeypatch.setenv("VISION_REQUEST_TIMEOUT_SECONDS", value)

    with pytest.raises(ValueError, match="VISION_REQUEST_TIMEOUT_SECONDS"):
        VisionSettings.from_environment()


def test_vision_settings_reject_invalid_worker_endpoint(monkeypatch):
    monkeypatch.setenv("VISION_INFERENCE_ENDPOINT", "not-a-url")

    with pytest.raises(ValueError, match="VISION_INFERENCE_ENDPOINT"):
        VisionSettings.from_environment()


def test_vision_settings_allow_missing_provider_secret_until_use(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    settings = VisionSettings.from_environment()

    assert settings.openrouter_api_key is None



def test_production_requires_provider_credentials(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        VisionSettings.from_environment()
