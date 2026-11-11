from unittest.mock import MagicMock, patch

import pytest

from vision_service.detection import grounding_dino_detector as detector_module


def test_detector_requires_configured_endpoint(monkeypatch):
    monkeypatch.setattr(detector_module, "VISION_INFERENCE_ENDPOINT", "")
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.delenv("VISION_API_TOKEN", raising=False)

    with pytest.raises(RuntimeError, match="VISION_INFERENCE_ENDPOINT"):
        detector_module.GroundingDINODetector()


def test_detector_allows_tokenless_development(monkeypatch):
    monkeypatch.setattr(
        detector_module,
        "VISION_INFERENCE_ENDPOINT",
        "http://localhost:8001",
    )
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.delenv("VISION_API_TOKEN", raising=False)

    detector = detector_module.GroundingDINODetector()

    assert detector.api_token is None


def test_detector_requires_token_in_production(monkeypatch):
    monkeypatch.setattr(
        detector_module,
        "VISION_INFERENCE_ENDPOINT",
        "http://localhost:8001",
    )
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.delenv("VISION_API_TOKEN", raising=False)

    with pytest.raises(RuntimeError, match="VISION_API_TOKEN"):
        detector_module.GroundingDINODetector()


def test_detector_omits_auth_header_when_development_token_is_absent(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        detector_module,
        "VISION_INFERENCE_ENDPOINT",
        "http://localhost:8001",
    )
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.delenv("VISION_API_TOKEN", raising=False)

    image_path = tmp_path / "sample.jpg"
    image_path.write_bytes(b"image-bytes")
    response = MagicMock()
    response.json.return_value = {"foods": []}

    with patch.object(
        detector_module.requests,
        "post",
        return_value=response,
    ) as mock_post:
        detector_module.GroundingDINODetector().detect(
            str(image_path),
            labels=["pizza"],
        )

    assert mock_post.call_args.kwargs["headers"] == {}
