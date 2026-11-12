import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import json
import os
import numpy as np

# Set the token before importing the app to pass startup validation
os.environ["VISION_API_TOKEN"] = "test-token"

from vision_service.inference.api import (
    app,
    _parse_scale_map,
    _validate_auth_configuration,
)

client = TestClient(app)


@patch("vision_service.inference.api.detect_and_measure")
def test_vision_api_accepts_backend_payload(mock_detect_and_measure):
    """
    CONTRACT TEST: Simulates the exact payload sent by grounding_dino_detector.py
    and verifies the Vision API accepts it without a 422 Unprocessable Entity error.
    """
    mock_result = MagicMock()
    mock_result.foods = []
    mock_result.masks = []
    mock_result.image_width = 100
    mock_result.image_height = 100
    mock_result.focal_length_px = 50.0
    mock_result.depth_unit = "relative"
    mock_result.depth_map = np.ones((100, 100), dtype=np.float32)
    mock_detect_and_measure.return_value = mock_result

    scale_map_payload = {
        "burger": {"thickness": 5.0, "diameter": 10.0},
        "fries": {"thickness": 1.0, "diameter": 2.0},
    }
    payload_data = {
        "labels": "burger,fries",
        "debug": "true",
        "scale_map": json.dumps(scale_map_payload),
    }
    files = {"file": ("test.jpg", b"fake_image_bytes", "image/jpeg")}
    headers = {"x-api-token": "test-token"}

    response = client.post(
        "/detect",
        data=payload_data,
        files=files,
        headers=headers,
    )

    assert response.status_code == 200
    assert response.json()["request_id"] is not None


def test_vision_api_rejects_invalid_scale_map_json():
    with pytest.raises(HTTPException) as error:
        _parse_scale_map("{invalid_json_string}")
    assert error.value.status_code == 400


@pytest.mark.parametrize(
    "raw_scale_map",
    [
        "[]",
        '"burger"',
        '{"burger": []}',
        '{"burger": {"diameter": true}}',
        '{"burger": {"diameter": NaN}}',
        '{"burger": {"thickness": -1}}',
        '{"": {"diameter": 10}}',
    ],
)
def test_scale_map_rejects_invalid_shapes_and_dimensions(raw_scale_map):
    with pytest.raises(HTTPException) as error:
        _parse_scale_map(raw_scale_map)
    assert error.value.status_code == 400


def test_scale_map_normalizes_labels_and_allows_unknown_dimensions():
    assert _parse_scale_map(
        '{" Burger ": {"diameter": 10, "thickness": null}}'
    ) == {"burger": {"diameter": 10.0, "thickness": None}}


def test_vision_api_rejects_invalid_token():
    payload_data = {"labels": "burger"}
    files = {"file": ("test.jpg", b"fake_image_bytes", "image/jpeg")}
    headers = {"x-api-token": "WRONG-TOKEN"}

    response = client.post(
        "/detect",
        data=payload_data,
        files=files,
        headers=headers,
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid API token."


def test_vision_auth_is_optional_outside_production():
    _validate_auth_configuration("development", None)


def test_vision_auth_is_required_in_production():
    with pytest.raises(RuntimeError, match="VISION_API_TOKEN"):
        _validate_auth_configuration("production", None)


def test_vision_auth_accepts_configured_production_token():
    _validate_auth_configuration("production", "configured-token")
