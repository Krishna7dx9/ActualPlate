import base64
import io
import json
import os
from unittest.mock import patch

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

# Set the token before importing the app.
os.environ["VISION_API_TOKEN"] = "test-token"

from vision_service.inference.api import app
from vision_service.inference.inference import InferenceResult, FoodDetection

client = TestClient(app)


@pytest.fixture
def mock_inference_result():
    """Mocks the output of the heavy GPU pipeline."""
    mask = np.ones((500, 500), dtype=np.uint8)
    success, encoded_mask = cv2.imencode(".png", mask)
    assert success
    mask_png_b64 = base64.b64encode(encoded_mask).decode("ascii")

    food1 = FoodDetection(
        label="burger",
        score=0.95,
        mask_pixels=250000,
        depth_stats={"min": 1.0, "max": 1.5},
        bbox=[10.0, 10.0, 110.0, 110.0],
        mask_b64=mask_png_b64,
    )
    food2 = FoodDetection(
        label="fries",
        score=0.90,
        mask_pixels=250000,
        depth_stats={"min": 1.2, "max": 1.3},
        bbox=[200.0, 200.0, 300.0, 300.0],
        mask_b64=mask_png_b64,
    )

    return InferenceResult(
        foods=[food1, food2],
        image_width=500,
        image_height=500,
        focal_length_px=800.0,
        depth_unit="relative",
        depth_map=np.ones((500, 500), dtype=np.float32),
        masks=[mask.astype(bool), mask.astype(bool)],
    )


@patch("vision_service.inference.api.detect_and_measure")
def test_api_applies_per_food_scale_map(mock_detect_and_measure, mock_inference_result):
    """The API applies each food's own unvalidated diagnostic dimensions."""
    mock_detect_and_measure.return_value = mock_inference_result
    scale_map_payload = {
        "burger": {"thickness": 5.0, "diameter": 10.0},
        "fries": {"thickness": 1.0, "diameter": 2.0},
    }

    response = client.post(
        "/detect",
        files={"file": ("test.jpg", io.BytesIO(b"fake image bytes"), "image/jpeg")},
        data={
            "labels": "burger,fries",
            "scale_map": json.dumps(scale_map_payload),
        },
        headers={"x-api-token": "test-token"},
    )

    assert response.status_code == 200
    foods = response.json()["foods"]
    burger = next(food for food in foods if food["label"] == "burger")
    fries = next(food for food in foods if food["label"] == "fries")

    assert burger["volume_status"] == "unvalidated"
    assert burger["volume_cm3"] is None
    assert burger["experimental_volume_cm3"] is None
    assert abs(burger["experimental_scale_factor_cm_per_px"] - 0.1) < 0.0001

    assert fries["volume_status"] == "unvalidated"
    assert fries["volume_cm3"] is None
    assert fries["experimental_volume_cm3"] is None
    assert abs(fries["experimental_scale_factor_cm_per_px"] - 0.02) < 0.0001

    assert (
        burger["experimental_scale_factor_cm_per_px"]
        != fries["experimental_scale_factor_cm_per_px"]
    )
    assert response.json()["depth_unit"] == "relative"


@patch("vision_service.inference.api.detect_and_measure")
def test_include_arrays_keeps_visualization_mask_png_compatible(
    mock_detect_and_measure,
    mock_inference_result,
):
    """Optional NPZ arrays must not overwrite the PNG consumed by the backend."""
    mock_detect_and_measure.return_value = mock_inference_result

    response = client.post(
        "/detect",
        files={"file": ("test.jpg", io.BytesIO(b"fake image bytes"), "image/jpeg")},
        data={"labels": "burger", "include_arrays": "true"},
        headers={"x-api-token": "test-token"},
    )

    assert response.status_code == 200
    food = response.json()["foods"][0]

    png_bytes = base64.b64decode(food["mask_b64"])
    decoded_png = cv2.imdecode(
        np.frombuffer(png_bytes, dtype=np.uint8),
        cv2.IMREAD_GRAYSCALE,
    )
    assert decoded_png is not None

    npz_bytes = base64.b64decode(food["mask_array_b64"])
    with np.load(io.BytesIO(npz_bytes)) as arrays:
        assert arrays["arr"].ndim == 2
        assert arrays["arr"].dtype == np.uint8

    depth_npz = base64.b64decode(response.json()["depth_map_b64"])
    with np.load(io.BytesIO(depth_npz)) as arrays:
        assert arrays["arr"].ndim == 2
