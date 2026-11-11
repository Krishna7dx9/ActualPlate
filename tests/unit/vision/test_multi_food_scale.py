import pytest
import json
import io
from fastapi.testclient import TestClient
from unittest.mock import patch
import numpy as np
import os

# We must set the token before importing the app
os.environ["VISION_API_TOKEN"] = "test-token"

from vision_service.inference.api import app
from vision_service.inference.inference import InferenceResult, FoodDetection

client = TestClient(app)

@pytest.fixture
def mock_inference_result():
    """Mocks the output of the heavy GPU pipeline."""
    food1 = FoodDetection(
        label="burger",
        score=0.95,
        mask_pixels=250000, # 500x500 area
        depth_stats={"min": 1.0, "max": 1.5},
        bbox=[10.0, 10.0, 110.0, 110.0] # 100x100 box
    )
    food2 = FoodDetection(
        label="fries",
        score=0.90,
        mask_pixels=250000, 
        depth_m={"min": 1.2, "max": 1.3},
        bbox=[200.0, 200.0, 300.0, 300.0] # 100x100 box
    )
    
    # FIX: Make the masks 500x500 to match the depth map
    mask1 = np.ones((500, 500), dtype=bool)
    mask2 = np.ones((500, 500), dtype=bool)
    
    return InferenceResult(
        foods=[food1, food2],
        image_width=500,
        image_height=500,
        focal_length_px=800.0,
        depth_unit="relative",
        depth_map=np.ones((500, 500), dtype=np.float32), 
        masks=[mask1, mask2]
    )

@patch("vision_service.inference.api.detect_and_measure")
def test_api_applies_per_food_scale_map(mock_detect_and_measure, mock_inference_result):
    """Test that the API correctly parses scale_map and applies correct dimensions."""
    mock_detect_and_measure.return_value = mock_inference_result

    # Build scale_map with DIFFERENT scales
    scale_map_payload = {
        "burger": {"thickness": 5.0, "diameter": 10.0},
        "fries": {"thickness": 1.0, "diameter": 2.0}
    }

    fake_image = io.BytesIO(b"fake image bytes")
    
    response = client.post(
        "/detect",
        files={"file": ("test.jpg", fake_image, "image/jpeg")},
        data={
            "labels": "burger,fries",
            "scale_map": json.dumps(scale_map_payload)
        },
        headers={"x-api-token": "test-token"}
    )

    # Assertions
    assert response.status_code == 200
    data = response.json()
    foods = data["foods"]
    
    assert len(foods) == 2
    
    burger = next(f for f in foods if f["label"] == "burger")
    fries = next(f for f in foods if f["label"] == "fries")

    # VLM-provided dimensions may drive diagnostics, but never a metric result.
    assert burger["volume_status"] == "unvalidated"
    assert burger["volume_cm3"] is None
    assert burger["experimental_volume_cm3"] is None
    assert abs(burger["experimental_scale_factor_cm_per_px"] - 0.1) < 0.0001

    assert fries["volume_status"] == "unvalidated"
    assert fries["volume_cm3"] is None
    assert fries["experimental_volume_cm3"] is None
    assert abs(fries["experimental_scale_factor_cm_per_px"] - 0.02) < 0.0001

    # Each food keeps its own diagnostic scale, but neither is certified metric.
    assert (
        burger["experimental_scale_factor_cm_per_px"]
        != fries["experimental_scale_factor_cm_per_px"]
    )
    assert data["depth_unit"] == "relative"
    print("\n✅ TEST PASSED: Burger and Fries received different scale factors.")