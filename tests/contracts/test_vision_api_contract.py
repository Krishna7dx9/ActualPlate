import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import json
import os
import numpy as np

# Set the token before importing the app to pass startup validation
os.environ["VISION_API_TOKEN"] = "test-token"

from vision_service.inference.api import app

client = TestClient(app)

@patch("vision_service.inference.api.detect_and_measure")
def test_vision_api_accepts_backend_payload(mock_detect_and_measure):
    """
    CONTRACT TEST: Simulates the exact payload sent by grounding_dino_detector.py
    and verifies the Vision API accepts it without a 422 Unprocessable Entity error.
    """
    # 1. Setup realistic Mock
    mock_result = MagicMock()
    mock_result.foods = []
    mock_result.masks = []
    mock_result.image_width = 100
    mock_result.image_height = 100
    mock_result.focal_length_px = 50.0
    mock_result.depth_unit = "meters"
    # THE FIX: Provide a real numpy array instead of None
    mock_result.depth_map = np.ones((100, 100), dtype=np.float32)
    
    mock_detect_and_measure.return_value = mock_result

    # 2. Simulate the exact payload sent by GroundingDINO Detector
    scale_map_payload = {
        "burger": {"thickness": 5.0, "diameter": 10.0},
        "fries": {"thickness": 1.0, "diameter": 2.0}
    }
    
    payload_data = {
        "labels": "burger,fries",
        "debug": "true",
        "scale_map": json.dumps(scale_map_payload) # Backend serializes this
    }
    
    files = {"file": ("test.jpg", b"fake_image_bytes", "image/jpeg")}
    headers = {"x-api-token": "test-token"}

    # 3. Act
    response = client.post("/detect", data=payload_data, files=files, headers=headers)

    # 4. Assert
    assert response.status_code == 200
    assert response.json()["request_id"] is not None

@patch("vision_service.inference.api.detect_and_measure")
def test_vision_api_rejects_invalid_scale_map_json(mock_detect_and_measure):
    """
    CONTRACT TEST: Ensures the API gracefully handles malformed JSON strings
    instead of crashing with an internal server error.
    """
    payload_data = {
        "labels": "burger,fries",
        "scale_map": "{invalid_json_string}" # Malformed JSON
    }
    
    files = {"file": ("test.jpg", b"fake_image_bytes", "image/jpeg")}
    headers = {"x-api-token": "test-token"}

    response = client.post("/detect", data=payload_data, files=files, headers=headers)

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid scale_map JSON format."

def test_vision_api_rejects_invalid_token():
    """
    SECURITY CONTRACT TEST: Ensures the endpoint blocks unauthorized requests.
    """
    payload_data = {"labels": "burger"}
    files = {"file": ("test.jpg", b"fake_image_bytes", "image/jpeg")}
    headers = {"x-api-token": "WRONG-TOKEN"}

    response = client.post("/detect", data=payload_data, files=files, headers=headers)

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid API token."