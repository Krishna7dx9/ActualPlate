from fastapi.testclient import TestClient
from io import BytesIO
from PIL import Image
from unittest.mock import patch, MagicMock
import json
from backend.app.main import app

client = TestClient(app)


def _valid_jpeg_bytes():
    image_buffer = BytesIO()
    Image.new("RGB", (8, 8)).save(image_buffer, format="JPEG")
    return image_buffer.getvalue()

@patch("requests.post")
@patch("backend.app.services.pipeline.detect_food")
def test_regression_multi_food_scale_mapping(mock_detect_food, mock_requests_post):
    """
    REGRESSION TEST: Ensures that multiple foods receive their own distinct scale 
    estimates, fixing the bug where the first food's scale was applied to all foods.
    """
    mock_detect_food.return_value = [
        {"search_query": "burger", "density_g_cm3": 1.0, "estimated_thickness_cm": 5.0, "estimated_diameter_cm": 10.0},
        {"search_query": "fries", "density_g_cm3": 1.0, "estimated_thickness_cm": 1.0, "estimated_diameter_cm": 2.0}
    ]
    
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "foods": [
            {"label": "burger", "volume_cm3": 250.0, "bbox": [0, 0, 100, 100], "mask_b64": "fake"},
            {"label": "fries", "volume_cm3": 50.0, "bbox": [100, 100, 200, 200], "mask_b64": "fake"}
        ],
        "image_width": 300, "image_height": 300, "focal_length_px": 50, "depth_unit": "meters", "depth_b64": "fake"
    }
    mock_requests_post.return_value = mock_response

    response = client.post(
        "/upload-image",
        files={"file": ("food.jpg", _valid_jpeg_bytes(), "image/jpeg")}
    )
    
    assert response.status_code == 200
    
    # FIX: Check the FIRST requests.post call (Vision Worker), not the LAST one (FatSecret)
    vision_call = mock_requests_post.call_args_list[0]
    sent_scale_map = json.loads(vision_call.kwargs["data"]["scale_map"])
    
    assert sent_scale_map["burger"]["thickness"] == 5.0
    assert sent_scale_map["fries"]["thickness"] == 1.0
    assert sent_scale_map["burger"]["diameter"] == 10.0
    assert sent_scale_map["fries"]["diameter"] == 2.0