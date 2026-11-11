from fastapi.testclient import TestClient
from io import BytesIO
from PIL import Image
from unittest.mock import patch, MagicMock, AsyncMock
from backend.app.main import app

client = TestClient(app)


def _valid_png_bytes():
    image_buffer = BytesIO()
    Image.new("RGB", (8, 8)).save(image_buffer, format="PNG")
    return image_buffer.getvalue()

@patch("requests.post")
@patch("backend.app.services.pipeline.detect_food")
def test_single_food(mock_detect_food, mock_requests_post):
    mock_detect_food.return_value = [
        {"search_query": "pizza", "density_g_cm3": 1.0, "estimated_thickness_cm": 2.0, "estimated_diameter_cm": 10.0}
    ]
    
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "foods": [{"label": "pizza", "volume_cm3": 250.0, "bbox": [10, 10, 100, 100], "mask_b64": "fake"}],
        "image_width": 100, "image_height": 100, "focal_length_px": 50, "depth_unit": "meters", "depth_b64": "fake"
    }
    mock_requests_post.return_value = mock_response

    response = client.post(
        "/upload-image",
        files={"file": ("pizza.png", _valid_png_bytes(), "image/png")}
    )
    assert response.status_code == 200

def test_invalid_file():
    # This now hits the validation in upload.py, so it doesn't need mocks
    response = client.post(
        "/upload-image",
        files={"file": ("doc.pdf", b"fake", "application/pdf")}
    )
    assert response.status_code == 400
    assert response.json() == {"error": "Only image files allowed"}


def test_unhandled_exception_does_not_expose_internal_details():
    """Public API errors must not disclose exception messages."""
    safe_client = TestClient(app, raise_server_exceptions=False)

    with patch(
        "backend.app.api.routes.upload.process_image",
        new_callable=AsyncMock,
        side_effect=RuntimeError("upstream credential leaked"),
    ):
        response = safe_client.post(
            "/upload-image",
            files={"file": ("image.png", _valid_png_bytes(), "image/png")},
        )

    assert response.status_code == 500
    assert response.json() == {"detail": "Internal Server Error"}
    assert "credential" not in response.text
