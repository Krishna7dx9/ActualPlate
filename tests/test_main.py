from fastapi.testclient import TestClient
from unittest.mock import patch
from backend.app.main import app

client = TestClient(app)


@patch("backend.services.pipeline.detect_food")
def test_single_food(mock_detect):
    mock_detect.return_value = "pizza"
    response = client.post(
        "/upload-image",
        files={"file": ("pizza.jpg", b"fake", "image/jpeg")}
    )
    assert response.status_code == 200


@patch("backend.services.pipeline.detect_food")
def test_multiple_food(mock_detect):
    mock_detect.return_value = "pizza,burger,coke"
    response = client.post(
        "/upload-image",
        files={"file": ("food.jpg", b"fake", "image/jpeg")}
    )
    assert response.status_code == 200


@patch("backend.services.pipeline.detect_food")
def test_empty_food(mock_detect):
    mock_detect.return_value = ""
    response = client.post(
        "/upload-image",
        files={"file": ("food.jpg", b"fake", "image/jpeg")}
    )
    assert response.json()["error"] == "Food not detected"


def test_invalid_file():
    response = client.post(
        "/upload-image",
        files={"file": ("doc.pdf", b"fake", "application/pdf")}
    )
    assert response.json()["error"] == "Only image files allowed"