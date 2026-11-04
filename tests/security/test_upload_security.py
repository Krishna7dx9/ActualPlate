import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from backend.app.main import app

client = TestClient(app)

@patch("backend.app.services.pipeline.process_image")
def test_reject_missing_content_type(mock_process_image):
    """SECURITY TEST: Rejects files with no MIME type."""
    # Simulate a file upload with an empty content type
    response = client.post(
        "/upload-image",
        files={"file": ("test.jpg", b"fake_image_data", "")}
    )
    
    assert response.status_code == 400
    assert response.json() == {"error": "Only image files allowed"}
    # Prove the pipeline was never triggered (saving GPU money)
    mock_process_image.assert_not_called()

@patch("backend.app.services.pipeline.process_image")
def test_reject_spoofed_pdf(mock_process_image):
    """SECURITY TEST: Rejects a PDF disguised as an image upload."""
    response = client.post(
        "/upload-image",
        files={"file": ("invoice.pdf", b"%PDF-1.4 fake pdf data", "application/pdf")}
    )
    
    assert response.status_code == 400
    assert response.json() == {"error": "Only image files allowed"}
    mock_process_image.assert_not_called()

@patch("backend.app.services.pipeline.process_image")
def test_reject_malicious_executable(mock_process_image):
    """SECURITY TEST: Rejects executable files."""
    response = client.post(
        "/upload-image",
        files={"file": ("virus.exe", b"MZ\x90\x00", "application/x-msdownload")}
    )
    
    assert response.status_code == 400
    assert response.json() == {"error": "Only image files allowed"}
    mock_process_image.assert_not_called()

@patch("backend.app.services.pipeline.process_image")
def test_reject_text_file(mock_process_image):
    """SECURITY TEST: Rejects raw text files."""
    response = client.post(
        "/upload-image",
        files={"file": ("notes.txt", b"hello world", "text/plain")}
    )
    
    assert response.status_code == 400
    assert response.json() == {"error": "Only image files allowed"}
    mock_process_image.assert_not_called()