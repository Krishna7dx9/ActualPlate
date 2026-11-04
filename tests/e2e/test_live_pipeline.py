import os
import pytest
import requests
from pathlib import Path

@pytest.mark.e2e
def test_live_pipeline_end_to_end():
    """
    E2E TEST: Hits the live Kaggle GPU backend to verify the entire pipeline.
    Run manually with: pytest tests/e2e/ -v -m e2e
    """
    # 1. Get the live URL from the environment
    # (You must set VISION_INFERENCE_ENDPOINT in your terminal before running this)
    url = os.environ.get("VISION_INFERENCE_ENDPOINT")
    if not url:
        pytest.skip("VISION_INFERENCE_ENDPOINT not set. Skipping E2E test.")

    # 2. Use a real image for the test
    img_path = Path("C:/Users/Krish Sharma/OneDrive/Desktop/images.jpg")
    
    if not img_path.exists():
        pytest.skip(f"Test image not found at {img_path}. Skipping E2E test.")

    # 3. Make the real HTTP request to the live backend
    with open(img_path, "rb") as f:
        response = requests.post(
            url,
            files={"file": (img_path.name, f, "image/jpeg")},
            timeout=300  # Long timeout for cold GPU starts
        )

    # 4. Assertions
    assert response.status_code == 200, f"API returned {response.status_code}: {response.text}"
    
    data = response.json()
    assert "geometry" in data, "Response missing 'geometry' key"
    
    # If foods were detected, verify the math ran
    if data["geometry"]:
        first_food = data["geometry"][0]
        assert "volume_cm3" in first_food, "Geometry missing volume calculation"
        assert first_food["volume_status"] == "valid", "Volume math failed"