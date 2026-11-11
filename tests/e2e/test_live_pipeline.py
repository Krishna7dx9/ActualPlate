"""Opt-in smoke test for a deployed ActualPlate backend.

Configure ACTUALPLATE_E2E_URL with the backend's /upload-image URL and
ACTUALPLATE_E2E_IMAGE_PATH with a representative local image before running:
    pytest tests/e2e/ -m e2e -v

The test is intentionally skipped in ordinary CI because it requires a live
GPU-backed deployment and a real image fixture.
"""

from __future__ import annotations

import math
import mimetypes
import os
from pathlib import Path
from urllib.parse import urlparse

import pytest
import requests


@pytest.mark.e2e
def test_live_pipeline_end_to_end() -> None:
    endpoint = os.environ.get("ACTUALPLATE_E2E_URL", "").strip()
    image_path_value = os.environ.get("ACTUALPLATE_E2E_IMAGE_PATH", "").strip()

    if not endpoint or not image_path_value:
        pytest.skip(
            "Set ACTUALPLATE_E2E_URL and ACTUALPLATE_E2E_IMAGE_PATH "
            "to run the live pipeline smoke test."
        )

    parsed_endpoint = urlparse(endpoint)
    if parsed_endpoint.scheme not in {"http", "https"} or not parsed_endpoint.netloc:
        pytest.fail("ACTUALPLATE_E2E_URL must be an absolute HTTP(S) URL.")

    image_path = Path(image_path_value).expanduser()
    if not image_path.is_file():
        pytest.fail("ACTUALPLATE_E2E_IMAGE_PATH must point to an existing image file.")

    content_type, _ = mimetypes.guess_type(image_path.name)
    if not content_type or not content_type.startswith("image/"):
        pytest.fail("The configured E2E fixture must have a recognized image extension.")

    raw_timeout = os.environ.get("ACTUALPLATE_E2E_TIMEOUT_SECONDS", "300")
    try:
        timeout_seconds = float(raw_timeout)
    except ValueError:
        pytest.fail("ACTUALPLATE_E2E_TIMEOUT_SECONDS must be a positive number.")
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        pytest.fail("ACTUALPLATE_E2E_TIMEOUT_SECONDS must be a positive number.")

    with image_path.open("rb") as image_file:
        response = requests.post(
            endpoint,
            files={"file": (image_path.name, image_file, content_type)},
            timeout=timeout_seconds,
        )

    assert response.status_code == 200, (
        f"Backend returned HTTP {response.status_code}: {response.text[:1000]}"
    )

    payload = response.json()
    assert isinstance(payload, dict), "Backend response must be a JSON object."
    geometry = payload.get("geometry")
    assert isinstance(geometry, list), "Backend response must include a geometry list."
    assert geometry, "The configured E2E image must contain at least one detected food."

    for item in geometry:
        assert isinstance(item, dict), "Each geometry record must be a JSON object."
        assert "volume_cm3" in item, "Geometry record is missing volume_cm3."
        assert item["volume_cm3"] is None, (
            "Unvalidated monocular scale must not be exposed as physical volume."
        )
        assert item.get("volume_status") in {"unvalidated", "unavailable"}, (
            "Geometry must explicitly report that metric volume is not validated."
        )
