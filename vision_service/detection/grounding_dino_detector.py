import os
import json
import logging

import requests

from vision_service.config import VISION_INFERENCE_ENDPOINT
from vision_service.detection.detector import BaseDetector


logger = logging.getLogger(__name__)


class GroundingDINODetector(BaseDetector):
    """HTTP client for the separately hosted vision inference service."""

    def __init__(self):
        if not VISION_INFERENCE_ENDPOINT:
            raise RuntimeError(
                "VISION_INFERENCE_ENDPOINT must be configured for the backend."
            )

        self.endpoint = f"{VISION_INFERENCE_ENDPOINT.rstrip('/')}/detect"
        self.api_token = os.environ.get("VISION_API_TOKEN")
        self.app_environment = os.environ.get(
            "APP_ENV",
            "development",
        ).strip().lower()

        if self.app_environment == "production" and not self.api_token:
            raise RuntimeError(
                "VISION_API_TOKEN must be configured when APP_ENV=production."
            )

        logger.info("Vision inference endpoint configured")

    def detect(
        self,
        image_path: str,
        labels: list[str],
        scale_map: dict | None = None,
        debug: bool = False,
    ):
        data = {
            "labels": ",".join(labels),
            "debug": "true" if debug else "false",
        }

        if scale_map:
            data["scale_map"] = json.dumps(scale_map)

        headers = {}
        if self.api_token:
            headers["x-api-token"] = self.api_token

        with open(image_path, "rb") as image:
            response = requests.post(
                self.endpoint,
                files={
                    "file": (
                        os.path.basename(image_path),
                        image,
                        "image/jpeg",
                    )
                },
                data=data,
                headers=headers,
                timeout=300,
            )

        response.raise_for_status()
        return response.json()
