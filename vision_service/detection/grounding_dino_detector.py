import os
import json
import requests

from vision_service.config import VISION_INFERENCE_ENDPOINT
from vision_service.detection.detector import BaseDetector


class GroundingDINODetector(BaseDetector):

    def __init__(self):
        self.endpoint = f"{VISION_INFERENCE_ENDPOINT}/detect"
        self.api_token = os.environ.get("VISION_API_TOKEN")

        if not self.api_token:
            raise ValueError(
                "VISION_API_TOKEN must be set for the backend to call the vision worker."
            )

        print(f"GroundingDINO endpoint: {self.endpoint}")

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

        headers = {
            "x-api-token": self.api_token,
        }

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