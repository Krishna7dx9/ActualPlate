import os
import json
import requests

from vision_service.config import VISION_INFERENCE_ENDPOINT
from vision_service.detection.detector import BaseDetector


class GroundingDINODetector(BaseDetector):

    def __init__(self):
        self.endpoint = f"{VISION_INFERENCE_ENDPOINT}/detect"
        print(self.endpoint)

    def detect(
        self,
        image_path: str,
        labels: list[str],
        scale_map: dict | None = None,
        debug: bool = False,
    ):
        
        # Build the data payload for the Vision Worker
        data = {
            "labels": ",".join(labels),
            "debug": "true" if debug else "false"
        }
        
        # Pass Gemma's real-world scale map to the worker for the 3D math
        if scale_map:
            data["scale_map"] = json.dumps(scale_map)

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
                timeout=300
            )

        response.raise_for_status()

        return response.json()