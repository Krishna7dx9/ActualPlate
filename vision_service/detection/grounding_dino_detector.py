import os

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
        labels: list[str]
    ):

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
                data={
                    "labels": ",".join(labels)
                },
                timeout=120
            )

        response.raise_for_status()

        return response.json()