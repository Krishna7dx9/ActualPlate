import os
from dotenv import load_dotenv

load_dotenv()


# Vision inference provider
# Examples:
# colab
# cloud_run
# dedicated_gpu
# self_hosted
VISION_INFERENCE_PROVIDER = os.getenv(
    "VISION_INFERENCE_PROVIDER",
    "colab"
)


# Base URL of the Vision Inference Service
# Example:
# https://xxxx.ngrok-free.app
# https://vision.actualplate.ai
VISION_INFERENCE_ENDPOINT = os.getenv(
    "VISION_INFERENCE_ENDPOINT",
    ""
)


# Grounding DINO thresholds
GROUNDING_DINO_BOX_THRESHOLD = float(
    os.getenv(
        "GROUNDING_DINO_BOX_THRESHOLD",
        "0.35"
    )
)

GROUNDING_DINO_TEXT_THRESHOLD = float(
    os.getenv(
        "GROUNDING_DINO_TEXT_THRESHOLD",
        "0.25"
    )
)