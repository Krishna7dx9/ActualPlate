"""Compatibility exports for the centralized vision settings."""

from vision_service.settings import settings

VISION_INFERENCE_ENDPOINT = settings.inference_endpoint or ""
GROUNDING_DINO_BOX_THRESHOLD = settings.box_threshold
GROUNDING_DINO_TEXT_THRESHOLD = settings.text_threshold
