"""Thread-safe lazy model loading for the ActualPlate inference service.

MoGe-2 supplies monocular geometry/depth; it does not independently establish
absolute metric scale for an individual image.
"""
from __future__ import annotations

import threading
from typing import Any

import torch

from vision_service.settings import settings

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GROUNDING_DINO_MODEL_ID = settings.grounding_dino_model_id
SAM2_MODEL_ID = settings.sam2_model_id
MOGE_MODEL_ID = settings.moge_model_id

_processor: Any = None
_grounding_dino: Any = None
_sam2_predictor: Any = None
_moge_model: Any = None
_MODEL_LOAD_LOCK = threading.RLock()


def get_grounding_dino() -> tuple[Any, Any]:
    global _processor, _grounding_dino
    with _MODEL_LOAD_LOCK:
        if _grounding_dino is None:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            _processor = AutoProcessor.from_pretrained(GROUNDING_DINO_MODEL_ID)
            _grounding_dino = (
                AutoModelForZeroShotObjectDetection
                .from_pretrained(GROUNDING_DINO_MODEL_ID)
                .to(DEVICE)
            )
            _grounding_dino.eval()
        return _processor, _grounding_dino


def get_sam2_predictor() -> Any:
    global _sam2_predictor
    with _MODEL_LOAD_LOCK:
        if _sam2_predictor is None:
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            _sam2_predictor = SAM2ImagePredictor.from_pretrained(SAM2_MODEL_ID)
        return _sam2_predictor


def get_moge() -> Any:
    """Load MoGe-2 geometry/depth without asserting absolute metric scale."""
    global _moge_model
    with _MODEL_LOAD_LOCK:
        if _moge_model is None:
            from moge.model.v2 import MoGeModel
            _moge_model = MoGeModel.from_pretrained(MOGE_MODEL_ID).to(DEVICE)
            _moge_model.eval()
        return _moge_model


def models_loaded() -> bool:
    """Return whether every inference model has completed initialization."""
    return all(
        obj is not None
        for obj in (_processor, _grounding_dino, _sam2_predictor, _moge_model)
    )
