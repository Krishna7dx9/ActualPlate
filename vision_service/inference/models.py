"""
Model loading for the ActualPlate inference service.

Owns the lifetime of every GPU model the service uses:
- Grounding DINO (open-vocabulary object detection)
- SAM2 (image segmentation)
- MoGe-2 (metric monocular depth - MIT License)

Models are loaded lazily on first access and cached for the process
lifetime. Loading is expensive (~1–3 minutes on a cold GPU host), so
the service loads once and serves many requests.
"""

from __future__ import annotations

from typing import Any

import torch

from vision_service.settings import settings

# --------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GROUNDING_DINO_MODEL_ID = settings.grounding_dino_model_id
SAM2_MODEL_ID = settings.sam2_model_id
MOGE_MODEL_ID = settings.moge_model_id

# --------------------------------------------------------------------
# Lazy-loaded singletons
# --------------------------------------------------------------------
_processor: Any = None
_grounding_dino: Any = None
_sam2_predictor: Any = None
_moge_model: Any = None


def get_grounding_dino() -> tuple[Any, Any]:
    global _processor, _grounding_dino
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
    if _sam2_predictor is None:
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        _sam2_predictor = SAM2ImagePredictor.from_pretrained(SAM2_MODEL_ID)
    return _sam2_predictor


def get_moge() -> Any:
    """
    Load (once) and return the MoGe-2 model.
    MoGe-2 predicts affine-invariant 3D point maps and metric depth.
    """
    global _moge_model
    if _moge_model is None:
        from moge.model.v2 import MoGeModel
        _moge_model = MoGeModel.from_pretrained(MOGE_MODEL_ID).to(DEVICE)
        _moge_model.eval()
    return _moge_model


def models_loaded() -> bool:
    """Used by /health to report readiness."""
    return all(
        obj is not None
        for obj in (_processor, _grounding_dino, _sam2_predictor, _moge_model)
    )