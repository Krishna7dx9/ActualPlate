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

import os
import re
import threading
from typing import Any

import torch

# --------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GROUNDING_DINO_MODEL_ID = os.environ.get(
    "GROUNDING_DINO_MODEL_ID",
    "IDEA-Research/grounding-dino-base",
)

SAM2_MODEL_ID = os.environ.get(
    "SAM2_MODEL_ID",
    "facebook/sam2-hiera-large",
)

# MoGe-2 (MIT License) for metric monocular depth.
MOGE_MODEL_ID = os.environ.get(
    "MOGE_MODEL_ID",
    "Ruicheng/moge-2-vitl-normal",
)

# --------------------------------------------------------------------
# Lazy-loaded singletons
# --------------------------------------------------------------------
_processor: Any = None
_grounding_dino: Any = None
_sam2_predictor: Any = None
_moge_model: Any = None
_MODEL_LOAD_LOCK = threading.RLock()


def _required_model_revision(variable_name: str) -> str:
    revision = os.environ.get(variable_name, "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise RuntimeError(
            f"{variable_name} must be configured as a full 40-character model commit SHA."
        )
    return revision


def get_grounding_dino() -> tuple[Any, Any]:
    global _processor, _grounding_dino
    with _MODEL_LOAD_LOCK:
        if _grounding_dino is None:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            revision = _required_model_revision("GROUNDING_DINO_MODEL_REVISION")
            _processor = AutoProcessor.from_pretrained(GROUNDING_DINO_MODEL_ID, revision=revision)
            _grounding_dino = (
                AutoModelForZeroShotObjectDetection
                .from_pretrained(GROUNDING_DINO_MODEL_ID, revision=revision)
                .to(DEVICE)
            )
            _grounding_dino.eval()
        return _processor, _grounding_dino


def get_sam2_predictor() -> Any:
    global _sam2_predictor
    with _MODEL_LOAD_LOCK:
        if _sam2_predictor is None:
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            revision = _required_model_revision("SAM2_MODEL_REVISION")
            _sam2_predictor = SAM2ImagePredictor.from_pretrained(SAM2_MODEL_ID, revision=revision)
        return _sam2_predictor


def get_moge() -> Any:
    """
    Load (once) and return the MoGe-2 model.
    MoGe-2 predicts affine-invariant 3D point maps and metric depth.
    """
    global _moge_model
    with _MODEL_LOAD_LOCK:
        if _moge_model is None:
            from moge.model.v2 import MoGeModel
            revision = _required_model_revision("MOGE_MODEL_REVISION")
            _moge_model = MoGeModel.from_pretrained(MOGE_MODEL_ID, revision=revision).to(DEVICE)
            _moge_model.eval()
        return _moge_model


def models_loaded() -> bool:
    """Used by /health to report readiness."""
    return all(
        obj is not None
        for obj in (_processor, _grounding_dino, _sam2_predictor, _moge_model)
    )