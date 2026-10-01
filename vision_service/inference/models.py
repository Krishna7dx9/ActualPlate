"""
Model loading for the ActualPlate inference service.

Owns the lifetime of every GPU model the service uses:

- Grounding DINO (open-vocabulary object detection)
- SAM2 (image segmentation)
- Apple Depth Pro (metric monocular depth)

Models are loaded lazily on first access and cached for the process
lifetime. Loading is expensive (~1–3 minutes on a cold GPU host), so
the service loads once and serves many requests.

This module owns no HTTP, no FastAPI, and no request handling. It only
loads models and exposes them.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from typing import Any

import torch


# --------------------------------------------------------------------
# Configuration
#
# Defaults match the notebook. Every value can be overridden via
# environment variables so the same module works on Colab, RunPod,
# GCP, or a local GPU without code changes.
# --------------------------------------------------------------------

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

GROUNDING_DINO_MODEL_ID = os.environ.get(
    "GROUNDING_DINO_MODEL_ID",
    "IDEA-Research/grounding-dino-base",
)

SAM2_MODEL_ID = os.environ.get(
    "SAM2_MODEL_ID",
    "facebook/sam2-hiera-large",
)

DEPTH_PRO_REPO_URL = os.environ.get(
    "DEPTH_PRO_REPO_URL",
    "https://github.com/apple/ml-depth-pro.git",
)

DEPTH_PRO_CHECKPOINT_URL = os.environ.get(
    "DEPTH_PRO_CHECKPOINT_URL",
    "https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt",
)

DEPTH_PRO_WORKDIR = os.environ.get(
    "DEPTH_PRO_WORKDIR",
    "/content/ml-depth-pro",
)

DEPTH_PRO_CHECKPOINT_PATH = os.environ.get(
    "DEPTH_PRO_CHECKPOINT_PATH",
    os.path.join(DEPTH_PRO_WORKDIR, "checkpoints", "depth_pro.pt"),
)


# --------------------------------------------------------------------
# Lazy-loaded singletons
#
# These are module-private. Access them only through the getters below.
# --------------------------------------------------------------------

_processor: Any = None
_grounding_dino: Any = None
_sam2_predictor: Any = None
_depth_pro_module: Any = None
_depth_model: Any = None
_depth_transform: Any = None


def get_grounding_dino() -> tuple[Any, Any]:
    """
    Load (once) and return (processor, grounding_dino).

    Grounding DINO is an open-vocabulary detector: it takes an image and
    a text prompt and returns bounding boxes with text labels and scores.
    """
    global _processor, _grounding_dino

    if _grounding_dino is None:
        from transformers import (
            AutoProcessor,
            AutoModelForZeroShotObjectDetection,
        )

        _processor = AutoProcessor.from_pretrained(
            GROUNDING_DINO_MODEL_ID,
        )
        _grounding_dino = (
            AutoModelForZeroShotObjectDetection
            .from_pretrained(GROUNDING_DINO_MODEL_ID)
            .to(DEVICE)
        )
        _grounding_dino.eval()

    return _processor, _grounding_dino


def get_sam2_predictor() -> Any:
    """
    Load (once) and return the SAM2 image predictor.

    SAM2 turns a bounding box into a pixel-accurate segmentation mask.
    """
    global _sam2_predictor

    if _sam2_predictor is None:
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        _sam2_predictor = (
            SAM2ImagePredictor
            .from_pretrained(SAM2_MODEL_ID)
        )

    return _sam2_predictor


def _ensure_depth_pro_installed() -> Any:
    """
    Clone, install, and import Apple's ml-depth-pro package.

    The package is not on PyPI; it must be cloned and installed from
    source. The checkpoint is downloaded separately.
    """
    if not os.path.isdir(DEPTH_PRO_WORKDIR):
        subprocess.run(
            ["git", "clone", "-q", DEPTH_PRO_REPO_URL, DEPTH_PRO_WORKDIR],
            check=True,
        )
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-e", "."],
            cwd=DEPTH_PRO_WORKDIR,
            check=True,
        )

    src_path = os.path.join(DEPTH_PRO_WORKDIR, "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    checkpoint_dir = os.path.dirname(DEPTH_PRO_CHECKPOINT_PATH)
    os.makedirs(checkpoint_dir, exist_ok=True)

    if not os.path.isfile(DEPTH_PRO_CHECKPOINT_PATH):
        subprocess.run(
            [
                "wget",
                "-q",
                DEPTH_PRO_CHECKPOINT_URL,
                "-O",
                DEPTH_PRO_CHECKPOINT_PATH,
            ],
            check=True,
        )

    import depth_pro  # type: ignore

    return depth_pro


def get_depth_pro() -> tuple[Any, Any, Any]:
    """
    Load (once) and return (depth_pro_module, depth_model, depth_transform).

    Apple Depth Pro produces a metric depth map (in meters) and an
    estimated focal length in pixels from a single RGB image.
    """
    global _depth_pro_module, _depth_model, _depth_transform

    if _depth_model is None:
        depth_pro = _ensure_depth_pro_installed()

        from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT

        config = DEFAULT_MONODEPTH_CONFIG_DICT
        config.checkpoint_uri = DEPTH_PRO_CHECKPOINT_PATH

        _depth_model, _depth_transform = (
            depth_pro.create_model_and_transforms(
                config=config,
                device=DEVICE,
            )
        )
        _depth_model.eval()
        _depth_pro_module = depth_pro

    return _depth_pro_module, _depth_model, _depth_transform


def models_loaded() -> bool:
    """
    Return True if every model has already been loaded into memory.

    Used by /health to report readiness without triggering lazy loads.
    """
    return all(
        obj is not None
        for obj in (
            _processor,
            _grounding_dino,
            _sam2_predictor,
            _depth_model,
            _depth_transform,
        )
    )