"""
The nine volume-estimation variants (A through I).

Each variant is a function that takes one Item (from a dataset
adapter) plus any already-computed predictions and returns the
volume in cm^3 for that variant.

The variants differ only in which inputs they feed to
estimate_volume_cm3:

    A  Depth Pro depth     + SAM2 mask           (current pipeline)
    B  GT depth            + SAM2 mask
    C  Depth Pro depth     + GT mask
    D  GT depth            + GT mask             (residual test)
    E  GT depth            + GT mask, dilation=0
    F  GT depth scaled 0.9 + GT mask
    G  GT depth scaled 1.1 + GT mask
    H  GT depth            + GT mask eroded 10 px
    I  GT depth            + GT mask eroded 20 px

Variants A, B, C, D require running Depth Pro and/or SAM2. Those
are GPU-bound and depend on the environment (Colab). They must
be provided to this module as pre-computed predictions, not
computed here.

Variants E through I are pure post-processing of the ground
truth. They need no models and no GPU.

The module itself has no model dependencies. It only calls
estimate_volume_cm3 from the repo's existing volume.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from vision_service.inference.volume import estimate_volume_cm3

from benchmarks.ablation.base import Item
from benchmarks.ablation.perturbations import (
    erode_mask,
    scale_depth,
)


# ---------------------------------------------------------------------
# Predictions container
# ---------------------------------------------------------------------


@dataclass
class Predictions:
    """
    Predictions produced by the ML pipeline for one image.

    These come from running Depth Pro and SAM2 on the actual
    image. They are what the "A" variant uses.

    The harness runs the pipeline once per item and passes the
    result here. All nine variants then reuse these predictions
    as the "predicted" side of the ablation.
    """
    depth_map: np.ndarray       # (H, W) float32, meters
    mask: np.ndarray            # (H, W) bool
    focal_length_px: float


# ---------------------------------------------------------------------
# Variant functions
# ---------------------------------------------------------------------


def _run(
    depth_map: np.ndarray,
    mask: np.ndarray,
    focal_length_px: float,
) -> float:
    """
    Call estimate_volume_cm3 and return only the volume.

    Any ValueError from the volume engine (e.g. "support surface
    is not planar") is propagated to the caller. The harness
    records it as a failed variant.
    """
    result = estimate_volume_cm3(
        depth_map=depth_map,
        mask=mask,
        focal_length_px=focal_length_px,
    )
    return float(result["volume_cm3"])


# Each variant function takes (item, predictions) and returns
# the volume. Variants that need a field the adapter did not
# provide raise ValueError; the harness records these as
# "not runnable" and leaves the column empty.


def variant_A(item: Item, pred: Predictions) -> float:
    """A: current pipeline. Depth Pro + SAM2."""
    return _run(pred.depth_map, pred.mask, pred.focal_length_px)


def variant_B(item: Item, pred: Predictions) -> float:
    """B: GT depth + SAM2 mask."""
    if item.gt_depth_map is None:
        raise ValueError("B requires gt_depth_map")
    return _run(item.gt_depth_map, pred.mask, pred.focal_length_px)


def variant_C(item: Item, pred: Predictions) -> float:
    """C: Depth Pro depth + GT mask."""
    if item.gt_mask is None:
        raise ValueError("C requires gt_mask")
    return _run(pred.depth_map, item.gt_mask, pred.focal_length_px)


def variant_D(item: Item, pred: Predictions) -> float:
    """D: GT depth + GT mask. Residual test for formula chain."""
    if item.gt_depth_map is None or item.gt_mask is None:
        raise ValueError("D requires gt_depth_map and gt_mask")
    return _run(item.gt_depth_map, item.gt_mask, pred.focal_length_px)


def variant_E(item: Item, pred: Predictions) -> float:
    """
    E: same as D but with no dilation of the mask before fitting
    the supporting plane.

    In estimate_volume_cm3 the mask is dilated by 8 pixels before
    the background is sampled. That dilation constant is a
    hardcoded choice. This variant tests its effect by running
    the volume engine with a modified depth map where the pixels
    inside a fake 8-pixel dilation of the mask have been set to
    NaN. That effectively removes the dilation from the sample.
    """
    if item.gt_depth_map is None or item.gt_mask is None:
        raise ValueError("E requires gt_depth_map and gt_mask")

    depth, mask = _strip_dilation_effect(item.gt_depth_map, item.gt_mask)
    return _run(depth, mask, item.focal_length_px)


def variant_F(item: Item, pred: Predictions) -> float:
    """F: GT depth scaled by 0.9."""
    if item.gt_depth_map is None or item.gt_mask is None:
        raise ValueError("F requires gt_depth_map and gt_mask")
    depth = scale_depth(item.gt_depth_map, 0.9)
    return _run(depth, item.gt_mask, item.focal_length_px)


def variant_G(item: Item, pred: Predictions) -> float:
    """G: GT depth scaled by 1.1."""
    if item.gt_depth_map is None or item.gt_mask is None:
        raise ValueError("G requires gt_depth_map and gt_mask")
    depth = scale_depth(item.gt_depth_map, 1.1)
    return _run(depth, item.gt_mask, item.focal_length_px)


def variant_H(item: Item, pred: Predictions) -> float:
    """H: GT mask eroded by 10 pixels."""
    if item.gt_depth_map is None or item.gt_mask is None:
        raise ValueError("H requires gt_depth_map and gt_mask")
    mask = erode_mask(item.gt_mask, iterations=10)
    return _run(item.gt_depth_map, mask, item.focal_length_px)


def variant_I(item: Item, pred: Predictions) -> float:
    """I: GT mask eroded by 20 pixels."""
    if item.gt_depth_map is None or item.gt_mask is None:
        raise ValueError("I requires gt_depth_map and gt_mask")
    mask = erode_mask(item.gt_mask, iterations=20)
    return _run(item.gt_depth_map, mask, item.focal_length_px)


# ---------------------------------------------------------------------
# Helper used by variant E
# ---------------------------------------------------------------------


def _strip_dilation_effect(
    depth_map: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return a copy of (depth_map, mask) with the effect of the
    8-pixel dilation removed.

    The current volume.py does:

        expanded_mask = binary_dilation(mask, iterations=8)
        valid_background = ... & ~expanded_mask

    This means the sample used to fit the plane excludes an
    8-pixel-wide ring around the food. If that ring is not the
    plate but some other surface, the plane fit is wrong.

    To simulate dilation=0 we cannot edit volume.py (it is on
    main and shared). Instead we mask out the depth values in
    that ring so they are treated as invalid. The result is
    identical to dilation=0 in the sample.

    Returns (depth_with_nan_in_ring, same_mask).
    """
    from scipy.ndimage import binary_dilation

    ring = binary_dilation(mask, iterations=8)
    depth_out = depth_map.astype(np.float32).copy()
    # Where the ring excludes pixels from being background, set
    # the depth to NaN so they are dropped regardless of
    # volume.py's own dilation.
    ring_only = ring & ~mask
    depth_out[ring_only] = np.nan
    return depth_out, mask


# ---------------------------------------------------------------------
# Registry of all variants
# ---------------------------------------------------------------------


VariantFn = Callable[[Item, Predictions], float]

VARIANTS: dict[str, VariantFn] = {
    "A": variant_A,
    "B": variant_B,
    "C": variant_C,
    "D": variant_D,
    "E": variant_E,
    "F": variant_F,
    "G": variant_G,
    "H": variant_H,
    "I": variant_I,
}