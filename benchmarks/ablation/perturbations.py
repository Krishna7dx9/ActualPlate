"""
Pure input-modifying functions for the ablation harness.

Each function takes a depth map or a mask and returns a new one
with a single, well-defined perturbation applied. No I/O, no
state, no randomness.

The perturbations correspond to the "sensitivity" and "residual"
variants:

    E: no dilation of the mask before fitting the plane
    F: depth scaled by 0.9
    G: depth scaled by 1.1
    H: mask eroded by 10 pixels
    I: mask eroded by 20 pixels

Variants A through D use the inputs unchanged and do not need
this module. They are handled in variants.py directly.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_erosion


def scale_depth(depth_map: np.ndarray, factor: float) -> np.ndarray:
    """
    Return a new depth map with every value multiplied by factor.

    Used for variants F and G: tests how the pipeline's volume
    output responds to a systematic depth-scale error.

    The pipeline's depth is in meters. Multiplying by 1.1 makes
    the whole scene appear 10% farther from the camera.
    """
    return (depth_map.astype(np.float32) * factor).astype(np.float32)


def erode_mask(mask: np.ndarray, iterations: int) -> np.ndarray:
    """
    Return a new binary mask with `iterations` pixels eroded from
    its boundary.

    Used for variants H and I: tests how sensitive volume is to
    mask-edge over- or under-segmentation.

    Uses a default 3x3 cross-shaped structure (scipy's default),
    which removes a 1-pixel-thick ring per iteration.
    """
    if iterations <= 0:
        return mask.astype(bool, copy=True)
    return binary_erosion(mask.astype(bool), iterations=iterations)


def dilate_mask(mask: np.ndarray, iterations: int) -> np.ndarray:
    """
    Return a new binary mask with `iterations` pixels added to its
    boundary.

    Not used by the current variants but kept for symmetry and for
    future sensitivity tests (how does volume respond if the mask
    is too generous?).
    """
    from scipy.ndimage import binary_dilation
    if iterations <= 0:
        return mask.astype(bool, copy=True)
    return binary_dilation(mask.astype(bool), iterations=iterations)


def restrict_to_depth_range(
    mask: np.ndarray,
    depth_map: np.ndarray,
    low_percentile: float = 5.0,
    high_percentile: float = 95.0,
) -> np.ndarray:
    """
    Return a new mask restricted to pixels whose depth is within
    the given percentile range of the masked depth distribution.

    Removes outliers in the mask: pixels whose depth is far from
    the rest of the food. These are usually plate pixels captured
    by an over-generous SAM2 mask.

    Not used by current variants. Kept as a candidate fix once the
    ablation tells us mask boundary is the dominant error.
    """
    depth_in_mask = depth_map[mask]
    finite = depth_in_mask[np.isfinite(depth_in_mask)]
    if finite.size == 0:
        return mask.astype(bool, copy=True)

    lo = float(np.percentile(finite, low_percentile))
    hi = float(np.percentile(finite, high_percentile))

    keep = np.zeros_like(mask, dtype=bool)
    keep[mask] = (
        np.isfinite(depth_map[mask])
        & (depth_map[mask] >= lo)
        & (depth_map[mask] <= hi)
    )
    return keep


def depth_scale_sensitivity_ratio(
    volume_at_0_9: float,
    volume_at_1_0: float,
    volume_at_1_1: float,
) -> tuple[float, float]:
    """
    Given the volumes from three scale variants, return two
    sensitivity ratios:

        ratio_up   = V(1.1) / V(1.0)
        ratio_down = V(0.9) / V(1.0)

    For a formula that behaves as z^3, both ratios should be close
    to (1.1)^3 = 1.331 and (0.9)^3 = 0.729 respectively. If the
    pipeline produces something very different, the depth
    handling is not behaving as a cubic in z.
    """
    if volume_at_1_0 <= 0:
        return (float("nan"), float("nan"))
    return (volume_at_1_1 / volume_at_1_0, volume_at_0_9 / volume_at_1_0)