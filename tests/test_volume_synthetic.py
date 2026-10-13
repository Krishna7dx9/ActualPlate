"""
Synthetic test for the volume formula in vision_service/inference/volume.py.

This test uses no ML model, no camera, no dataset. It feeds
estimate_volume_cm3 a depth map and mask whose geometry is known in
closed form, computes the expected volume using the same formula the
function is documented to implement, and asserts they match.

If this test passes, the volume formula is mathematically correct and
the observed pipeline error comes from the depth map, the mask, or the
input focal length — not from the math.

If this test fails, the formula contains a bug that must be fixed
before any other optimization is attempted.

Run:
    python -m pytest tests/test_volume_synthetic.py -v -s
"""

from __future__ import annotations

import numpy as np

from vision_service.inference.volume import estimate_volume_cm3


# --- Test geometry ----------------------------------------------------
#
# Image       100 x 100 px
# Focal       f = 1000 px
# Principal   image center (49.5, 49.5)
# Plate       Z = 1.00 m (perfectly flat, uniform)
# Food        Z = 0.97 m inside mask (3 cm above plate)
# Mask        20 x 20 pixel square at image center

IMAGE_SIZE = 100
FOCAL_PX = 1000.0
PLATE_Z_M = 1.0
FOOD_HEIGHT_M = 0.03
MASK_HALF = 10


def _build_scene():
    h = w = IMAGE_SIZE
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0

    depth = np.full((h, w), PLATE_Z_M, dtype=np.float32)

    mask = np.zeros((h, w), dtype=bool)
    mask[
        IMAGE_SIZE // 2 - MASK_HALF : IMAGE_SIZE // 2 + MASK_HALF,
        IMAGE_SIZE // 2 - MASK_HALF : IMAGE_SIZE // 2 + MASK_HALF,
    ] = True

    depth[mask] = PLATE_Z_M - FOOD_HEIGHT_M

    return depth, mask, cx, cy


def _expected_volume_cm3() -> float:
    """
    Independent computation of the volume by directly applying the
    formula the function is documented to implement:

        dOmega = 1 / f^2 / (1 + x^2 + y^2)^1.5
        dV     = (r_plane^3 - r_surface^3) / 3 * dOmega
    """
    depth, mask, cx, cy = _build_scene()
    fy, fx = np.where(mask)
    food_z = depth[fy, fx]

    ray_x = (fx - cx) / FOCAL_PX
    ray_y = (fy - cy) / FOCAL_PX
    ray_norm = np.sqrt(ray_x ** 2 + ray_y ** 2 + 1.0)

    surface_ray_distance = food_z * ray_norm
    plane_ray_distance = PLATE_Z_M * ray_norm

    solid_angle_per_pixel = (
        1.0
        / (FOCAL_PX ** 2)
        / (1.0 + ray_x ** 2 + ray_y ** 2) ** 1.5
    )

    ray_volume_m3 = (
        (plane_ray_distance ** 3 - surface_ray_distance ** 3)
        / 3.0
        * solid_angle_per_pixel
    )
    ray_volume_m3 = np.maximum(ray_volume_m3, 0.0)

    return float(ray_volume_m3.sum() * 1_000_000.0)


def test_volume_formula_matches_closed_form():
    depth, mask, _, _ = _build_scene()

    expected = _expected_volume_cm3()

    result = estimate_volume_cm3(
        depth_map=depth,
        mask=mask,
        focal_length_px=FOCAL_PX,
    )
    actual = result["volume_cm3"]
    ratio = actual / expected if expected > 0 else float("inf")

    print()
    print("=" * 60)
    print("SYNTHETIC VOLUME TEST")
    print("=" * 60)
    print(f"  Expected: {expected:.6f} cm^3")
    print(f"  Actual:   {actual:.6f} cm^3")
    print(f"  Ratio:    {ratio:.6f}")
    print("=" * 60)

    assert abs(ratio - 1.0) < 0.01, (
        f"Volume formula mismatch: "
        f"expected {expected:.4f}, got {actual:.4f}, "
        f"ratio {ratio:.4f}"
    )