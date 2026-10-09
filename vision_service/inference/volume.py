"""
Physical food-volume estimation from a monocular depth map and a food mask.

Ported from Cell 6 of the ActualPlate Colab notebook
(vision_inference_service.ipynb). The math is unchanged. This module
is pure NumPy/SciPy: it has no GPU, torch, or model dependencies, so
it can be unit-tested locally without a Colab runtime.

Public API:
    estimate_volume_cm3(depth_map, mask, focal_length_px) -> dict

The function raises ValueError when the supporting geometry is not
reliable rather than returning a fabricated volume. Callers must
decide how to surface that failure.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation


def _plane_from_points(
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
) -> tuple[np.ndarray, float] | None:
    """
    Return (unit_normal, offset) of the plane through three 3D points.

    Returns None if the points are collinear (normal degenerates).
    The plane satisfies: normal · x = offset.
    """
    v1 = p2 - p1
    v2 = p3 - p1

    normal = np.cross(v1, v2)
    length = float(np.linalg.norm(normal))

    if length < 1e-8:
        return None

    normal = normal / length
    offset = float(np.dot(normal, p1))

    return normal, offset


def _refine_plane(
    points: np.ndarray,
    normal: np.ndarray,
    offset: float,
) -> tuple[np.ndarray, float, int]:
    """
    Refine a plane fit using its inliers.

    Inliers are selected with a MAD-based threshold, then the plane
    is refit by SVD on the inlier points. If fewer than 100 inliers
    are found, the input plane is returned unchanged along with the
    inlier count.

    Returns (normal, offset, inlier_count).
    """
    distances = points @ normal - offset

    median_distance = float(np.median(distances))
    mad = float(np.median(np.abs(distances - median_distance)))

    threshold = max(0.015, 3.0 * 1.4826 * mad)

    inliers = np.abs(distances) <= threshold
    inlier_count = int(np.count_nonzero(inliers))

    if inlier_count < 100:
        return normal, offset, inlier_count

    inlier_points = points[inliers]
    center = inlier_points.mean(axis=0)
    centered = inlier_points - center

    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    refined_normal = vh[-1]
    refined_normal = refined_normal / np.linalg.norm(refined_normal)
    refined_offset = float(np.dot(refined_normal, center))

    return refined_normal, refined_offset, inlier_count


def estimate_volume_cm3(
    depth_map: np.ndarray,
    mask: np.ndarray,
    focal_length_px: float,
) -> dict:
    """
    Estimate physical food volume in cm^3.

    Inputs:
        depth_map: metric depth in meters, shape (H, W).
        mask: binary food mask, shape (H, W).
        focal_length_px: camera focal length in pixels.

    Method:
        1. Back-project background pixels into 3D.
        2. Robustly fit the supporting plane (RANSAC + SVD refine).
        3. Back-project food pixels into camera rays.
        4. Intersect each ray with the supporting plane.
        5. Integrate the occupied ray volume (cone approximation).
        6. Convert m^3 to cm^3.

    Returns:
        {
            "volume_cm3": float,
            "support_inliers": int,
            "support_ratio": float,
            "valid_volume_pixels": int,
        }

    Raises:
        ValueError: if inputs are malformed, if there are too few
            valid pixels, if the supporting plane cannot be fit
            reliably, or if the reconstruction yields no positive
            volume.
    """
    depth = np.asarray(depth_map, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    focal_length_px = float(focal_length_px)

    if depth.shape != mask.shape:
        raise ValueError("Depth map and mask dimensions do not match.")

    if focal_length_px <= 0:
        raise ValueError("Focal length must be positive.")

    valid_food = mask & np.isfinite(depth) & (depth > 0)

    if int(np.count_nonzero(valid_food)) < 50:
        raise ValueError("Food segmentation contains too few valid pixels.")

    height, width = depth.shape
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0

    # --- Supporting-plane points: background pixels away from the food.

    expanded_mask = binary_dilation(mask, iterations=8)
    valid_background = np.isfinite(depth) & (depth > 0) & ~expanded_mask

    by, bx = np.where(valid_background)

    if len(bx) < 200:
        raise ValueError("Not enough visible supporting-surface pixels.")

    if len(bx) > 5000:
        rng = np.random.default_rng(42)
        selected = rng.choice(len(bx), size=5000, replace=False)
        bx = bx[selected]
        by = by[selected]

    bz = depth[by, bx]
    valid = np.isfinite(bz) & (bz > 0)
    bx = bx[valid]
    by = by[valid]
    bz = bz[valid]

    if len(bx) < 200:
        raise ValueError("Not enough valid supporting-surface depth points.")

    background_x = (bx - cx) * bz / focal_length_px
    background_y = (by - cy) * bz / focal_length_px
    background_points = np.column_stack((background_x, background_y, bz))

    # --- RANSAC plane fit.

    rng = np.random.default_rng(42)
    best_normal = None
    best_offset = None
    best_count = 0
    point_count = len(background_points)

    for _ in range(300):
        indices = rng.choice(point_count, size=3, replace=False)
        plane = _plane_from_points(
            background_points[indices[0]],
            background_points[indices[1]],
            background_points[indices[2]],
        )
        if plane is None:
            continue

        normal, offset = plane
        distances = np.abs(background_points @ normal - offset)
        count = int(np.count_nonzero(distances <= 0.015))

        if count > best_count:
            best_count = count
            best_normal = normal
            best_offset = offset

    if best_normal is None:
        raise ValueError("Could not estimate supporting surface.")

    plane_normal, plane_offset, support_inliers = _refine_plane(
        background_points, best_normal, best_offset
    )

    support_ratio = support_inliers / len(background_points)

    if support_ratio < 0.25:
        raise ValueError("Supporting surface is not sufficiently planar.")

    # Orient normal toward the camera (positive z).
    if plane_normal[2] < 0:
        plane_normal = -plane_normal
        plane_offset = -plane_offset

    # --- Food pixels -> rays.

    fy, fx = np.where(valid_food)
    food_z = depth[fy, fx]

    ray_x = (fx - cx) / focal_length_px
    ray_y = (fy - cy) / focal_length_px
    ray_z = np.ones_like(ray_x, dtype=np.float32)
    ray_norm = np.sqrt(ray_x ** 2 + ray_y ** 2 + 1.0)

    surface_ray_distance = food_z * ray_norm

    unit_x = ray_x / ray_norm
    unit_y = ray_y / ray_norm
    unit_z = ray_z / ray_norm

    # --- Ray / plane intersection: t = d / (n . ray).

    denominator = (
        plane_normal[0] * unit_x
        + plane_normal[1] * unit_y
        + plane_normal[2] * unit_z
    )

    valid_rays = np.abs(denominator) > 1e-6
    plane_ray_distance = np.full(
        surface_ray_distance.shape, np.nan, dtype=np.float64
    )
    plane_ray_distance[valid_rays] = (
        plane_offset / denominator[valid_rays]
    )

    # Keep only rays where the food surface is closer to the camera
    # than the supporting plane, and the plane is in front of the camera.
    valid_volume = (
        np.isfinite(plane_ray_distance)
        & (plane_ray_distance > surface_ray_distance)
        & (plane_ray_distance > 0)
    )

    if int(np.count_nonzero(valid_volume)) < 50:
        raise ValueError(
            "No reliable positive food volume could be reconstructed."
        )

    surface_ray_distance = surface_ray_distance[valid_volume]
    plane_ray_distance = plane_ray_distance[valid_volume]
    ray_x = ray_x[valid_volume]
    ray_y = ray_y[valid_volume]

    # --- Cone-volume integration.
    # Solid angle per pixel (pinhole): dOmega = 1 / f^2 / (1 + x^2 + y^2)^1.5
    # Volume along a ray: dV = (r_plane^3 - r_surface^3) / 3 * dOmega

    solid_angle_per_pixel = (
        1.0
        / (focal_length_px ** 2)
        / (1.0 + ray_x ** 2 + ray_y ** 2) ** 1.5
    )

    ray_volume_m3 = (
        (plane_ray_distance ** 3 - surface_ray_distance ** 3)
        / 3.0
        * solid_angle_per_pixel
    )
    ray_volume_m3 = np.maximum(ray_volume_m3, 0.0)

    volume_m3 = float(np.sum(ray_volume_m3))
    print("=" * 60, flush=True)
    print("VOLUME ENGINE INTERMEDIATE", flush=True)
    print(f"  focal_length_px: {focal_length_px}", flush=True)
    print(f"  plane_normal: {plane_normal}", flush=True)
    print(f"  plane_offset: {plane_offset}", flush=True)
    print(f"  valid_volume pixels: {int(np.count_nonzero(valid_volume))}", flush=True)
    print(
        f"  surface_ray_distance min/med/max: "
        f"{surface_ray_distance.min():.4f} / "
        f"{float(np.median(surface_ray_distance)):.4f} / "
        f"{surface_ray_distance.max():.4f}",
        flush=True,
    )
    print(
        f"  plane_ray_distance min/med/max: "
        f"{plane_ray_distance.min():.4f} / "
        f"{float(np.median(plane_ray_distance)):.4f} / "
        f"{plane_ray_distance.max():.4f}",
        flush=True,
    )
    _height = plane_ray_distance - surface_ray_distance
    print(
        f"  height min/med/max: "
        f"{_height.min():.4f} / {float(np.median(_height)):.4f} / {_height.max():.4f}",
        flush=True,
    )

    volume_cm3 = volume_m3 * 1_000_000.0

    if not np.isfinite(volume_cm3) or volume_cm3 <= 0:
        raise ValueError(
            "Physical volume reconstruction produced an invalid result."
        )

    return {
        "volume_cm3": float(volume_cm3),
        "support_inliers": int(support_inliers),
        "support_ratio": float(support_ratio),
        "valid_volume_pixels": int(np.count_nonzero(valid_volume)),
    }