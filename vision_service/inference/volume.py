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
    return_debug: bool = False,
) -> dict:
    """
    Estimate physical food volume in cm^3.

    Inputs:
        depth_map: metric depth in meters, shape (H, W).
        mask: binary food mask, shape (H, W).
        focal_length_px: camera focal length in pixels.
        return_debug: if True, include a "_debug" key with plane-fit
            parameters and per-stage depth/height statistics. Used by
            benchmarks/ablation/formula_diagnostic.py to isolate whether
            an error is in the depth, the mask, the focal, or the plane.

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
            "_debug": { ... }   # only when return_debug=True
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

    # Fit the plane to pixels adjacent to the mask, not the whole
    # image. The whole-image sample includes distant background
    # (measured: 34 cm depth spread on real photos), which tilts
    # the plane. Local ring is the plate under the food.
    expanded_mask = binary_dilation(mask, iterations=8)
    local_ring = binary_dilation(mask, iterations=60) & ~expanded_mask
    valid_background = (
        np.isfinite(depth) & (depth > 0) & local_ring
    )
    if int(valid_background.sum()) < 200:
        valid_background = (
            np.isfinite(depth) & (depth > 0) & ~expanded_mask
        )

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
        count = int(np.count_nonzero(distances <= 0.005))

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
    volume_cm3 = volume_m3 * 1_000_000.0

    if not np.isfinite(volume_cm3) or volume_cm3 <= 0:
        raise ValueError(
            "Physical volume reconstruction produced an invalid result."
        )

    result = {
        "volume_cm3": float(volume_cm3),
        "support_inliers": int(support_inliers),
        "support_ratio": float(support_ratio),
        "valid_volume_pixels": int(np.count_nonzero(valid_volume)),
    }

    if return_debug:
        height = plane_ray_distance - surface_ray_distance
        result["_debug"] = {
            "plane_normal": plane_normal.tolist(),
            "plane_offset": float(plane_offset),
            "support_inliers": int(support_inliers),
            "support_ratio": float(support_ratio),
            "food_z_min": float(np.min(food_z)),
            "food_z_median": float(np.median(food_z)),
            "food_z_max": float(np.max(food_z)),
            "surface_ray_distance_min": float(np.min(surface_ray_distance)),
            "surface_ray_distance_median": float(np.median(surface_ray_distance)),
            "surface_ray_distance_max": float(np.max(surface_ray_distance)),
            "plane_ray_distance_min": float(np.min(plane_ray_distance)),
            "plane_ray_distance_median": float(np.median(plane_ray_distance)),
            "plane_ray_distance_max": float(np.max(plane_ray_distance)),
            "height_min": float(np.min(height)),
            "height_median": float(np.median(height)),
            "height_max": float(np.max(height)),
            "valid_volume_pixels": int(np.count_nonzero(valid_volume)),
            "volume_m3": float(volume_m3),
        }

    return result
def estimate_volume_cm3_v2(
    depth_map: np.ndarray,
    mask: np.ndarray,
    focal_length_px: float,
    label: str = "",
    adaptive_percentile: float = 90.0,
    adaptive_factor: float = 0.35,
    min_threshold_m: float = 0.003,
    height_cap_percentile: float = 99.0,
    scale_min: float = 0.5,
    scale_max: float = 5.0,
    return_debug: bool = False,
) -> dict:
    """
    Volume from the mask itself, with class-prior scale correction.

    Phase 1: adaptive threshold picks the food pixels (mask covers
    plate+food; plane is fit to the mask's dominant surface).
    Phase 2: cap heights at the 99th percentile to clip per-pixel noise.
    Phase 2b: Depth Pro under-reports the food-dome height by 2-5x on
    many photos (measured: hcap 8-12mm where real bagel is ~40mm).
    Correct by scaling volume by prior_height / measured_height.
    Bounded to [scale_min, scale_max] so a wrong label or a
    catastrophically failed depth cannot cause runaway correction.

    See vision_service.inference.food_priors for the prior table.
    """
    depth = np.asarray(depth_map, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    h, w = depth.shape
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0

    fy, fx = np.where(mask & np.isfinite(depth) & (depth > 0))
    if len(fx) < 200:
        raise ValueError("Mask too small")

    z = depth[fy, fx]
    x = (fx - cx) * z / focal_length_px
    y = (fy - cy) * z / focal_length_px

    from sklearn.linear_model import RANSACRegressor, LinearRegression
    ransac = RANSACRegressor(
        LinearRegression(),
        residual_threshold=0.010,
        random_state=42,
        min_samples=10,
    )
    ransac.fit(np.column_stack([x, y]), z)
    a = float(ransac.estimator_.coef_[0])
    b = float(ransac.estimator_.coef_[1])
    c = float(ransac.estimator_.intercept_)

    plane_z = a * x + b * y + c
    height = plane_z - z

    positive = height[height > 0]
    if len(positive) < 100:
        raise ValueError(f"Not enough positive-height pixels (n={len(positive)})")

    p_high = float(np.percentile(positive, adaptive_percentile))
    threshold = max(min_threshold_m, p_high * adaptive_factor)

    if p_high < min_threshold_m:
        raise ValueError(
            f"Depth did not resolve food (p{adaptive_percentile:.0f}="
            f"{p_high*1000:.1f}mm < {min_threshold_m*1000:.1f}mm)"
        )

    food = height > threshold
    if int(food.sum()) < 50:
        raise ValueError(
            f"No food pixels above adaptive plate "
            f"(thr={threshold*1000:.1f}mm p{adaptive_percentile:.0f}="
            f"{p_high*1000:.1f}mm)"
        )

    food_h = height[food]
    h_cap = float(np.percentile(food_h, height_cap_percentile))
    h_clipped = np.minimum(food_h, h_cap)

    pixel_area_m2 = (z[food] ** 2) / (focal_length_px ** 2)
    volume_m3_raw = float(np.sum(h_clipped * pixel_area_m2))
    volume_cm3_raw = volume_m3_raw * 1e6

    from vision_service.inference.food_priors import get_prior
    prior = get_prior(label)
    prior_h_mm = float(prior["height_mm"])
    h_cap_mm = h_cap * 1000.0

    if h_cap_mm > 0:
        scale = prior_h_mm / h_cap_mm
    else:
        scale = 1.0

    if scale_min <= scale <= scale_max:
        scale_applied = scale
    else:
        scale_applied = 1.0

    volume_cm3 = volume_cm3_raw * scale_applied

    # Global calibration factor: pipeline systematically over-estimates
    # by 1.26x on real bagels. Confirmed by 5-fold CV (scale std 0.01
    # across folds, 40 bagel photos in SimpleFood45).
    #   raw MAPE 32.69%  ->  calibrated MAPE 17.95%
    # Honest caveat: n=40, near-duplicate photos; diverse bagels will
    # be worse. See merge commit for full disclosure.
    CALIBRATION_FACTOR = 1.26
    volume_cm3 = volume_cm3 / CALIBRATION_FACTOR

    result = {
        "volume_cm3": volume_cm3,
        "support_inliers": int(ransac.inlier_mask_.sum()),
        "support_ratio": float(ransac.inlier_mask_.mean()),
        "valid_volume_pixels": int(food.sum()),
    }

    if return_debug:
        result["_debug"] = {
            "plane_normal": [a, b, -1.0],
            "plane_offset": float(c),
            "support_inliers": int(ransac.inlier_mask_.sum()),
            "support_ratio": float(ransac.inlier_mask_.mean()),
            "adaptive_threshold_m": float(threshold),
            "p_high_m": float(p_high),
            "height_cap_m": float(h_cap),
            "volume_cm3_raw": float(volume_cm3_raw),
            "food_z_min": float(np.min(z[food])),
            "food_z_median": float(np.median(z[food])),
            "food_z_max": float(np.max(z[food])),
            "height_min": float(np.min(food_h)),
            "height_median": float(np.median(food_h)),
            "height_max": float(np.max(food_h)),
            "height_p95": float(np.percentile(food_h, 95)),
            "valid_volume_pixels": int(food.sum()),
        }

    return result


def estimate_volume_cm3_v3(
    depth_map: np.ndarray,
    mask: np.ndarray,
    focal_length_px: float,
    label: str = "",
    adaptive_percentile: float = 90.0,
    adaptive_factor: float = 0.35,
    min_threshold_m: float = 0.003,
    height_cap_percentile: float = 99.0,
    scale_min: float = 0.5,
    scale_max: float = 5.0,
    column_grid_px: int = 256,
    return_debug: bool = False,
) -> dict:
    """Vertical-column volume integration (phase 3)."""
    depth = np.asarray(depth_map, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    h, w = depth.shape
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0

    fy, fx = np.where(mask & np.isfinite(depth) & (depth > 0))
    if len(fx) < 200:
        raise ValueError("Mask too small")

    z = depth[fy, fx]
    x = (fx - cx) * z / focal_length_px
    y = (fy - cy) * z / focal_length_px

    from sklearn.linear_model import RANSACRegressor, LinearRegression
    ransac = RANSACRegressor(
        LinearRegression(),
        residual_threshold=0.010,
        random_state=42,
        min_samples=10,
    )
    ransac.fit(np.column_stack([x, y]), z)
    a = float(ransac.estimator_.coef_[0])
    b = float(ransac.estimator_.coef_[1])
    c = float(ransac.estimator_.intercept_)

    plane_z = a * x + b * y + c
    height = plane_z - z

    positive = height[height > 0]
    if len(positive) < 100:
        raise ValueError(
            f"Not enough positive-height pixels (n={len(positive)})"
        )

    p_high = float(np.percentile(positive, adaptive_percentile))
    threshold = max(min_threshold_m, p_high * adaptive_factor)
    if p_high < min_threshold_m:
        raise ValueError(
            f"Depth did not resolve food (p{adaptive_percentile:.0f}="
            f"{p_high*1000:.1f}mm < {min_threshold_m*1000:.1f}mm)"
        )

    food = height > threshold
    n_food = int(food.sum())
    if n_food < 50:
        raise ValueError(
            f"No food pixels above adaptive plate "
            f"(thr={threshold*1000:.1f}mm p{adaptive_percentile:.0f}="
            f"{p_high*1000:.1f}mm)"
        )

    food_h = height[food]
    h_cap = float(np.percentile(food_h, height_cap_percentile))
    food_h = np.minimum(food_h, h_cap)

    max_dim = max(w, h)
    cell = max(1, int(max_dim / column_grid_px))
    gy = (fy[food] // cell).astype(np.int64)
    gx = (fx[food] // cell).astype(np.int64)
    grid_h = int(h // cell) + 1
    grid_w = int(w // cell) + 1

    # Column-p80, not column-max. Depth Pro has 5-10mm per-pixel
    # noise on real photos. Column-max picks the noisiest outlier
    # in each cell, which inflated v3 median by 17% vs v2 on the
    # 120-photo sweep (319.9 vs 272.7) even though v3 is correct
    # on synthetic cubes. p80 filters per-pixel noise while
    # preserving the dome's shape.
    cell_lists = {}
    for gyy, gxx, hh in zip(gy, gx, food_h):
        cell_lists.setdefault((gyy, gxx), []).append(float(hh))
    column_h = np.zeros((grid_h, grid_w), dtype=np.float32)
    for (gyy, gxx), vals in cell_lists.items():
        column_h[gyy, gxx] = float(np.percentile(vals, 80))

    z_median = float(np.median(z[food]))
    cell_footprint_m2 = (float(cell) * z_median / focal_length_px) ** 2

    n_columns = int((column_h > 0).sum())
    volume_m3_raw = float(np.sum(column_h) * cell_footprint_m2)
    volume_cm3_raw = volume_m3_raw * 1e6
    volume_cm3 = volume_cm3_raw

    result = {
        "volume_cm3": volume_cm3,
        "support_inliers": int(ransac.inlier_mask_.sum()),
        "support_ratio": float(ransac.inlier_mask_.mean()),
        "valid_volume_pixels": n_food,
    }

    if return_debug:
        result["_debug"] = {
            "plane_normal": [a, b, -1.0],
            "plane_offset": float(c),
            "support_inliers": int(ransac.inlier_mask_.sum()),
            "support_ratio": float(ransac.inlier_mask_.mean()),
            "adaptive_threshold_m": float(threshold),
            "p_high_m": float(p_high),
            "height_cap_m": float(h_cap),
            "volume_cm3_raw": float(volume_cm3_raw),
            "volume_cm3_corrected": float(volume_cm3),
            "n_columns": n_columns,
            "cell_px": int(cell),
            "cell_footprint_m2": float(cell_footprint_m2),
            "valid_volume_pixels": n_food,
        }

    return result

def estimate_volume_cm3_v4(
    depth_map: np.ndarray,
    mask: np.ndarray,
    focal_length_px: float,
    label: str = "",
    ring_dilation_px: int = 30,
    adaptive_percentile: float = 90.0,
    adaptive_factor: float = 0.35,
    min_threshold_m: float = 0.003,
    column_grid_px: int = 256,
    return_debug: bool = False,
) -> dict:
    """
    Exact column integration.
    Area uses plane_z (plate depth), not food depth.
    """
    from scipy.ndimage import binary_dilation, median_filter
    from sklearn.linear_model import RANSACRegressor, LinearRegression
    
    depth = np.asarray(depth_map, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    h, w = depth.shape
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0

    if not np.any(mask):
        raise ValueError("Mask is empty")

    depth = median_filter(depth, size=5)

    dilated = binary_dilation(mask, iterations=ring_dilation_px)
    ring = dilated & ~mask & np.isfinite(depth) & (depth > 0)
    if ring.sum() < 100:
        ring = ~mask & np.isfinite(depth) & (depth > 0)

    ry, rx = np.where(ring)
    if len(rx) < 100:
        raise ValueError("Not enough background ring pixels")

    rz = depth[ry, rx]
    valid_ring = np.isfinite(rz) & (rz > 0)
    ry, rx, rz = ry[valid_ring], rx[valid_ring], rz[valid_ring]
    
    rx_3d = (rx - cx) * rz / focal_length_px
    ry_3d = (ry - cy) * rz / focal_length_px
    ring_pts = np.column_stack([rx_3d, ry_3d, rz])

    ransac = RANSACRegressor(LinearRegression(), residual_threshold=0.010, random_state=42, min_samples=10)
    ransac.fit(ring_pts[:, :2], ring_pts[:, 2])
    a = float(ransac.estimator_.coef_[0])
    b = float(ransac.estimator_.coef_[1])
    c = float(ransac.estimator_.intercept_)

    fy, fx = np.where(mask & np.isfinite(depth) & (depth > 0))
    z = depth[fy, fx]
    x = (fx - cx) * z / focal_length_px
    y = (fy - cy) * z / focal_length_px

    plane_z = a * x + b * y + c
    height = plane_z - z

    positive = height[height > 0]
    if len(positive) < 100:
        raise ValueError(f"Not enough positive-height pixels (n={len(positive)})")

    p_high = float(np.percentile(positive, adaptive_percentile))
    threshold = max(min_threshold_m, p_high * adaptive_factor)

    if p_high < min_threshold_m:
        raise ValueError(f"Depth did not resolve food")

    food = height > threshold
    n_food = int(food.sum())
    if n_food < 50:
        raise ValueError("No food pixels above plate")

    food_h = height[food]
    food_plane_z = plane_z[food]

    max_dim = max(w, h)
    cell = max(1, int(max_dim / column_grid_px))
    gy = (fy[food] // cell).astype(np.int64)
    gx = (fx[food] // cell).astype(np.int64)
    grid_h, grid_w = int(h // cell) + 1, int(w // cell) + 1

    column_h = np.zeros((grid_h, grid_w), dtype=np.float32)
    np.maximum.at(column_h, (gy, gx), food_h)

    # Use plane_z for area (the correct footprint on the plate)
    pixel_area_m2 = (food_plane_z / focal_length_px) ** 2
    cell_areas = np.zeros((grid_h, grid_w), dtype=np.float32)
    np.add.at(cell_areas, (gy, gx), pixel_area_m2)

    volume_m3 = float(np.sum(column_h * cell_areas))
    volume_cm3 = volume_m3 * 1e6

    result = {
        "volume_cm3": volume_cm3,
        "support_inliers": int(ransac.inlier_mask_.sum()),
        "support_ratio": float(ransac.inlier_mask_.mean()),
        "valid_volume_pixels": n_food,
    }
    if return_debug:
        result["_debug"] = {
            "plane_normal": [a, b, -1.0],
            "plane_offset": float(c),
            "volume_cm3_raw": float(volume_cm3),
        }
    return result
