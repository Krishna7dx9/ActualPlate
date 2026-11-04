"""
Formula audit -- Phase 0 of the critical path.

Four tests, each with a hard gate. Read-only. No GPU.

  Task 0.3  Flat-slab analytic volume test.
            Rectangular mask, constant depth d_food on a
            background at Z_plane.  Expected volume has a
            closed form:

              V_true = A_pix * (Z_plane^3 - d_food^3) / (3 * f^2)

            with A_pix the inclusive pixel area of the mask.
            Ratio must be 1.00 +/- 0.03.

  Task 0.4  Tilted plane, analytic volume.
            Exercises the RANSAC + SVD plane-fit path that
            Task 0.3 cannot, because 0.3 uses a fronto-parallel
            plane with a trivial fit.  Ratio must be 1.00 +/- 0.05.

  Task 0.1  Resolution audit on one real MetaFood3D item.
  Task 0.2  Focal exponent test on one real MetaFood3D item.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from vision_service.inference.volume import estimate_volume_cm3


def _banner(t: str) -> None:
    print()
    print("=" * 72)
    print(t)
    print("=" * 72)


def _gate(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
          + (f" -- {detail}" if detail else ""))


# ----------------------------------------------------------------------
# Task 0.3 -- Flat slab, closed-form expected volume
# ----------------------------------------------------------------------


def build_flat_slab(
    H: int,
    W: int,
    f: float,
    Z_plane: float,
    d_food: float,
    margin: int,
):
    """
    Build a flat-slab scene.

    Background (outside the mask): z-depth = Z_plane everywhere.
    Mask (interior rectangle):     z-depth = d_food everywhere.

    The 3D "food" region is the frustum of rays through the mask,
    between z = d_food and z = Z_plane.  Its exact volume is

        V_true = A_pix * (Z_plane^3 - d_food^3) / (3 * f^2)

    where A_pix counts the inclusive pixels of the mask:
        A_pix = (px_max - px_min + 1) * (py_max - py_min + 1)
    """
    depth_map = np.full((H, W), Z_plane, dtype=np.float32)
    mask = np.zeros((H, W), dtype=bool)

    px_min = margin
    px_max = W - margin - 1
    py_min = margin
    py_max = H - margin - 1

    mask[py_min:py_max + 1, px_min:px_max + 1] = True
    depth_map[mask] = d_food

    A_pix = (px_max - px_min + 1) * (py_max - py_min + 1)
    V_true_m3 = A_pix * (Z_plane ** 3 - d_food ** 3) / (3.0 * f ** 2)
    V_true_cm3 = V_true_m3 * 1e6

    return depth_map, mask, V_true_cm3, (px_min, px_max, py_min, py_max)


def task_0_3_flat_slab(
    H: int = 1024,
    W: int = 1024,
    f: float = 1120.0,
    Z_plane: float = 1.0,
    d_food: float = 0.96,
    margin: int = 250,
) -> bool:
    _banner("TASK 0.3 -- FLAT-SLAB ANALYTIC VOLUME")

    print(f"  image size:     {H} x {W}")
    print(f"  focal:          {f} px")
    print(f"  plane depth:    {Z_plane} m")
    print(f"  food depth:     {d_food} m  (height {Z_plane - d_food:.3f} m)")
    print(f"  mask margin:    {margin} px")

    depth, mask, V_true, (px0, px1, py0, py1) = build_flat_slab(
        H, W, f, Z_plane, d_food, margin,
    )

    n_pix = int(mask.sum())
    print(f"  mask pixels:    {n_pix}")
    print(f"  mask x range:   [{px0}, {px1}]")
    print(f"  mask y range:   [{py0}, {py1}]")
    print()
    print(f"  expected volume (cm^3): {V_true:.6f}")
    print()

    try:
        res = estimate_volume_cm3(depth, mask, f, return_debug=True)
    except Exception as exc:
        print(f"  estimate_volume_cm3 raised: {exc}")
        _gate("0.3 flat slab", False, str(exc))
        return False

    measured = res["volume_cm3"]
    ratio = measured / V_true

    print(f"  measured volume (cm^3): {measured:.6f}")
    print(f"  ratio:                  {ratio:.6f}")
    print(f"  target:                 1.000000 +/- 0.030000")
    print()

    dbg = res.get("_debug", {})
    if dbg:
        print(f"  support_inliers:        {res['support_inliers']}")
        print(f"  support_ratio:          {res['support_ratio']:.6f}")
        print(f"  plane_normal:           {dbg.get('plane_normal')}")
        print(f"  plane_offset:           {dbg.get('plane_offset')}")
        print(f"  food_z_median:          {dbg.get('food_z_median')}")
        print(f"  plane_ray_distance_med: {dbg.get('plane_ray_distance_median')}")
        print(f"  surface_ray_dist_med:   {dbg.get('surface_ray_distance_median')}")
        print(f"  height_median:          {dbg.get('height_median')}")

    ok = abs(ratio - 1.0) <= 0.03
    _gate("0.3 flat slab", ok, f"ratio={ratio:.6f}")
    return ok


# ----------------------------------------------------------------------
# Task 0.4 -- Tilted plane, analytic volume
# ----------------------------------------------------------------------


def build_tilted_slab(
    H: int,
    W: int,
    f: float,
    plane_normal: tuple,
    plane_offset: float,
    food_height: float,
    margin: int,
):
    """
    Tilted plane + constant-thickness food layer above it.

    Plane satisfies  n . x = plane_offset  with n a unit normal.
    Food top surface: same normal, offset reduced by food_height
    (so it is closer to the camera by that distance along n).

    The returned depth_map is z-depth, which is what the volume
    formula expects.  Ray distance along the unit ray is
    t = plane_offset / (n . u).  The z-depth is t * uz where
    uz = 1 / ray_norm.
    """
    cy = (H - 1) / 2.0
    cx = (W - 1) / 2.0
    fy, fx = np.meshgrid(
        np.arange(H, dtype=np.float64),
        np.arange(W, dtype=np.float64),
        indexing="ij",
    )
    ray_x = (fx - cx) / f
    ray_y = (fy - cy) / f
    ray_norm = np.sqrt(ray_x ** 2 + ray_y ** 2 + 1.0)
    ux = ray_x / ray_norm
    uy = ray_y / ray_norm
    uz = 1.0 / ray_norm

    n = np.asarray(plane_normal, dtype=np.float64)
    n = n / np.linalg.norm(n)

    denom = n[0] * ux + n[1] * uy + n[2] * uz
    valid = np.abs(denom) > 1e-9

    # Distance along ray to plane and to food top.
    t_plane = np.full_like(denom, np.nan)
    t_plane[valid] = plane_offset / denom[valid]

    plane_top_offset = plane_offset - food_height
    t_food = np.full_like(denom, np.nan)
    t_food[valid] = plane_top_offset / denom[valid]

    mask = np.zeros((H, W), dtype=bool)
    mask[margin:H - margin, margin:W - margin] = True

    # Convert ray distance -> z-depth.  z = t * uz.
    depth_map = np.full((H, W), np.nan, dtype=np.float32)
    depth_map[mask] = (t_food[mask] * uz[mask]).astype(np.float32)
    bg = ~mask & np.isfinite(t_plane)
    depth_map[bg] = (t_plane[bg] * uz[bg]).astype(np.float32)

    # Any residual NaNs (invalid denom) -> plane z-depth at center.
    depth_map = np.nan_to_num(depth_map, nan=float(plane_offset))

    # True volume per pixel: dV = (r_plane^3 - r_surface^3) / 3 * dOmega
    # where r_* are ray distances and dOmega = 1/(f^2 * ray_norm^3).
    dV = (
        (t_plane ** 3 - t_food ** 3)
        / 3.0
        / (f ** 2)
        / (ray_norm ** 3)
    )
    dV = np.where(mask, dV, 0.0)
    V_true_cm3 = float(np.nansum(dV)) * 1e6

    return depth_map, mask, V_true_cm3


def task_0_4_tilted_plane(
    H: int = 1024,
    W: int = 1024,
    f: float = 1120.0,
) -> bool:
    _banner("TASK 0.4 -- TILTED PLANE, ANALYTIC VOLUME")

    # Tilted plane: n ~ (0.2005, 0.3008, 0.9324) after normalization.
    # Offset chosen so the plane passes through (0, 0, 1.0).
    n_raw = (0.2, 0.3, 0.93)
    n = np.asarray(n_raw) / np.linalg.norm(n_raw)
    plane_offset = 1.0 * n[2]
    food_height = 0.04

    print(f"  image size:      {H} x {W}")
    print(f"  focal:           {f} px")
    print(f"  plane normal:    {n.tolist()}")
    print(f"  plane offset:    {plane_offset:.6f}")
    print(f"  food thickness:  {food_height} m")
    print()

    depth, mask, V_true = build_tilted_slab(
        H, W, f, tuple(n), plane_offset, food_height, margin=250,
    )

    print(f"  mask pixels:     {int(mask.sum())}")
    print(f"  expected volume: {V_true:.6f} cm^3")
    print()

    try:
        res = estimate_volume_cm3(depth, mask, f, return_debug=True)
    except Exception as exc:
        print(f"  estimate_volume_cm3 raised: {exc}")
        _gate("0.4 tilted plane", False, str(exc))
        return False

    measured = res["volume_cm3"]
    ratio = measured / V_true if V_true > 0 else float("nan")

    print(f"  measured volume: {measured:.6f} cm^3")
    print(f"  ratio:           {ratio:.6f}")
    print(f"  target:          1.000000 +/- 0.050000")
    print()

    dbg = res.get("_debug", {})
    if dbg:
        print(f"  true normal:     {n.tolist()}")
        print(f"  true offset:     {plane_offset:.6f}")
        print(f"  fitted normal:   {dbg.get('plane_normal')}")
        print(f"  fitted offset:   {dbg.get('plane_offset')}")
        print(f"  support_ratio:   {res['support_ratio']:.6f}")
        print(f"  height_median:   {dbg.get('height_median')}")

    ok = abs(ratio - 1.0) <= 0.05
    _gate("0.4 tilted plane", ok, f"ratio={ratio:.6f}")
    return ok


# ----------------------------------------------------------------------
# Task 0.1 -- Resolution audit
# ----------------------------------------------------------------------


def task_0_1_resolution(item) -> bool:
    _banner("TASK 0.1 -- RESOLUTION AUDIT")

    mask_shape = item.gt_mask.shape
    depth_shape = item.gt_depth_map.shape

    from PIL import Image
    with Image.open(item.rgb_path) as im:
        image_w, image_h = im.size

    print(f"  item:             {item.id}")
    print(f"  rgb image (W,H):  ({image_w}, {image_h})")
    print(f"  depth_map.shape:  {depth_shape}")
    print(f"  gt_mask.shape:    {mask_shape}")
    print(f"  focal_length_px:  {item.focal_length_px}")

    same = (image_w == mask_shape[1] and image_h == mask_shape[0])
    if same:
        _gate("0.1 resolution", True, "image and mask same resolution")
        return True

    print()
    print(f"  W_image / W_mask = {image_w / mask_shape[1]:.6f}")
    print(f"  H_image / H_mask = {image_h / mask_shape[0]:.6f}")
    _gate("0.1 resolution", False,
          f"mismatch: W ratio {image_w / mask_shape[1]:.4f}")
    return False


# ----------------------------------------------------------------------
# Task 0.2 -- Focal exponent
# ----------------------------------------------------------------------


def task_0_2_focal_exponent(item) -> bool:
    _banner("TASK 0.2 -- FOCAL EXPONENT TEST")

    depth = item.gt_depth_map
    mask = item.gt_mask
    true_f = float(item.focal_length_px)
    focals = [true_f * 0.5, true_f * 1.0, true_f * 2.0, true_f * 4.0]

    print(f"  true focal: {true_f:.4f} px")
    print()
    print(f"  {'focal (px)':>14}  {'volume (cm^3)':>18}")

    volumes = []
    for f in focals:
        try:
            r = estimate_volume_cm3(depth, mask, f)
            v = float(r["volume_cm3"])
            volumes.append(v)
            print(f"  {f:>14.4f}  {v:>18.6f}")
        except Exception as exc:
            volumes.append(None)
            print(f"  {f:>14.4f}  FAILED: {exc}")

    if any(v is None or v <= 0 for v in volumes):
        _gate("0.2 focal exponent", False, "one or more focals failed")
        return False

    v1, v4 = volumes[0], volumes[-1]
    f1, f4 = focals[0], focals[-1]
    n = np.log(v4 / v1) / np.log(f1 / f4)

    print()
    print(f"  measured exponent (V ~ 1/f^n): {n:.6f}")
    print(f"  expected:                      2.000000")
    print(f"  tolerance:                     +/- 0.100000")

    ok = abs(n - 2.0) <= 0.1
    _gate("0.2 focal exponent", ok, f"n={n:.4f}")
    return ok


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="formula_audit")
    p.add_argument("--scene-size", type=int, default=1024)
    p.add_argument("--focal", type=float, default=1120.0)
    p.add_argument("--skip-real-item", action="store_true")
    args = p.parse_args(argv)

    print("=" * 72)
    print("FORMULA AUDIT -- PHASE 0")
    print("=" * 72)
    print(f"  scene-size:      {args.scene_size}")
    print(f"  synthetic focal: {args.focal}")

    results = []

    results.append((
        "0.3 flat slab",
        task_0_3_flat_slab(
            H=args.scene_size,
            W=args.scene_size,
            f=args.focal,
        ),
    ))
    results.append((
        "0.4 tilted plane",
        task_0_4_tilted_plane(
            H=args.scene_size,
            W=args.scene_size,
            f=args.focal,
        ),
    ))

    if not args.skip_real_item:
        try:
            from benchmarks.ablation.metafood3d import MetaFood3DAdapter
            adapter = MetaFood3DAdapter()
            items = adapter.load_items(limit=1)
            if items:
                item = items[0]
                results.append(("0.1 resolution", task_0_1_resolution(item)))
                results.append(("0.2 focal exponent", task_0_2_focal_exponent(item)))
            else:
                print("\nMetaFood3D returned no items. Skipping 0.1 and 0.2.")
        except Exception as exc:
            print(f"\nMetaFood3D not available: {exc}")
            print("Skipping 0.1 and 0.2.")

    _banner("PHASE 0 GATES")
    for name, ok in results:
        _gate(name, ok)

    all_ok = all(ok for _, ok in results)
    print()
    if all_ok:
        print("PHASE 0 COMPLETE. Proceed to Phase 1.")
        return 0
    print("PHASE 0 INCOMPLETE. Do not proceed.")
    return 1


if __name__ == "__main__":
    sys.exit(main())