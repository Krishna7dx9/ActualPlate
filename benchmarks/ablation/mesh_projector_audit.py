"""
Mesh projector audit.

Two checks against analytically known ground truth. No assumptions.

Check 1 — Cube mesh, exact volume 500 cm^3.
  Build a 10 x 10 x 5 cm cube sitting on the plate at world
  origin. Project through the MetaFood3D camera. Compute volume
  with the same formula. Expected result: 500 cm^3 to within 5%.
  Also cross-check trimesh's own mesh volume, which should be
  exactly 500 cm^3.

Check 2 — Direct volume of the real almond mesh.
  trimesh computes mesh volume from geometry alone, no
  projection, no formula. Compare against the Excel GT of
  341.29 cm^3. If it matches, the mesh represents the food. If
  not, the mesh represents something else.

The two checks are independent. Together they say whether the
projector is correct and whether the data is what we assume.
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import numpy as np
import trimesh


REAL_TRANSFORMS = Path(
    r"C:\Users\Krish Sharma\Downloads\mf3d_partial\Blender_render_images"
    r"\Almond(bowl)\almond_1\Almond_1\transforms.json"
)
REAL_MESH = Path(
    r"C:\Users\Krish Sharma\Downloads\mf3d_mesh\3D_Mesh"
    r"\Almond(bowl)\almond_1\Almond_1.obj"
)
GT_ALMOND_CM3 = 341.29


def load_camera(tf_path: Path):
    with open(tf_path) as f:
        d = json.load(f)
    angle_x = float(d["camera_angle_x"])
    T = np.asarray(d["frames"][0]["transform_matrix"], dtype=np.float64)
    return T[:3, 3], T[:3, :3], angle_x


def project(mesh_path, cam_pos, R, angle_x, size):
    t0 = time.time()
    W = H = size
    fx = (W / 2.0) / np.tan(angle_x / 2.0)
    cx = (W - 1) / 2.0
    cy = (H - 1) / 2.0

    vs, us = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    xc = (us - cx) / fx
    yc = (vs - cy) / fx
    zc = -np.ones_like(xc)
    d_cam = np.stack([xc, yc, zc], axis=-1)
    d_cam /= np.linalg.norm(d_cam, axis=-1, keepdims=True)
    d_world = d_cam @ R.T
    origins = np.broadcast_to(cam_pos, d_world.shape).reshape(-1, 3)
    dirs = d_world.reshape(-1, 3)

    mesh = trimesh.load_mesh(str(mesh_path), process=False)
    hit_loc, hit_ray_idx, _ = mesh.ray.intersects_location(
        ray_origins=origins, ray_directions=dirs, multiple_hits=False,
    )

    pts = np.full((H * W, 3), np.nan)
    if len(hit_loc):
        pts[hit_ray_idx] = hit_loc
    mask_flat = np.zeros(H * W, dtype=bool)
    if len(hit_ray_idx):
        mask_flat[hit_ray_idx] = True

    missing = ~mask_flat
    if missing.any():
        dz = dirs[missing, 2]
        t = np.where(dz < -1e-9, (0.0 - cam_pos[2]) / dz, np.nan)
        plate_pts = cam_pos[None, :] + t[:, None] * dirs[missing]
        pts[missing] = plate_pts

    deltas = pts - cam_pos[None, :]
    deltas_cam = deltas @ R
    z = -deltas_cam[:, 2]
    depth = z.reshape(H, W).astype(np.float32)
    mask = mask_flat.reshape(H, W)

    print(f"  [project] {size}x{size}: {int(mask.sum())} hits in "
          f"{time.time()-t0:.1f}s", flush=True)
    return depth, mask, fx


def check_1_cube(size: int = 512) -> bool:
    print()
    print("=" * 72)
    print("CHECK 1 — ANALYTIC CUBE")
    print("=" * 72)

    side = (0.10, 0.10, 0.05)
    expected_cm3 = side[0] * side[1] * side[2] * 1e6
    print(f"  cube:            {side[0]*100:.0f} x {side[1]*100:.0f} "
          f"x {side[2]*100:.0f} cm")
    print(f"  expected volume: {expected_cm3:.4f} cm^3")

    box = trimesh.creation.box(extents=side)
    box.apply_translation((0.0, 0.0, side[2] / 2.0))

    print(f"  trimesh volume:  {box.volume * 1e6:.4f} cm^3")

    with tempfile.NamedTemporaryFile(suffix=".obj", delete=False) as f:
        tmp_path = Path(f.name)
    box.export(str(tmp_path))

    cam_pos, R, angle_x = load_camera(REAL_TRANSFORMS)
    depth, mask, fx = project(tmp_path, cam_pos, R, angle_x, size=size)
    tmp_path.unlink(missing_ok=True)

    food = depth[mask]
    if food.size:
        print(f"  food z-depth: min={food.min():.4f} "
              f"med={np.median(food):.4f} max={food.max():.4f} m")

    from vision_service.inference.volume import estimate_volume_cm3
    try:
        res = estimate_volume_cm3(depth, mask, fx)
    except Exception as exc:
        print(f"  formula raised: {exc}")
        return False

    measured = res["volume_cm3"]
    ratio = measured / expected_cm3
    print(f"  measured volume: {measured:.4f} cm^3")
    print(f"  ratio:           {ratio:.6f}")
    print(f"  target:          1.000000 +/- 0.050000")
    ok = abs(ratio - 1.0) <= 0.05
    print(f"  [{'PASS' if ok else 'FAIL'}]")
    return ok


def check_2_almond_mesh():
    print()
    print("=" * 72)
    print("CHECK 2 — REAL ALMOND MESH, DIRECT VOLUME")
    print("=" * 72)
    if not REAL_MESH.exists():
        print(f"  missing: {REAL_MESH}")
        return

    m = trimesh.load_mesh(str(REAL_MESH), process=False)
    print(f"  watertight:  {m.is_watertight}")
    print(f"  vertices:    {len(m.vertices)}")
    print(f"  triangles:   {len(m.triangles)}")
    print(f"  bbox min:    {m.bounds[0].round(4).tolist()}")
    print(f"  bbox max:    {m.bounds[1].round(4).tolist()}")

    try:
        v = m.volume * 1e6
        print(f"  mesh volume:        {v:.4f} cm^3")
        print(f"  ratio vs GT:        {v / GT_ALMOND_CM3:.4f}")
    except Exception as exc:
        print(f"  mesh volume failed: {exc}")

    try:
        vh = m.convex_hull.volume * 1e6
        print(f"  convex hull volume: {vh:.4f} cm^3")
        print(f"  ratio vs GT:        {vh / GT_ALMOND_CM3:.4f}")
    except Exception as exc:
        print(f"  hull volume failed: {exc}")

    print(f"  GT (Excel):         {GT_ALMOND_CM3:.4f} cm^3")


if __name__ == "__main__":
    ok = check_1_cube(size=512)
    check_2_almond_mesh()
    print()
    print("=" * 72)
    print(f"CHECK 1: {'PASS' if ok else 'FAIL'}")
    print("=" * 72)