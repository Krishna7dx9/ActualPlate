"""
Mesh-based ground-truth depth projection for MetaFood3D.

The 3D_Mesh subfolder contains each food's surface as a Wavefront
.obj file, in world coordinates, in meters, z-up, sitting on the
plate at z = 0. Camera pose and FOV are in transforms.json. Ray-
casting the mesh through the camera produces per-pixel metric
depth with no encoding to recover.

Output: (depth_map_meters, food_mask), both at the requested
resolution. depth_map is z-depth in the camera frame.
food_mask is True where the mesh was hit.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import trimesh


# Mesh cache: (path, mtime) -> trimesh.Trimesh
_MESH_CACHE: dict[tuple[str, float], trimesh.Trimesh] = {}


def _load_camera(transforms_path: Path):
    """Return (cam_pos_world, R_world_from_cam, camera_angle_x_rad)."""
    with open(transforms_path, "r") as f:
        data = json.load(f)
    angle_x = float(data["camera_angle_x"])
    first = data["frames"][0]
    T = np.asarray(first["transform_matrix"], dtype=np.float64)
    R = T[:3, :3]
    pos = T[:3, 3]
    return pos, R, angle_x


def _load_mesh(mesh_path: Path) -> trimesh.Trimesh:
    key = (str(mesh_path), mesh_path.stat().st_mtime)
    if key in _MESH_CACHE:
        return _MESH_CACHE[key]
    mesh = trimesh.load_mesh(str(mesh_path), process=False)
    _MESH_CACHE.clear()
    _MESH_CACHE[key] = mesh
    return mesh


def project_depth(
    mesh_path,
    transforms_path,
    image_size: tuple[int, int] = (1024, 1024),
    plate_z: float = 0.0,
    chunk: int = 4096,
):
    """
    Ray-cast the mesh through the camera, chunked to bound memory.

    trimesh without embree builds a (rays x triangles) intermediate
    matrix. At 1024x1024 and 650k triangles that is ~1.9 GB per
    array, and there are three of them. Chunking to 65k rays bounds
    peak memory to ~120 MB per intermediate.

    Returns
    -------
    depth_map : (H, W) float32, meters, z-depth in camera frame
    mask      : (H, W) bool, True where the mesh was hit (food)
    """
    t0 = time.time()
    mesh_path = Path(mesh_path)
    transforms_path = Path(transforms_path)
    W, H = image_size

    cam_pos, R, angle_x = _load_camera(transforms_path)
    fx = (W / 2.0) / np.tan(angle_x / 2.0)
    cx = (W - 1) / 2.0
    cy = (H - 1) / 2.0

    vs, us = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    x_cam = (us - cx) / fx
    y_cam = (vs - cy) / fx
    z_cam = -np.ones_like(x_cam)

    dir_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)
    dir_cam /= np.linalg.norm(dir_cam, axis=-1, keepdims=True)
    dir_world = dir_cam @ R.T
    dirs_all = dir_world.reshape(-1, 3)
    n_rays = dirs_all.shape[0]

    print(f"  [project] rays prepared: {n_rays} in {time.time()-t0:.1f}s",
          flush=True)

    t1 = time.time()
    mesh = _load_mesh(mesh_path)
    print(f"  [project] mesh loaded: {len(mesh.triangles)} triangles in "
          f"{time.time()-t1:.1f}s", flush=True)

    pts = np.full((n_rays, 3), np.nan, dtype=np.float64)
    mask_flat = np.zeros(n_rays, dtype=bool)

    origins_base = np.broadcast_to(cam_pos, (chunk, 3))

    t2 = time.time()
    n_chunks = (n_rays + chunk - 1) // chunk
    print(f"  [project] ray-casting {n_rays} rays in {n_chunks} chunks "
          f"of {chunk}...", flush=True)

    for i in range(0, n_rays, chunk):
        end = min(i + chunk, n_rays)
        size = end - i
        origins = origins_base[:size]
        dirs = dirs_all[i:end]

        hit_loc, hit_ray_idx, _ = mesh.ray.intersects_location(
            ray_origins=origins,
            ray_directions=dirs,
            multiple_hits=False,
        )
        if len(hit_loc):
            pts[i + hit_ray_idx] = hit_loc
            mask_flat[i + hit_ray_idx] = True

        done = end
        pct = 100.0 * done / n_rays
        elapsed = time.time() - t2
        rate = done / elapsed if elapsed > 0 else 0
        eta = (n_rays - done) / rate if rate > 0 else 0
        print(f"  [project] {done}/{n_rays} ({pct:.1f}%) "
              f"{rate:.0f} ray/s, eta {eta:.0f}s", flush=True)

    print(f"  [project] ray-cast done in {time.time()-t2:.1f}s, "
          f"{int(mask_flat.sum())} hits", flush=True)

    missing = ~mask_flat
    if missing.any():
        dz = dirs_all[missing, 2]
        t = np.where(dz < -1e-9, (plate_z - cam_pos[2]) / dz, np.nan)
        plate_pts = cam_pos[None, :] + t[:, None] * dirs_all[missing]
        pts[missing] = plate_pts

    deltas = pts - cam_pos[None, :]
    deltas_cam = deltas @ R
    depth_flat = -deltas_cam[:, 2]

    depth_map = depth_flat.reshape(H, W).astype(np.float32)
    mask = mask_flat.reshape(H, W)

    print(f"  [project] total: {time.time()-t0:.1f}s", flush=True)
    return depth_map, mask


def _cli():
    import argparse
    p = argparse.ArgumentParser(prog="mesh_projector")
    p.add_argument("--mesh", required=True)
    p.add_argument("--transforms", required=True)
    p.add_argument("--gt-volume", type=float, required=True,
                   help="Ground-truth volume in cm^3 (from the Excel).")
    p.add_argument("--size", type=int, default=1024,
                   help="Projection resolution (square). Lower is faster.")
    args = p.parse_args()

    depth, mask = project_depth(
        args.mesh, args.transforms,
        image_size=(args.size, args.size),
    )

    print(f"depth shape: {depth.shape}")
    print(f"depth min/median/max: "
          f"{depth.min():.4f} / {np.median(depth):.4f} / {depth.max():.4f} m")

    food = depth[mask]
    if food.size:
        print(f"food-only depth min/median/max: "
              f"{food.min():.4f} / {np.median(food):.4f} / {food.max():.4f} m")
    print(f"mask pixels: {int(mask.sum())}")

    _, _, angle_x = _load_camera(Path(args.transforms))
    fx = (args.size / 2.0) / np.tan(angle_x / 2.0)
    print(f"focal from camera_angle_x at size {args.size}: {fx:.4f} px")

    from vision_service.inference.volume import estimate_volume_cm3
    res = estimate_volume_cm3(depth, mask, fx)
    ratio = res["volume_cm3"] / args.gt_volume
    print(f"measured volume: {res['volume_cm3']:.4f} cm^3")
    print(f"GT volume:       {args.gt_volume:.4f} cm^3")
    print(f"ratio:           {ratio:.6f}")
    print(f"target:          1.000000 +/- 0.050000")
    print(f"[{'PASS' if abs(ratio - 1.0) <= 0.05 else 'FAIL'}]")


if __name__ == "__main__":
    _cli()