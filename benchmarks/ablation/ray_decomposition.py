"""
Per-ray decomposition on the analytic box.

For the 10 x 10 x 5 cm box at the MetaFood3D camera pose, compute
per ray:
  r_entry : distance from camera to first hit on box
  r_exit  : distance from camera to last hit on box (before exit)
  r_plate : distance from camera to plate plane z=0 intersection

Three volumes:
  V_exit  = sum (r_exit^3 - r_entry^3) / 3 * dOmega   -> should be 500
  V_plate = sum (r_plate^3 - r_entry^3) / 3 * dOmega  -> should be 755
  V_shadow = V_plate - V_exit                          -> the over-count

Also report the fraction of rays with r_exit < r_plate - 1e-6.
"""

from __future__ import annotations
import json
import numpy as np
import trimesh
from pathlib import Path


BOX = trimesh.creation.box(extents=(0.10, 0.10, 0.05))
BOX.apply_translation((0.0, 0.0, 0.025))

TRANSFORMS = Path(r"C:\Users\Krish Sharma\Downloads\mf3d_partial"
                  r"\Blender_render_images\Almond(bowl)\almond_1"
                  r"\Almond_1\transforms.json")


def load_camera(tf_path):
    with open(tf_path) as f:
        d = json.load(f)
    angle_x = float(d["camera_angle_x"])
    T = np.asarray(d["frames"][0]["transform_matrix"], dtype=np.float64)
    return T[:3, 3], T[:3, :3], angle_x


def main():
    cam_pos, R, angle_x = load_camera(TRANSFORMS)
    W = H = 256
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
    dirs = d_world.reshape(-1, 3)
    origins = np.broadcast_to(cam_pos, dirs.shape)

    # Collect ALL hits per ray.
    loc, ray_idx, _ = BOX.ray.intersects_location(
        ray_origins=origins, ray_directions=dirs, multiple_hits=True,
    )
    print(f"total hits: {len(loc)}")

    # Per ray: first and last hit distance.
    r_entry = np.full(len(dirs), np.inf)
    r_exit = np.full(len(dirs), -np.inf)
    if len(ray_idx):
        t_all = np.linalg.norm(loc - cam_pos[None, :], axis=1)
        for k in range(len(ray_idx)):
            ri = ray_idx[k]
            if t_all[k] < r_entry[ri]:
                r_entry[ri] = t_all[k]
            if t_all[k] > r_exit[ri]:
                r_exit[ri] = t_all[k]

    # Plate distance.
    dz = dirs[:, 2]
    t_plate = np.where(dz < -1e-9, (0.0 - cam_pos[2]) / dz, np.nan)

    mask = np.isfinite(r_entry) & np.isfinite(r_exit) & (r_exit > r_entry)
    print(f"rays hitting box: {mask.sum()}")

    # Solid angle per pixel.
    rn = np.linalg.norm(d_cam, axis=-1).ravel()
    dOmega = 1.0 / (fx ** 2) / (rn ** 3)

    # Volumes.
    V_exit = np.sum((r_exit[mask] ** 3 - r_entry[mask] ** 3) / 3.0
                    * dOmega[mask]) * 1e6
    V_plate = np.sum((t_plate[mask] ** 3 - r_entry[mask] ** 3) / 3.0
                     * dOmega[mask]) * 1e6

    print(f"V_exit  (true volume)     : {V_exit:.4f} cm^3")
    print(f"V_plate (current formula) : {V_plate:.4f} cm^3")
    print(f"V_shadow (over-count)     : {V_plate - V_exit:.4f} cm^3")
    print(f"fraction rays where r_exit < r_plate: "
          f"{(r_exit[mask] < t_plate[mask] - 1e-6).mean():.4f}")


if __name__ == "__main__":
    main()