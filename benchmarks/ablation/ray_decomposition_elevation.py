"""
Box volume via radial formula at three camera elevations.

Same 10x10x5 cm box on the plate. Camera looks at origin from
three positions at the same distance:

  nadir    : 90 degrees from horizontal (straight above)
  60 deg   : 60 degrees above the plate
  31 deg   : the MetaFood3D pose (camera at [0, 1.0, 0.6],
             elevation atan2(0.6, 1.0) = 31 degrees)

For each, print:
  V_exit   : true volume via mesh entries and exits
  V_plate  : current formula (integrates to the plate)
  ratio    : V_plate / V_exit
  frac     : fraction of rays that exit through a side face
             before reaching the plate

Mechanism to confirm: at nadir, every ray enters and exits
through the top and bottom faces, so frac should be ~0 and ratio
~1. At lower elevations, side-face exits increase, frac rises,
and ratio grows.
"""

from __future__ import annotations

import numpy as np
import trimesh


BOX = trimesh.creation.box(extents=(0.10, 0.10, 0.05))
BOX.apply_translation((0.0, 0.0, 0.025))

W = H = 256
FOCAL_ANGLE_X = 0.8575560450553894


def make_camera(elev_deg, dist):
    """
    Return (cam_pos, R) where R's columns are the camera axes
    expressed in world, world is z-up, and the camera looks at
    the origin. Elevation is degrees above the horizontal plane.
    """
    elev = np.deg2rad(elev_deg)
    cam_pos = np.array([
        0.0,
        dist * np.cos(elev),
        dist * np.sin(elev),
    ])

    cam_z = cam_pos / np.linalg.norm(cam_pos)   # points away from origin
    world_up = np.array([0.0, 0.0, 1.0])

    if abs(np.dot(world_up, cam_z)) > 0.999:
        cam_x = np.array([1.0, 0.0, 0.0])
    else:
        cam_x = np.cross(world_up, cam_z)
        cam_x /= np.linalg.norm(cam_x)
    cam_y = np.cross(cam_z, cam_x)

    R = np.stack([cam_x, cam_y, cam_z], axis=1)  # columns
    return cam_pos, R


def rays_for(cam_pos, R):
    fx = (W / 2.0) / np.tan(FOCAL_ANGLE_X / 2.0)
    cx = (W - 1) / 2.0
    cy = (H - 1) / 2.0
    vs, us = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    xc = (us - cx) / fx
    yc = (vs - cy) / fx
    zc = -np.ones_like(xc)
    d_cam = np.stack([xc, yc, zc], axis=-1)
    d_cam /= np.linalg.norm(d_cam, axis=-1, keepdims=True)
    d_world = d_cam @ R.T
    return d_cam.reshape(-1, 3), d_world.reshape(-1, 3), fx


def main():
    for label, elev, dist in [
        ("nadir   ", 90.0, 1.166),
        ("60 deg  ", 60.0, 1.166),
        ("31 deg  ", 31.0, 1.166),
    ]:
        cam_pos, R = make_camera(elev, dist)
        d_cam, d_world, fx = rays_for(cam_pos, R)
        origins = np.broadcast_to(cam_pos, d_world.shape)

        loc, ray_idx, _ = BOX.ray.intersects_location(
            ray_origins=origins, ray_directions=d_world,
            multiple_hits=True,
        )
        r_entry = np.full(len(d_world), np.inf)
        r_exit = np.full(len(d_world), -np.inf)
        if len(ray_idx):
            t_all = np.linalg.norm(loc - cam_pos[None, :], axis=1)
            for k in range(len(ray_idx)):
                ri = ray_idx[k]
                if t_all[k] < r_entry[ri]:
                    r_entry[ri] = t_all[k]
                if t_all[k] > r_exit[ri]:
                    r_exit[ri] = t_all[k]

        dz = d_world[:, 2]
        t_plate = np.where(dz < -1e-9, (0.0 - cam_pos[2]) / dz, np.nan)
        mask = np.isfinite(r_entry) & np.isfinite(r_exit) & (r_exit > r_entry)

        rn = np.linalg.norm(d_cam, axis=-1)
        dOmega = 1.0 / (fx ** 2) / (rn ** 3)

        if mask.sum() == 0:
            print(f"  {label}: no hits")
            continue

        V_exit = np.sum((r_exit[mask] ** 3 - r_entry[mask] ** 3) / 3.0
                        * dOmega[mask]) * 1e6
        V_plate = np.sum((t_plate[mask] ** 3 - r_entry[mask] ** 3) / 3.0
                         * dOmega[mask]) * 1e6
        frac = float((r_exit[mask] < t_plate[mask] - 1e-6).mean())
        ratio = V_plate / V_exit if V_exit > 0 else float("nan")

        print(f"  {label}: V_exit={V_exit:7.3f}  V_plate={V_plate:7.3f}  "
              f"ratio={ratio:.4f}  frac_exit_before_plate={frac:.3f}")


if __name__ == "__main__":
    main()