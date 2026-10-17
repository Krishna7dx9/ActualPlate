"""
Synthetic cube test. True volume of a 10 x 10 x 5 cm box is 500 cm3.

Renders the box depth map synthetically (no GPU needed), runs both
estimate_volume_cm3_v2 (ray-to-plane) and estimate_volume_cm3_v3
(column integration) on it, and prints the ratio for each.

Gate for phase 3: v3 ratio < 1.05. v2 measured 1.51 on the same input.
"""
import numpy as np
from vision_service.inference.volume import (
    estimate_volume_cm3_v2,
    estimate_volume_cm3_v3,
)


def make_box_depth(
    h_px=400, w_px=400,
    box_side_m=0.10, box_height_m=0.05,
    cam_distance_m=0.60, focal_px=800.0,
):
    """Render a top-down view of a square box sitting on a flat plate."""
    plate_z = cam_distance_m
    depth = np.full((h_px, w_px), plate_z, dtype=np.float32)

    cx = (w_px - 1) / 2.0
    cy = (h_px - 1) / 2.0

    # Box footprint in pixels at plate depth
    box_half_px = (box_side_m / 2.0) * focal_px / plate_z
    x1 = int(round(cx - box_half_px))
    x2 = int(round(cx + box_half_px))
    y1 = int(round(cy - box_half_px))
    y2 = int(round(cy + box_half_px))

    # Inside box: depth decreases by box height
    depth[y1:y2, x1:x2] = plate_z - box_height_m

    mask = np.zeros((h_px, w_px), dtype=bool)
    mask[y1:y2, x1:x2] = True

    return depth, mask, focal_px, box_side_m ** 2 * box_height_m * 1e6


def main():
    depth, mask, focal, true_cm3 = make_box_depth()
    print(f"True volume: {true_cm3:.1f} cm^3")
    print(f"Depth shape: {depth.shape}")
    print(f"Mask pixels: {int(mask.sum())}")
    print(f"Focal: {focal}")
    print()

    # v2 (ray-to-plane)
    try:
        r2 = estimate_volume_cm3_v2(
            depth, mask, focal, label="box", return_debug=True,
        )
        v2 = r2["volume_cm3"]
        print(f"v2 (ray-to-plane):  {v2:>8.1f} cm^3  ratio={v2/true_cm3:.3f}")
    except Exception as e:
        print(f"v2 failed: {e}")
        v2 = None

    # v3 (column integration)
    try:
        r3 = estimate_volume_cm3_v3(
            depth, mask, focal, label="box", return_debug=True,
        )
        v3 = r3["volume_cm3"]
        print(f"v3 (column grid):   {v3:>8.1f} cm^3  ratio={v3/true_cm3:.3f}")
    except Exception as e:
        print(f"v3 failed: {e}")
        v3 = None

    print()
    if v2 and v3:
        print(f"v2 over-count: {(v2/true_cm3 - 1)*100:+.1f}%")
        print(f"v3 over-count: {(v3/true_cm3 - 1)*100:+.1f}%")
        if v3 / true_cm3 < 1.05:
            print("PASS: v3 within 5%")
        else:
            print("FAIL: v3 outside 5%")


if __name__ == "__main__":
    main()
