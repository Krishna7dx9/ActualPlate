"""
P1: validate the column estimator on exact 3D shapes.
No raytracing needed: we build the depth map from the 3D shape.
"""
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from vision_service.inference.volume import estimate_volume_cm3_v4

F_PX = 800.0
H_PX, W_PX = 400, 400
CAM_DIST = 0.60

def make_slab(thickness_m=0.01, side_m=0.10):
    depth = np.full((H_PX, W_PX), CAM_DIST, dtype=np.float32)
    cx, cy = (W_PX - 1) / 2.0, (H_PX - 1) / 2.0
    half_px = (side_m / 2.0) * F_PX / CAM_DIST
    mask = np.zeros((H_PX, W_PX), dtype=bool)
    mask[int(cy-half_px):int(cy+half_px), int(cx-half_px):int(cx+half_px)] = True
    depth[mask] = CAM_DIST - thickness_m
    true_vol = (side_m ** 2) * thickness_m * 1e6
    return depth, mask, true_vol

def make_dome(radius_m=0.05, height_m=0.03):
    depth = np.full((H_PX, W_PX), CAM_DIST, dtype=np.float32)
    cx, cy = (W_PX - 1) / 2.0, (H_PX - 1) / 2.0
    r_px = radius_m * F_PX / CAM_DIST
    ys, xs = np.mgrid[0:H_PX, 0:W_PX]
    dist = np.sqrt((xs - cx)**2 + (ys - cy)**2)
    mask = dist < r_px
    
    h = height_m * np.sqrt(np.clip(1 - (dist[mask] / r_px)**2, 0, 1))
    depth[mask] = CAM_DIST - h
    true_vol = (2/3) * np.pi * radius_m**2 * height_m * 1e6
    return depth, mask, true_vol

def run_case(name, depth, mask, true_vol):
    try:
        res = estimate_volume_cm3_v4(depth, mask, F_PX, label="", adaptive_factor=0.0, min_threshold_m=0.0001)
        pred = float(res["volume_cm3"])
        ratio = pred / true_vol
    except Exception as e:
        print(f"  {name:<8} FAILED: {type(e).__name__}: {e}")
        return None
    status = "OK" if 0.95 <= ratio <= 1.05 else "OUT"
    print(f"  {name:<8} true={true_vol:>8.1f}  pred={pred:>8.1f}  ratio={ratio:>6.3f}  {status}")
    return ratio

def main():
    print("=" * 74)
    print("P1: ESTIMATOR VALIDATION")
    print("Gate: every ratio in [0.95, 1.05]")
    print("=" * 74)
    print()

    results = []
    d, m, tv = make_slab()
    r = run_case("slab", d, m, tv)
    if r is not None: results.append(r)

    d, m, tv = make_dome()
    r = run_case("dome", d, m, tv)
    if r is not None: results.append(r)

    print("=" * 74)
    if not results:
        print("FAIL: no cases completed")
        sys.exit(1)
    passes = sum(1 for r in results if 0.95 <= r <= 1.05)
    fails = len(results) - passes
    print(f"PASS: {passes}/{len(results)}   FAIL: {fails}/{len(results)}")
    print("=" * 74)
    if fails > 0: sys.exit(1)

if __name__ == "__main__":
    main()
