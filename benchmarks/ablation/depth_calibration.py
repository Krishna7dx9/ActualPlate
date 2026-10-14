"""
Depth scale calibration -- Phase 1.

The MetaFood3D adapter normalises the Blender depth PNG to the
arbitrary range [0.9, 1.0] metres.  Real Blender renders have a
metric near/far mapping that is not recorded in transforms.json.
Without it, the "GT depth" is not in metres.

This script sweeps a linear scale factor k on top of the current
normalised depth and reports, per item, the k that makes the C
variant (GT depth + GT mask + true focal) return exactly the
ground-truth volume.

If k is consistent across items, it is the missing metric scale.
If k varies, the depth PNG is not a consistent physical
measurement and cannot be used as a metric ground truth.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import numpy as np

from vision_service.inference.volume import estimate_volume_cm3


def _banner(t):
    print()
    print("=" * 72)
    print(t)
    print("=" * 72)


def find_scale_for_item(item, scale_lo=0.1, scale_hi=3.0, n=300):
    gt_depth = item.gt_depth_map
    gt_mask = item.gt_mask
    focal = float(item.focal_length_px)
    target = float(item.gt_volume_cm3)

    scales = np.geomspace(scale_lo, scale_hi, n)
    ratios = np.full_like(scales, np.nan)

    for i, k in enumerate(scales):
        try:
            r = estimate_volume_cm3(gt_depth * k, gt_mask, focal)
            ratios[i] = r["volume_cm3"] / target
        except Exception:
            pass

    # Find the first crossing of 1.0.
    crossings = np.where(np.diff(np.sign(ratios - 1.0)))[0]
    if len(crossings) == 0:
        return None, scales, ratios

    idx = crossings[0]
    r1, r2 = ratios[idx], ratios[idx + 1]
    s1, s2 = scales[idx], scales[idx + 1]
    if r2 == r1:
        return float(s1), scales, ratios
    k = s1 + (1.0 - r1) * (s2 - s1) / (r2 - r1)
    return float(k), scales, ratios


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--scale-lo", type=float, default=0.1)
    p.add_argument("--scale-hi", type=float, default=3.0)
    p.add_argument("--steps", type=int, default=300)
    args = p.parse_args(argv)

    from benchmarks.ablation.metafood3d import MetaFood3DAdapter
    adapter = MetaFood3DAdapter()
    items = adapter.load_items(limit=args.limit)

    _banner("DEPTH SCALE CALIBRATION -- PHASE 1")
    print(f"  items:  {len(items)}")
    print(f"  sweep:  k in [{args.scale_lo}, {args.scale_hi}], {args.steps} steps")
    print()

    ks = []
    for i, item in enumerate(items, 1):
        k, scales, ratios = find_scale_for_item(
            item, args.scale_lo, args.scale_hi, args.steps
        )
        if k is None:
            print(f"  [{i}/{len(items)}] {item.id}  -- no crossing found")
            continue
        ks.append(k)
        print(f"  [{i}/{len(items)}] {item.id}  k = {k:.6f}")

    if not ks:
        print()
        print("No items calibrated. Check the adapter or the sweep range.")
        return 1

    _banner("SCALE CONSISTENCY")
    median = statistics.median(ks)
    mean = statistics.mean(ks)
    stdev = statistics.stdev(ks) if len(ks) > 1 else 0.0

    print(f"  n:        {len(ks)}")
    print(f"  median:   {median:.6f}")
    print(f"  mean:     {mean:.6f}")
    print(f"  std:      {stdev:.6f}")
    print(f"  std/mean: {stdev / mean:.6f}")
    print()

    if stdev / mean < 0.10:
        print(f"  [PASS] Scale is consistent.  Recommended value: {median:.6f}")
        print(f"         Set MF3D_DEPTH_SCALE={median:.6f} in the adapter.")
        return 0
    else:
        print(f"  [FAIL] Scale is not consistent across items.")
        print(f"         GT depth map is not a single metric scale.")
        return 1


if __name__ == "__main__":
    sys.exit(main())