"""
Depth encoding search -- Phase 1c.

The readme does not document the depth-PNG encoding. We recover it
from the data itself.

Known facts:
  - Camera position: fixed at (0, 1.0, 0.6) for every item.
  - Camera-to-origin distance: sqrt(1.0^2 + 0.6^2) = 1.166 m.
  - Raw PNG max = 255 always. That is the anchor.
  - Raw PNG min varies per item. That is the food top.
  - One global mapping must produce GT volume for every item.

Two hypotheses tested:
  linear:   Z = near + (raw / 255) * (far - near)
  inverse:  Z = 1 / (1/near + (raw / 255) * (1/far - 1/near))

For each hypothesis, we sweep (near, far) and report the pair that
makes the C-variant volume ratio most consistent across items.
Consistency = std/mean of per-item ratios.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from vision_service.inference.volume import estimate_volume_cm3


def load_raw(item):
    rgb = Path(item.rgb_path)
    inner = rgb.parent.parent
    depth_dir = inner / "Depth"
    pngs = sorted(depth_dir.glob("*.png"))
    if not pngs:
        return None
    with Image.open(pngs[0]) as im:
        arr = np.array(im, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr


def z_linear(raw, near, far):
    return near + (raw / 255.0) * (far - near)


def z_inverse(raw, near, far):
    inv = (1.0 / near) + (raw / 255.0) * (1.0 / far - 1.0 / near)
    return 1.0 / np.maximum(inv, 1e-9)


def evaluate(items, raw_list, mapping, near, far):
    ratios = []
    for item, raw in zip(items, raw_list):
        Z = mapping(raw, near, far).astype(np.float32)
        try:
            r = estimate_volume_cm3(Z, item.gt_mask, float(item.focal_length_px))
            ratios.append(r["volume_cm3"] / float(item.gt_volume_cm3))
        except Exception:
            ratios.append(np.nan)
    ratios = [r for r in ratios if np.isfinite(r)]
    if len(ratios) < 5:
        return None
    med = statistics.median(ratios)
    mean = statistics.mean(ratios)
    std = statistics.stdev(ratios)
    return {
        "n": len(ratios),
        "median": med,
        "mean": mean,
        "std": std,
        "cv": std / mean if mean else float("inf"),
    }


def main(argv=None):
    p = argparse.ArgumentParser(prog="depth_encoding_search")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--near-steps", type=int, default=15)
    p.add_argument("--far-steps", type=int, default=4)
    args = p.parse_args(argv)

    from benchmarks.ablation.metafood3d import MetaFood3DAdapter
    adapter = MetaFood3DAdapter()
    items = adapter.load_items(limit=args.limit)

    print("=" * 72)
    print("DEPTH ENCODING SEARCH -- PHASE 1c")
    print("=" * 72)
    print(f"  items: {len(items)}")

    raw_list = []
    keep = []
    for item in items:
        raw = load_raw(item)
        if raw is None:
            print(f"  skip {item.id}: no depth PNG")
            continue
        raw_list.append(raw)
        keep.append(item)
    items = keep
    print(f"  loaded: {len(items)}")

    near_vals = np.linspace(0.90, 1.15, args.near_steps)
    far_vals = np.linspace(1.15, 1.30, args.far_steps)

    results = []

    for hyp, mapping in (("linear", z_linear), ("inverse", z_inverse)):
        print()
        print(f"  hypothesis: {hyp}")
        print(f"  {'near':>8} {'far':>8} {'median':>10} {'cv':>10}")
        best = None
        for near in near_vals:
            for far in far_vals:
                if near >= far - 0.01:
                    continue
                r = evaluate(items, raw_list, mapping, float(near), float(far))
                if r is None:
                    continue
                marker = ""
                if best is None or r["cv"] < best["cv"]:
                    best = {**r, "near": float(near), "far": float(far), "hyp": hyp}
                    marker = "  *"
                print(f"  {near:>8.4f} {far:>8.4f} "
                      f"{r['median']:>10.4f} {r['cv']:>10.4f}{marker}")
        if best:
            results.append(best)
            print(f"  -> best {hyp}: near={best['near']:.4f} far={best['far']:.4f} "
                  f"median={best['median']:.4f} cv={best['cv']:.4f}")

    print()
    print("=" * 72)
    print("VERDICT")
    print("=" * 72)
    results.sort(key=lambda x: x["cv"])
    for r in results:
        tag = "PASS" if r["cv"] < 0.10 else ("marginal" if r["cv"] < 0.20 else "FAIL")
        print(f"  [{tag:>8}] {r['hyp']:<8} near={r['near']:.4f} far={r['far']:.4f} "
              f"cv={r['cv']:.4f} median={r['median']:.4f}")

    if results and results[0]["cv"] < 0.10:
        best = results[0]
        print()
        print(f"  Encoding recovered: {best['hyp']} with near={best['near']:.6f}, "
              f"far={best['far']:.6f}")
        return 0
    print()
    print("  No consistent mapping found.")
    print("  Next: download the 3D_Mesh subfolder (contains per-item geometry")
    print("        that fixes the near/far). Or contact the authors.")
    return 1


if __name__ == "__main__":
    sys.exit(main())