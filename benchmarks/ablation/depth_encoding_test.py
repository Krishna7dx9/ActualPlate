"""
Depth encoding test -- Phase 1b.

Phase 1 showed the adapter's per-item normalization produces
inconsistent scale factors k (std/mean = 0.28). That is the
symptom of a broken metric encoding.

This test tries several GLOBAL mappings of the raw PNG value to
depth, and checks whether any produces a consistent k across items.

A mapping is consistent if the k required to make the C variant
return the ground-truth volume is the same for every item to
within 10%.

Mappings tested:
  adapter            current per-item normalization, baseline
  raw_div_255        raw / 255
  inv_raw_div_255    (255 - raw) / 255
  linear_from_far    (raw - raw.min()) / (255 - raw.min())
  quadratic          (raw / 255) ** 2
  inverse_linear     1 / (1 + raw / 255)
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from vision_service.inference.volume import estimate_volume_cm3


def load_raw_png(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        arr = np.array(im, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr


def find_depth_png(item) -> Path | None:
    rgb = Path(item.rgb_path)
    original_dir = rgb.parent
    inner_dir = original_dir.parent
    depth_dir = inner_dir / "Depth"
    if not depth_dir.exists():
        return None
    pngs = sorted(depth_dir.glob("*.png"))
    return pngs[0] if pngs else None


# -- Mappings ---------------------------------------------------------


def m_adapter(raw):
    v_min = float(raw.min())
    v_max = float(raw.max())
    return 0.9 + 0.1 * (raw - v_min) / max(v_max - v_min, 1e-9)


def m_raw_div_255(raw):
    return raw / 255.0


def m_inv_raw_div_255(raw):
    return (255.0 - raw) / 255.0


def m_linear_from_far(raw):
    v_min = float(raw.min())
    return (raw - v_min) / max(255.0 - v_min, 1e-9)


def m_quadratic(raw):
    return (raw / 255.0) ** 2


def m_inverse_linear(raw):
    return 1.0 / (1.0 + raw / 255.0)


MAPPINGS = {
    "adapter": m_adapter,
    "raw_div_255": m_raw_div_255,
    "inv_raw_div_255": m_inv_raw_div_255,
    "linear_from_far": m_linear_from_far,
    "quadratic": m_quadratic,
    "inverse_linear": m_inverse_linear,
}


def find_k(depth, mask, focal, target_volume,
           k_lo=0.001, k_hi=1000.0, steps=400):
    ks = np.geomspace(k_lo, k_hi, steps)
    vs = np.full_like(ks, np.nan)
    for i, k in enumerate(ks):
        try:
            r = estimate_volume_cm3(depth * k, mask, focal)
            vs[i] = float(r["volume_cm3"])
        except Exception:
            pass

    diff = vs - target_volume
    for i in range(len(diff) - 1):
        a, b = diff[i], diff[i + 1]
        if np.isnan(a) or np.isnan(b):
            continue
        if a == 0:
            return float(ks[i])
        if a * b < 0:
            k1, k2 = ks[i], ks[i + 1]
            v1, v2 = vs[i], vs[i + 1]
            if v2 == v1:
                return float(k1)
            return float(k1 + (target_volume - v1) * (k2 - k1) / (v2 - v1))
    return None


def main(argv=None):
    p = argparse.ArgumentParser(prog="depth_encoding_test")
    p.add_argument("--limit", type=int, default=20)
    args = p.parse_args(argv)

    from benchmarks.ablation.metafood3d import MetaFood3DAdapter
    adapter = MetaFood3DAdapter()
    items = adapter.load_items(limit=args.limit)

    print("=" * 72)
    print("DEPTH ENCODING TEST -- PHASE 1b")
    print("=" * 72)
    print(f"  items:    {len(items)}")
    print(f"  mappings: {list(MAPPINGS.keys())}")
    print()

    results = {name: [] for name in MAPPINGS}
    missing_depth = 0

    for i, item in enumerate(items, 1):
        print(f"[{i}/{len(items)}] {item.id}", flush=True)
        depth_png = find_depth_png(item)
        if depth_png is None:
            print("    no depth PNG found; skipping")
            missing_depth += 1
            continue

        raw = load_raw_png(depth_png)
        mask = item.gt_mask
        focal = float(item.focal_length_px)
        target = float(item.gt_volume_cm3)

        line = []
        for name, fn in MAPPINGS.items():
            try:
                depth = fn(raw).astype(np.float32)
                k = find_k(depth, mask, focal, target)
            except Exception as exc:
                k = None
            if k is not None:
                results[name].append(k)
                line.append(f"{name}={k:.3f}")
            else:
                line.append(f"{name}=ERR")
        print("    " + "  ".join(line))

    print()
    print("=" * 72)
    print("CONSISTENCY PER MAPPING")
    print("=" * 72)
    print(f"  {'mapping':<20} {'n':>4} {'median':>10} {'mean':>10} {'std':>10} {'std/mean':>10}")
    best = None
    for name, ks in results.items():
        if len(ks) < 2:
            print(f"  {name:<20} {len(ks):>4}  (insufficient data)")
            continue
        med = statistics.median(ks)
        mean = statistics.mean(ks)
        std = statistics.stdev(ks)
        ratio = std / mean if mean else float("inf")
        marker = ""
        if ratio < 0.10:
            marker = "  <-- consistent"
            if best is None or ratio < best[1]:
                best = (name, ratio, med)
        print(f"  {name:<20} {len(ks):>4} {med:>10.4f} {mean:>10.4f} {std:>10.4f} {ratio:>10.4f}{marker}")

    print()
    if best is not None:
        name, ratio, med = best
        print(f"  [PASS] Mapping '{name}' gives consistent k (std/mean={ratio:.4f}).")
        print(f"         Recommended global scale: k = {med:.6f}")
        print(f"         Set MF3D_DEPTH_MODE={name} and MF3D_DEPTH_SCALE={med:.6f} in the adapter.")
        return 0
    print("  [FAIL] No mapping gives consistent k.")
    print("         The PNG encoding is not a single global function of raw value.")
    print("         We need the Blender .blend file or the dataset author's spec.")
    return 1


if __name__ == "__main__":
    sys.exit(main())