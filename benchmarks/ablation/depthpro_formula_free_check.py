"""
Formula-free check on Depth Pro's absolute accuracy on real photos.

For each image:

  1. Send to Colab /detect. Get back depth_map (meters), mask, focal.
  2. Measure w_px = the bagel's pixel width (horizontal extent of
     the mask at its widest row).
  3. Compute Z_implied = f_px * D_bagel_m / w_px.
     D_bagel is the physical diameter of a typical bagel. The
     SimpleFood45 bagels are ~10 cm. This is a fixed prior, not
     measured from the image.
  4. Read Z_depthpro = median of Depth Pro's depth values inside
     the mask.
  5. Compare. If Depth Pro is accurate, |Z_depthpro - Z_implied|
     / Z_implied should be small (the spread comes from bagel
     size variation, focal estimation, and edge effects).

No ray-plane formula involved. This isolates whether Depth Pro's
metric depth is correct on real photos. Nothing else can explain
a large gap.

Requires a live Colab tunnel and VISION_INFERENCE_ENDPOINT in .env.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from benchmarks.ablation.remote import predictions_from_endpoint
from benchmarks.ablation.simplefood45 import SimpleFood45Adapter


# Known bagel diameter. The SimpleFood45 bagels are all the same
# physical object, ~10 cm across. If a different food class is
# tested, change this.
D_BAGEL_M = 0.10

TOLERANCE = 0.15  # 15 percent agreement threshold


def mask_width(mask: np.ndarray) -> int:
    """Return the widest horizontal run of True pixels in the mask."""
    row_counts = mask.sum(axis=1)
    if row_counts.size == 0 or row_counts.max() == 0:
        return 0
    return int(row_counts.max())


def check_one(item, endpoint):
    preds = predictions_from_endpoint(item, endpoint, timeout_s=300)
    depth = preds.depth_map
    mask = preds.mask
    f_px = preds.focal_length_px

    if mask.sum() < 100:
        return None

    w_px = mask_width(mask)
    if w_px < 10:
        return None

    z_implied = f_px * D_BAGEL_M / w_px
    z_depthpro = float(np.median(depth[mask]))

    rel_err = abs(z_depthpro - z_implied) / z_implied

    return {
        "item_id": item.id,
        "f_px": float(f_px),
        "w_px": int(w_px),
        "z_implied_m": float(z_implied),
        "z_depthpro_m": float(z_depthpro),
        "rel_err": float(rel_err),
    }


def main():
    endpoint = os.environ.get("VISION_INFERENCE_ENDPOINT", "").strip()
    if not endpoint:
        print("Set VISION_INFERENCE_ENDPOINT in .env or the environment.")
        sys.exit(1)
    if not endpoint.endswith("/detect"):
        endpoint = endpoint.rstrip("/") + "/detect"

    print("=" * 72)
    print("FORMULA-FREE DEPTH PRO CHECK")
    print("=" * 72)
    print(f"  endpoint: {endpoint}")
    print(f"  D_bagel prior: {D_BAGEL_M*100:.1f} cm")
    print(f"  tolerance:    +/- {TOLERANCE*100:.0f}%")
    print()

    adapter = SimpleFood45Adapter()
    items = adapter.load_items(limit=5, category="bagel")

    print(f"  items to check: {len(items)}")
    print()

    results = []
    for i, item in enumerate(items, 1):
        print(f"[{i}/{len(items)}] {item.id}", flush=True)
        try:
            r = check_one(item, endpoint)
        except Exception as exc:
            print(f"    failed: {exc}")
            continue
        if r is None:
            print(f"    skipped: mask too small or unusable")
            continue
        results.append(r)
        print(
            f"    f_px={r['f_px']:.1f}  w_px={r['w_px']}  "
            f"Z_implied={r['z_implied_m']:.3f} m  "
            f"Z_depthpro={r['z_depthpro_m']:.3f} m  "
            f"err={r['rel_err']*100:.1f}%"
        )

    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    if not results:
        print("  no items processed")
        return

    errors = [r["rel_err"] for r in results]
    n_ok = sum(1 for e in errors if e <= TOLERANCE)
    n_bad = sum(1 for e in errors if e > TOLERANCE)

    print(f"  items:      {len(results)}")
    print(f"  within 15%: {n_ok}")
    print(f"  over 15%:   {n_bad}")
    print(f"  median rel err: {np.median(errors)*100:.1f}%")
    print(f"  min / max err:  {min(errors)*100:.1f}% / {max(errors)*100:.1f}%")

    print()
    if n_ok >= len(results) - 1:
        print("  [PASS] Depth Pro's absolute metric depth agrees with")
        print("         the formula-free estimate on real photos.")
        print("         The pipeline's volume error is NOT from Depth Pro.")
        print("         Look at the formula and the plane fit.")
    elif n_bad >= len(results) - 1:
        print("  [FAIL] Depth Pro's absolute metric depth disagrees with")
        print("         the formula-free estimate on real photos.")
        print("         Depth Pro is confirmed as a source of error.")
    else:
        print("  [MIXED] Depth Pro agrees on some items, disagrees on others.")
        print("          Inspect the per-item lines above.")


if __name__ == "__main__":
    main()