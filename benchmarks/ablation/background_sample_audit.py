"""
Analyze the depth distribution of the RANSAC sample pixels.

The volume formula fits a plane to background pixels (outside the
dilated mask). This script fetches the depth map and mask from the
server, reconstructs the background sample locally with the same
dilation the formula uses, and prints its depth statistics.

Decides: is the plane fit local-table (spread < 5 cm) or dominated
by distant background (spread > 50 cm)?
"""

import base64
import io
import os
import time
from pathlib import Path

import numpy as np
import requests
from scipy.ndimage import binary_dilation

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


JPG = (Path.home() / "Downloads" / "simple_food_45"
       / "ordered_dataset" / "20230927_102352.jpg")


def decode_npz(b64):
    return np.load(io.BytesIO(base64.b64decode(b64)))["arr"]


def post_with_retry(url, files, data, timeout_s, attempts=6):
    last = None
    for i in range(attempts):
        try:
            r = requests.post(url, files=files, data=data, timeout=timeout_s)
            if r.status_code < 500:
                return r
            last = requests.HTTPError(str(r.status_code))
        except (requests.ConnectionError, requests.Timeout) as exc:
            last = exc
        wait = min(2 ** i, 30)
        print(f"    retry {i+1}/{attempts} after {wait}s: {last}", flush=True)
        time.sleep(wait)
    raise last


def main():
    endpoint = os.environ["VISION_INFERENCE_ENDPOINT"].rstrip("/") + "/detect"

    print("[1/4] POST with include_arrays=true, debug=true")
    with open(JPG, "rb") as f:
        r = post_with_retry(
            endpoint,
            {"file": (JPG.name, f, "image/jpeg")},
            {"labels": "bagel", "include_arrays": "true", "debug": "true"},
            300, 6,
        )
    r.raise_for_status()
    body = r.json()

    print("[2/4] decode arrays")
    depth = decode_npz(body["depth_map_b64"]).astype(np.float32)
    mask = decode_npz(body["foods"][0]["mask_b64"]).astype(bool)
    print(f"  depth shape {depth.shape}  mask px {int(mask.sum())}")

    # Match the formula's sample: non-mask pixels, 8 iterations of
    # dilation on the mask. At full res that is 8 px; the arrays are
    # downsampled by depth.shape[1]/image_width relative to full res.
    scale = depth.shape[1] / float(body["image_width"])
    dilate = max(1, int(round(8 * scale)))
    print(f"[3/4] dilate {dilate} px (was 8 px at full res)")

    dilated = binary_dilation(mask, iterations=dilate)
    background = ~dilated & np.isfinite(depth) & (depth > 0)
    bg_depth = depth[background]

    print(f"[4/4] background sample analysis")
    print(f"  sample pixel count: {int(bg_depth.size)}")
    if bg_depth.size == 0:
        return

    print()
    print("=" * 72)
    print("RANSAC SAMPLE DEPTH DISTRIBUTION")
    print("=" * 72)
    for q in (1, 5, 10, 25, 50, 75, 90, 95, 99):
        print(f"  p{q:02d}: {np.percentile(bg_depth, q):.4f} m")
    print(f"  min: {bg_depth.min():.4f} m")
    print(f"  max: {bg_depth.max():.4f} m")
    print(f"  p10-p90 spread: "
          f"{np.percentile(bg_depth, 90) - np.percentile(bg_depth, 10):.4f} m")

    # Distribution of sample pixel counts by 20-cm depth bands.
    print()
    print("=" * 72)
    print("DEPTH BANDS  (how many sample pixels in each 20-cm band)")
    print("=" * 72)
    lo = float(np.floor(bg_depth.min() * 5) / 5)
    hi = float(np.ceil(bg_depth.max() * 5) / 5)
    bins = np.arange(lo, hi + 0.2, 0.2)
    hist, edges = np.histogram(bg_depth, bins=bins)
    total = hist.sum()
    for i, h in enumerate(hist):
        if h > 0:
            print(f"  {edges[i]:.2f} - {edges[i+1]:.2f} m: "
                  f"{int(h):>7} ({100*h/total:.1f}%)")

    print()
    print("=" * 72)
    print("VERDICT")
    print("=" * 72)
    spread = (float(np.percentile(bg_depth, 90))
              - float(np.percentile(bg_depth, 10)))
    print(f"  background sample p10-p90 spread: {spread:.4f} m")
    if spread < 0.05:
        print("  [TIGHT] Sample is local table. The plane fit is")
        print("          using the right pixels. The mask's own depth")
        print("          variation is then Depth Pro's output.")
    elif spread > 0.50:
        print("  [WIDE] Sample includes distant background. The plane")
        print("         fit is compromised. Fix: restrict the sample to")
        print("         pixels within ~100 px of the mask boundary.")
    else:
        print("  [MODERATE] Spread between 5 and 50 cm. Sample includes")
        print("             nearby-but-not-local pixels. Plane is roughly")
        print("             the table but is pulled by the far pixels.")


if __name__ == "__main__":
    main()