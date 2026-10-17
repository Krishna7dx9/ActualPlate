"""
Read the pipeline's own plane fit and volume intermediates from
the server.

Calls /detect with debug=true and include_arrays=false. The server
runs estimate_volume_cm3 on the full-resolution depth map and
mask, and returns the _debug sub-dict. No client-side
reconstruction, no focal mismatch, no downsampling.
"""

import io
import json
import os
import time
from pathlib import Path

import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

JPG = Path.home() / "Downloads" / "simple_food_45" / "ordered_dataset" / "20230927_102352.jpg"


def post_with_retry(url, make_files, data, timeout_s, attempts=6):
    last = None
    for i in range(attempts):
        try:
            r = requests.post(url, files=make_files(), data=data, timeout=timeout_s)
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
    print(f"POST {endpoint}")

    image_bytes = JPG.read_bytes()

    def make_files():
        return {"file": (JPG.name, io.BytesIO(image_bytes), "image/jpeg")}

    r = post_with_retry(
        endpoint,
        make_files,
        {"labels": "bagel", "include_arrays": "false", "debug": "true"},
        300, 6,
    )
    r.raise_for_status()
    body = r.json()

    foods = body.get("foods", [])
    if not foods:
        print("no foods")
        return
    food = foods[0]

    print()
    print("=" * 72)
    print("PIPELINE VOLUME (server-side, full resolution)")
    print("=" * 72)
    for k in ("label", "score", "mask_pixels", "volume_cm3",
              "volume_status", "volume_error", "support_inliers", "support_ratio",
              "valid_volume_pixels"):
        print(f"  {k:24s} {food.get(k)}")

    dbg = food.get("_debug")
    if not dbg:
        print("  (_debug missing from response)")
        return

    print()
    print("=" * 72)
    print("PLANE AND GEOMETRY")
    print("=" * 72)
    print(f"  plane_normal             {dbg.get('plane_normal')}")
    print(f"  plane_offset             {dbg.get('plane_offset'):.4f} m")
    print(f"  support_inliers          {dbg.get('support_inliers')}")
    print(f"  support_ratio            {dbg.get('support_ratio'):.4f}")
    for k in sorted(dbg.keys()):
        v = dbg[k]
        if isinstance(v, (int, float)):
            print(f"  {k:<28} {v:.4f}")
        else:
            print(f"  {k:<28} {v}")
    print(f"  valid_volume_pixels      {dbg.get('valid_volume_pixels')}")

    print()
    print("=" * 72)
    print("INTERPRETATION")
    print("=" * 72)
    hm = dbg.get("height_median")
    if hm is not None:
        print(f"  height_median = {hm:.4f} m")
        print("  a bagel is ~3-4 cm tall")
        print("  a plate-to-table is ~1-2 cm")
        if hm > 0.05:
            print("  > 5 cm of 'food' height. The plane is likely")
            print("  fit to the table, not the plate. Plate pixels")
            print("  contribute height and inflate the volume.")
        else:
            print("  heights look consistent with food-sized")
            print("  relief. The plane fit is likely correct.")


if __name__ == "__main__":
    main()
