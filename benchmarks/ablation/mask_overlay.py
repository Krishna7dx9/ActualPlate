"""
Fetch one SimpleFood45 image through /detect, overlay the mask,
print mask and image dimensions. Used to decide whether SAM2's
mask covers the bagel or the plate.
"""

import base64
import io
import os
from pathlib import Path

import numpy as np
import requests
from PIL import Image

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def decode_npz(b64: str) -> np.ndarray:
    return np.load(io.BytesIO(base64.b64decode(b64)))["arr"]


def main():
    endpoint = os.environ.get("VISION_INFERENCE_ENDPOINT", "").strip()
    if not endpoint:
        print("VISION_INFERENCE_ENDPOINT not set")
        return
    url = endpoint.rstrip("/") + "/detect"

    jpg = (Path.home() / "Downloads" / "simple_food_45"
           / "ordered_dataset" / "20230927_102352.jpg")
    print(f"  image: {jpg}")

    with open(jpg, "rb") as f:
        r = requests.post(
            url,
            files={"file": (jpg.name, f, "image/jpeg")},
            data={"labels": "bagel", "include_arrays": "true"},
            timeout=300,
        )
    r.raise_for_status()
    body = r.json()

    depth = decode_npz(body["depth_map_b64"]).astype(np.float32)
    mask = decode_npz(body["foods"][0]["mask_b64"]).astype(bool)
    f_px = float(body["focal_length_px"])
    print(f"  depth shape: {depth.shape}")
    print(f"  focal_length_px: {f_px}")

    h, w = mask.shape
    ys, xs = np.where(mask)
    if len(xs) == 0:
        print("  mask is empty")
        return

    bbox_w = int(xs.max() - xs.min() + 1)
    bbox_h = int(ys.max() - ys.min() + 1)
    print(f"  mask pixels: {int(mask.sum())}")
    print(f"  mask bbox:   {bbox_w} x {bbox_h}  (W x H in mask coords)")
    print(f"  image shape: {w} x {h}")

    # Approximate bagel physical size if the mask covers the bagel.
    z_med = float(np.median(depth[mask]))
    print(f"  depth median inside mask: {z_med:.4f} m")
    print(f"  f_px  = {f_px:.1f}")
    print(f"  implied physical width at that depth if mask = bagel:")
    print(f"     {f_px * 0.10 / bbox_w:.4f}  (assuming bagel 0.10 m)")
    print(f"     {f_px * 0.22 / bbox_w:.4f}  (assuming plate 0.22 m)")

    # Overlay.
    img = Image.open(jpg).convert("RGB").resize((w, h))
    arr = np.array(img).copy()
    arr[mask] = (arr[mask] * 0.4 + np.array([255, 0, 0]) * 0.6).astype(np.uint8)
    out = Path.home() / "Downloads" / "bagel_mask_overlay.png"
    Image.fromarray(arr).save(out)
    print(f"  overlay saved: {out}")


if __name__ == "__main__":
    main()