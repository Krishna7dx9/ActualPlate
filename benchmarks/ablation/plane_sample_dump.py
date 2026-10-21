"""
Dump the RANSAC sample: pixel count, depth stats, distance from mask.
Prints every stage so silence is unambiguous.
"""
import base64, io, os, time
from pathlib import Path
import numpy as np
import requests
from scipy.ndimage import binary_dilation

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

JPG = Path.home() / "Downloads" / "simple_food_45" / "ordered_dataset" / "20230927_102352.jpg"


def decode_npz(b):
    return np.load(io.BytesIO(base64.b64decode(b)))["arr"]


def post(url, files, data, attempts=6):
    last = None
    for i in range(attempts):
        try:
            r = requests.post(url, files=files, data=data, timeout=300)
            if r.status_code < 500:
                return r
            last = requests.HTTPError(str(r.status_code))
        except (requests.ConnectionError, requests.Timeout) as e:
            last = e
        wait = min(2 ** i, 30)
        print(f"    retry {i+1}/{attempts} after {wait}s: {last}", flush=True)
        time.sleep(wait)
    raise last


def main():
    print("[1/6] reading environment", flush=True)
    endpoint = os.environ.get("VISION_INFERENCE_ENDPOINT", "").strip()
    if not endpoint:
        print("  VISION_INFERENCE_ENDPOINT not set"); return
    url = endpoint.rstrip("/") + "/detect"
    print(f"  url: {url}", flush=True)

    print("[2/6] reading image", flush=True)
    if not JPG.exists():
        print(f"  image missing: {JPG}"); return
    image_bytes = JPG.read_bytes()
    print(f"  bytes: {len(image_bytes)}", flush=True)

    print("[3/6] POSTing to /detect (waiting for response)", flush=True)
    t0 = time.time()
    with open(JPG, "rb") as f:
        r = post(url, {"file": (JPG.name, f, "image/jpeg")},
                 {"labels": "bagel", "include_arrays": "true"}, 6)
    print(f"  status {r.status_code} in {time.time()-t0:.1f}s", flush=True)
    r.raise_for_status()
    body = r.json()
    print(f"  response keys: {list(body.keys())}", flush=True)

    print("[4/6] decoding arrays", flush=True)
    depth = decode_npz(body["depth_map_b64"]).astype(np.float32)
    mask = decode_npz(body["foods"][0]["mask_b64"]).astype(bool)
    img_w = body["image_width"]
    focal = body["focal_length_px"]
    print(f"  depth {depth.shape} mask px {int(mask.sum())}", flush=True)

    print("[5/6] computing scales", flush=True)
    scale = depth.shape[1] / float(img_w)
    dil_8 = max(1, int(round(8 * scale)))
    dil_60 = max(1, int(round(60 * scale)))
    plate_depth = float(np.median(depth[mask]))
    px_per_m = focal * scale / plate_depth

    print(f"  image width:      {img_w}")
    print(f"  scale:            {scale:.4f}")
    print(f"  focal orig px:    {focal:.1f}")
    print(f"  focal array px:   {focal * scale:.1f}")
    print(f"  plate depth:      {plate_depth:.4f} m")
    print(f"  px per meter:     {px_per_m:.0f}")
    print(f"  dil_8  = {dil_8} px  = {dil_8 / px_per_m * 1000:.1f} mm", flush=True)
    print(f"  dil_60 = {dil_60} px = {dil_60 / px_per_m * 1000:.1f} mm", flush=True)

    print("[6/6] sampling", flush=True)
    exp8 = binary_dilation(mask, iterations=dil_8)
    ring60 = binary_dilation(mask, iterations=dil_60) & ~exp8
    whole = ~exp8

    for name, m in (("ring 8-60 ", ring60), ("all non-mask", whole)):
        d = depth[m]
        if d.size == 0:
            print(f"  {name}: empty"); continue
        p10 = float(np.percentile(d, 10))
        p90 = float(np.percentile(d, 90))
        print(f"  {name}: n={int(d.size)} "
              f"min={d.min():.3f} p10={p10:.3f} "
              f"med={np.median(d):.3f} p90={p90:.3f} "
              f"max={d.max():.3f}  spread={p90-p10:.3f} m")


if __name__ == "__main__":
    main()
