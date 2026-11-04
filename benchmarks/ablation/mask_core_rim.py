import base64, io, os, time
from pathlib import Path
import numpy as np
import requests
from scipy.ndimage import binary_erosion

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

JPG = Path.home() / "Downloads" / "simple_food_45" / "ordered_dataset" / "20230927_102352.jpg"

def decode_npz(b64):
    return np.load(io.BytesIO(base64.b64decode(b64)))["arr"]

def post_with_retry(url, files, data, timeout_s, attempts=6):
    last = None
    for i in range(attempts):
        try:
            r = requests.post(url, files=files, data=data, timeout=timeout_s)
            if r.status_code < 500:
                return r
            last = requests.HTTPError(f"{r.status_code}")
        except (requests.ConnectionError, requests.Timeout) as exc:
            last = exc
        wait = min(2 ** i, 30)
        print(f"    retry {i+1}/{attempts} after {wait}s: {last}", flush=True)
        time.sleep(wait)
    raise last

def main():
    print("[1/5] env", flush=True)
    endpoint = os.environ.get("VISION_INFERENCE_ENDPOINT", "").strip()
    if not endpoint:
        print("  VISION_INFERENCE_ENDPOINT not set"); return
    url = endpoint.rstrip("/") + "/detect"
    print(f"  url: {url}")

    print("[2/5] reading image", flush=True)
    data = JPG.read_bytes()
    print(f"  bytes: {len(data)}")

    print("[3/5] POST with retry", flush=True)
    t0 = time.time()
    r = post_with_retry(url, {"file": (JPG.name, data, "image/jpeg")},
                        {"labels": "bagel", "include_arrays": "true"}, 300, 6)
    print(f"  status {r.status_code} in {time.time()-t0:.1f}s", flush=True)
    r.raise_for_status()
    body = r.json()

    print("[4/5] decode", flush=True)
    depth = decode_npz(body["depth_map_b64"]).astype(np.float32)
    mask = decode_npz(body["foods"][0]["mask_b64"]).astype(bool)
    print(f"  depth {depth.shape}  mask px {int(mask.sum())}")

    print("[5/5] core/rim", flush=True)
    # 30 pixels at 4000px width == 0.75% of the image width.
    # Scale by actual width to keep the core/rim split meaningful
    # across resolutions.
    h, w = mask.shape
    iters = max(1, int(round(0.0075 * max(h, w))))
    print(f"  erosion iterations at this resolution: {iters}")
    er = binary_erosion(mask, iterations=iters)
    rim = mask & ~er
    for name, m in (("core", er), ("rim ", rim), ("all ", mask)):
        if m.sum() == 0:
            print(f"  {name}: empty"); continue
        d = depth[m]
        print(f"  {name}: n={int(m.sum()):>8}  min={d.min():.3f}  "
              f"p10={np.percentile(d,10):.3f}  med={np.median(d):.3f}  "
              f"p90={np.percentile(d,90):.3f}  max={d.max():.3f} m")

if __name__ == "__main__":
    main()
