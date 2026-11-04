import os, time, requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
url = os.environ["VISION_INFERENCE_ENDPOINT"].rstrip("/")
print("URL:", url)

root = Path.home() / "Downloads" / "mf3d_partial" / "Blender_render_images"
depth = next(root.rglob("*_depth0001.png"))
base = depth.name.replace("_depth0001.png", "")
rgb = depth.parent.parent / "Original" / f"{base}.png"
print("Image:", rgb)

t = time.time()
with open(rgb, "rb") as f:
    r = requests.post(url, files={"file": (rgb.name, f, "image/jpeg")}, timeout=300)

print("status:", r.status_code)
print("elapsed_s:", round(time.time() - t, 1))
print("content_type:", r.headers.get("content-type"))

if r.status_code != 200:
    print(r.text[:1000])
    raise SystemExit(1)

j = r.json()
print("top_keys:", sorted(j.keys()))
print("foods_len:", len(j.get("foods") or []))
print("geometry_len:", len(j.get("geometry") or []))

if j.get("geometry"):
    g = j["geometry"][0]
    print("geom_keys:", sorted(g.keys()))
    print("bbox:", g.get("bbox"))
    print("mask_b64_len:", len(g.get("mask_b64") or ""))
else:
    print("NO GEOMETRY IN RESPONSE")
