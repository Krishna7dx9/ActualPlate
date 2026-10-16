"""
Sweep all photos in simple_food_45 and report pipeline success rate.

Success      = HTTP 200 + volume_cm3 is a number (not None)
Unreliable   = 200 but pipeline raised ValueError
No-detection = DINO found nothing (empty foods list)
Non-200      = any other HTTP status
Network-err  = request raised before response

Usage: python -m benchmarks.ablation.sweep_513
"""
import os
from pathlib import Path
import numpy as np
import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


LABEL = "bagel"
IMG_DIR = Path.home() / "Downloads" / "simple_food_45" / "ordered_dataset"


def main():
    url = os.environ["VISION_INFERENCE_ENDPOINT"].rstrip("/") + "/detect"
    files = sorted(IMG_DIR.glob("*.jpg"))
    print(f"URL:   {url}")
    print(f"label: {LABEL}")
    print(f"files: {len(files)}")
    print()

    success = 0
    unreliable = 0
    no_detection = 0
    non_200 = 0
    network_err = 0

    volumes = []
    h_maxes = []

    for i, jpg in enumerate(files, 1):
        try:
            r = requests.post(
                url,
                files={"file": (jpg.name, jpg.read_bytes(), "image/jpeg")},
                data={"labels": LABEL, "debug": "true"},
                timeout=180,
            )
        except Exception:
            network_err += 1
            continue

        if r.status_code != 200:
            non_200 += 1
            continue

        b = r.json()
        foods = b.get("foods") or []
        if not foods:
            no_detection += 1
            continue

        f = foods[0]
        vol = f.get("volume_cm3")
        if vol is None:
            unreliable += 1
            continue

        success += 1
        volumes.append(float(vol))
        d = f.get("_debug") or {}
        if d.get("height_max") is not None:
            h_maxes.append(float(d["height_max"]) * 1000.0)

        if i % 50 == 0:
            print(f"  ... {i}/{len(files)}  ok={success} unrel={unreliable} "
                  f"nodet={no_detection} non200={non_200} neterr={network_err}")

    n = len(files)
    print()
    print("=" * 70)
    print("SWEEP RESULT")
    print("=" * 70)
    print(f"  files          {n}")
    print(f"  success        {success}  ({100*success/n:.1f}%)")
    print(f"  unreliable     {unreliable}  ({100*unreliable/n:.1f}%)")
    print(f"  no-detection   {no_detection}  ({100*no_detection/n:.1f}%)")
    print(f"  non-200        {non_200}")
    print(f"  network-err    {network_err}")

    if volumes:
        v = np.array(volumes)
        print()
        print(f"  volume p10     {np.percentile(v, 10):.1f} cm^3")
        print(f"  volume median  {np.median(v):.1f} cm^3")
        print(f"  volume p90     {np.percentile(v, 90):.1f} cm^3")
        print(f"  volume min/max {v.min():.1f} / {v.max():.1f} cm^3")

    if h_maxes:
        h = np.array(h_maxes)
        print()
        print(f"  h_max median   {np.median(h):.1f} mm")
        print(f"  h_max p90      {np.percentile(h, 90):.1f} mm")

    print("=" * 70)


if __name__ == "__main__":
    main()
