import csv
import os
import time
from pathlib import Path
import requests
import numpy as np
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

url = os.environ["VISION_INFERENCE_ENDPOINT"].rstrip("/") + "/detect"
IMG = Path.home() / "Downloads" / "simple_food_45" / "ordered_dataset"
OUT = Path("benchmarks/ablation/results/filter_check_multiclass.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)


import csv as _csv
_LABELS = {}
with open(Path.home() / "Downloads" / "simple_food_45" / "ordered_dataset" / "labels.csv", "r", encoding="utf-8") as _f:
    for _row in _csv.DictReader(_f):
        _LABELS[_row["image"].strip()] = _row["label"].strip().lower()
files = sorted(IMG.glob("*.jpg"))

print(f"URL:   {url}")
print(f"files: {len(files)} (first 120)")
print(f"CSV:   {OUT}")
print()
print(f"{'#':>3}  {'file':<26} {'t':>5}  {'hcap_mm':>8}  {'vol':>8}  {'gate':<5}")
print("-" * 80)

rows = []

for i, jpg in enumerate(files, 1):
    t0 = time.time()
    row = {
        "idx": i,
        "file": jpg.name,
        "label": _LABELS.get(jpg.name, "food"),
        "t": None,
        "hcap_mm": None,
        "vol_cm3": None,
        "gate": "",
        "error": "",
    }

    try:
        r = requests.post(
            url,
            files={"file": (jpg.name, jpg.read_bytes(), "image/jpeg")},
            data={"labels": _LABELS.get(jpg.name, "food"), "debug": "true"},
            timeout=180,
        )
    except Exception as exc:
        row["t"] = round(time.time() - t0, 1)
        row["error"] = f"ERR {str(exc)[:30]}"
        rows.append(row)
        print(f"{i:>3}  {jpg.name:<26} {row['t']:>5.1f}  "
              f"{'-':>8}  {'-':>8}  ERR  {str(exc)[:20]}", flush=True)
        continue

    row["t"] = round(time.time() - t0, 1)

    if r.status_code != 200:
        row["error"] = f"HTTP{r.status_code}"
        rows.append(row)
        print(f"{i:>3}  {jpg.name:<26} {row['t']:>5.1f}  "
              f"{'-':>8}  {'-':>8}  HTTP{r.status_code}", flush=True)
        continue

    b = r.json()
    foods = b.get("foods") or []
    if not foods:
        row["error"] = "no-det"
        rows.append(row)
        print(f"{i:>3}  {jpg.name:<26} {row['t']:>5.1f}  "
              f"{'-':>8}  {'-':>8}  no-det", flush=True)
        continue

    f = foods[0]
    v = f.get("volume_cm3")
    d = f.get("_debug") or {}
    hcap = d.get("height_cap_m")

    if v is None:
        row["error"] = (f.get("volume_error") or "unrel")[:30]
        rows.append(row)
        print(f"{i:>3}  {jpg.name:<26} {row['t']:>5.1f}  "
              f"{'-':>8}  {'-':>8}  unrel  {row['error'][:20]}", flush=True)
        continue

    row["vol_cm3"] = round(float(v), 1)
    if hcap is not None:
        row["hcap_mm"] = round(hcap * 1000.0, 2)

    gate_ok = (
        row["hcap_mm"] is not None
        and 20.0 <= row["hcap_mm"] <= 55.0
        and 100.0 <= row["vol_cm3"] <= 500.0
    )
    row["gate"] = "PASS" if gate_ok else "drop"
    rows.append(row)

    h_str = f"{row['hcap_mm']:>8.2f}" if row["hcap_mm"] is not None else f"{'-':>8}"
    print(f"{i:>3}  {jpg.name:<26} {row['t']:>5.1f}  "
          f"{h_str}  {row['vol_cm3']:>8.1f}  {row['gate']}", flush=True)

# Write CSV
with open(OUT, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=["idx","file","label","t","hcap_mm","vol_cm3","gate","error"])
    w.writeheader()
    for r in rows:
        w.writerow(r)

raw = [r["vol_cm3"] for r in rows if r["vol_cm3"] is not None]
gated = [r["vol_cm3"] for r in rows if r["gate"] == "PASS"]

print()
print("=" * 80)
print(f"raw   n={len(raw)}")
if raw:
    print(f"      median={np.median(raw):.1f}  p10={np.percentile(raw,10):.1f}  p90={np.percentile(raw,90):.1f}")
if gated:
    print(f"gated n={len(gated)}   median={np.median(gated):.1f}  p10={np.percentile(gated,10):.1f}  p90={np.percentile(gated,90):.1f}")
    print(f"survival = {100*len(gated)/max(1,len(raw)):.1f}%")
else:
    print("gated n=0 ? filter rejects everything")
print(f"CSV saved: {OUT}")
print("=" * 80)
