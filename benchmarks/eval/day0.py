"""
Day 0: EXIF audit + statistical baseline.

Two outputs:
1. EXIF audit ? which metadata fields survive in SimpleFood45 photos
2. Per-class median error, signed bias, bootstrap 95% CI from existing sweep

Neither requires the GPU server. Both inform the decision after P2.
"""

import csv
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict

import numpy as np

try:
    from PIL import Image
    from PIL.ExifTags import TAGS
    HAVE_PIL = True
except ImportError:
    HAVE_PIL = False


IMG_DIR = Path.home() / "Downloads" / "simple_food_45" / "ordered_dataset"
GT_CSV = IMG_DIR / "labels.csv"
PRED_CSV = Path("benchmarks/ablation/results/filter_check_multiclass.csv")


def audit_exif():
    print("=" * 72)
    print("EXIF AUDIT")
    print("=" * 72)

    if not HAVE_PIL:
        print("PIL not available; install pillow")
        return

    photos = sorted(IMG_DIR.glob("*.jpg"))
    if not photos:
        print(f"No photos in {IMG_DIR}")
        return

    sample = photos[:50]
    field_counts = defaultdict(int)
    focal = focal_equiv = make = 0
    total = 0

    for p in sample:
        try:
            img = Image.open(p)
            exif = img._getexif() or {}
            total += 1
            has_f = has_fe = has_m = False
            for tag_id, _ in exif.items():
                tag = TAGS.get(tag_id, tag_id)
                field_counts[tag] += 1
                if tag == "FocalLength":
                    has_f = True
                elif tag == "FocalLengthIn35mmFilm":
                    has_fe = True
                elif tag == "Make":
                    has_m = True
            focal += has_f
            focal_equiv += has_fe
            make += has_m
        except Exception:
            pass

    print(f"Sampled {total} of {len(photos)} photos")
    print(f"  FocalLength present:            {focal}/{total} = {100*focal/max(1,total):.0f}%")
    print(f"  FocalLengthIn35mmFilm present:  {focal_equiv}/{total} = {100*focal_equiv/max(1,total):.0f}%")
    print(f"  Make/Model present:             {make}/{total} = {100*make/max(1,total):.0f}%")
    print()
    print("Top EXIF fields:")
    for field, count in sorted(field_counts.items(), key=lambda x: -x[1])[:12]:
        print(f"  {field:<32} {count}/{total}")

    if focal == 0 and focal_equiv == 0:
        print()
        print("*** No focal EXIF in sample. ***")
        print("*** Depth Pro used its own internal focal estimate on every photo. ***")
        print("*** Any correction must override focal; reading from EXIF is not possible here. ***")


def bootstrap_ci(values, n_iter=2000, ci=0.95):
    v = np.asarray(values)
    if len(v) < 5:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(42)
    meds = np.empty(n_iter)
    for i in range(n_iter):
        meds[i] = np.median(rng.choice(v, size=len(v), replace=True))
    a = (1 - ci) / 2
    return float(np.quantile(meds, a)), float(np.quantile(meds, 1 - a))


def baseline_stats():
    print()
    print("=" * 72)
    print("STATISTICAL BASELINE (existing sweep CSV)")
    print("=" * 72)

    if not GT_CSV.exists():
        print(f"GT missing: {GT_CSV}")
        return
    if not PRED_CSV.exists():
        print(f"Predictions missing: {PRED_CSV}")
        print("Run the multi-class sweep first, or point PRED_CSV at the right file.")
        return

    gt = {}
    with open(GT_CSV, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            gt[row["image"].strip()] = (row["label"].strip().lower(), float(row["volume"]))

    per_class = defaultdict(list)
    with open(PRED_CSV, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["vol_cm3"] in ("", None):
                continue
            name = row["file"].strip()
            if name not in gt:
                continue
            tl, tv = gt[name]
            pred = float(row["vol_cm3"])
            per_class[tl].append((pred - tv) / tv)

    print(f"{'class':<20} {'n':>4} {'median|e|':>10}  {'95% CI':>22}  {'median bias':>12}")
    print("-" * 74)

    all_rel = []
    for cls in sorted(per_class):
        rels = np.array(per_class[cls])
        all_rel.extend(rels)
        abs_pct = np.abs(rels) * 100
        med = np.median(abs_pct)
        lo, hi = bootstrap_ci(abs_pct)
        bias = np.median(rels) * 100
        print(f"{cls:<20} {len(rels):>4} {med:>9.1f}%  [{lo:>7.1f}, {hi:>7.1f}]%  {bias:>11.1f}%")

    all_rel = np.array(all_rel)
    print("-" * 74)
    pooled_med = np.median(np.abs(all_rel)) * 100
    pooled_lo, pooled_hi = bootstrap_ci(np.abs(all_rel) * 100)
    print(f"{'POOLED':<20} {len(all_rel):>4} {pooled_med:>9.1f}%  [{pooled_lo:>7.1f}, {pooled_hi:>7.1f}]%")

    print()
    print("Notes:")
    print("  - median|err| is per-photo absolute percentage error, not pooled mean MAPE.")
    print("  - 95% CI from 2000 bootstrap resamples of the per-photo errors.")
    print("  - Splits are random here; session-based split needs the photo date prefix.")


def main():
    try:
        git_hash = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True
        ).stdout.strip()
    except Exception:
        git_hash = "unknown"

    print("=" * 72)
    print(f"DAY 0 DIAGNOSTIC ? {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}")
    print(f"Git: {git_hash}")
    print("=" * 72)
    print()

    audit_exif()
    baseline_stats()

    print()
    print("=" * 72)
    print("INTERPRETATION")
    print("=" * 72)
    print("Focal scaling rule: volume_cm3 ? focal_px (linear, not quadratic).")
    print("Depth Pro's own focal guess on these photos was ~2440 px.")
    print("A 26mm-equivalent phone would be ~2758 px (13% higher).")
    print("Even a 13% focal correction changes volume by ~13%.")
    print("If pooled median|err| here is 40?67%, a 13% correction cannot close it.")
    print("The remaining gap is Depth Pro's depth estimate on close-range food ?")
    print("that is what P3 (depth swap) tests, not a focal correction.")


if __name__ == "__main__":
    main()
