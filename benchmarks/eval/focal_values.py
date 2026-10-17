from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS

IMG_DIR = Path.home() / "Downloads" / "simple_food_45" / "ordered_dataset"
photos = sorted(IMG_DIR.glob("*.jpg"))[:50]

focal_mm = []
focal_35mm = []
focal_35mm_unique = defaultdict(int)
sizes = []

for p in photos:
    try:
        img = Image.open(p)
        w, h = img.size
        sizes.append((w, h))
        exif = img._getexif() or {}
        data = {}
        for tag_id, value in exif.items():
            tag = TAGS.get(tag_id, tag_id)
            data[tag] = value
        if "FocalLength" in data:
            try:
                f = data["FocalLength"]
                mm = float(f) if isinstance(f, (int, float)) else float(f.numerator) / float(f.denominator) if hasattr(f, "numerator") else None
                if mm:
                    focal_mm.append(mm)
            except Exception:
                pass
        if "FocalLengthIn35mmFilm" in data:
            v = data["FocalLengthIn35mmFilm"]
            if isinstance(v, (int, float)) and v > 0:
                focal_35mm.append(v)
                focal_35mm_unique[v] += 1
    except Exception:
        pass

print(f"Photos sampled: {len(sizes)}")
if sizes:
    w, h = sizes[0]
    diag = (w**2 + h**2) ** 0.5
    print(f"Image size: {w} x {h}, diagonal = {diag:.0f} px")
    print()

if focal_mm:
    print(f"Actual FocalLength (mm):")
    print(f"  count={len(focal_mm)}, min={min(focal_mm):.2f}, max={max(focal_mm):.2f}, median={np.median(focal_mm):.2f}")
    print()

if focal_35mm:
    print(f"FocalLengthIn35mmFilm (mm-equivalent):")
    print(f"  count={len(focal_35mm)}, min={min(focal_35mm):.0f}, max={max(focal_35mm):.0f}, median={np.median(focal_35mm):.0f}")
    print(f"  Unique values: {dict(focal_35mm_unique)}")
    print()

# Compute implied f_px from FocalLengthIn35mmFilm
if focal_35mm and sizes:
    # f_px = focal_35mm_equiv / 36mm (full-frame width) * image_width_in_px
    # more standard: f_px = focal_35mm_equiv / 43.27 * diagonal_px
    diag = (w**2 + h**2) ** 0.5
    f_px_implied = [(f / 43.27) * diag for f in focal_35mm]
    print(f"Implied f_px from FocalLengthIn35mmFilm:")
    print(f"  median = {np.median(f_px_implied):.0f} px")
    print(f"  range  = [{min(f_px_implied):.0f}, {max(f_px_implied):.0f}] px")
    print()
    print(f"Depth Pro reported f_px = 2440 px")
    ratio = np.median(f_px_implied) / 2440
    print(f"Ratio (true / DepthPro) = {ratio:.3f}")
    print()
    if ratio > 1.15:
        print(f"*** Focal is UNDERESTIMATED by {100*(ratio-1):.0f}%.")
        print(f"*** Volume scales ~linearly with focal.")
        print(f"*** Correcting focal could raise volume by ~{100*(ratio-1):.0f}%.")
        print(f"*** This is enough to close a {100*(ratio-1):.0f}% under-estimate.")
        print(f"*** The pipeline may be reading focal wrong. Check the code path.")
    elif ratio < 0.85:
        print(f"*** Focal is OVERESTIMATED by {100*(1-ratio):.0f}%.***")
    else:
        print(f"*** Focal is approximately correct (within 15%). ***")
        print(f"*** The 40-67% error is NOT caused by focal. ***")
