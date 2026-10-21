"""
Focal audit on the SimpleFood45 bagel JPGs.

Question: is the pipeline's reported focal_length_px the same
as the focal encoded in the EXIF metadata?

If they match, Interpretation A holds: Depth Pro's depth is
wrong.
If they differ by a scale factor, Interpretation B holds: the
pipeline is using a wrong focal, and Depth Pro may be correct.

Reads EXIF directly with PIL. No tunnel, no Colab, no API.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from PIL.ExifTags import TAGS


JPG_DIR = Path.home() / "Downloads" / "simple_food_45" / "ordered_dataset"
N = 5


def read_exif(path: Path) -> dict:
    out = {}
    with Image.open(path) as im:
        out["ImageWidth"] = im.width
        out["ImageLength"] = im.height
        exif = im.getexif()
        for tag_id, value in exif.items():
            name = TAGS.get(tag_id, tag_id)
            out[name] = value
    return out


def focal_px_from_exif(exif: dict):
    """
    Return (f_px_35mm_equiv, f_px_actual_mm_derived) or (None, None).
    """
    f35 = exif.get("FocalLengthIn35mmFilm")
    f_mm = exif.get("FocalLength")
    w = exif.get("ImageWidth")
    h = exif.get("ImageLength")
    if not (w and h):
        return None, None

    # Diagonal in the 35mm standard: 43.266 mm. Use image diagonal.
    diag = (w ** 2 + h ** 2) ** 0.5

    f_px_35 = None
    if f35:
        f_px_35 = (float(f35) / 43.266) * diag

    f_px_mm = None
    if f_mm:
        # Need sensor width to convert. Assume a typical 1/1.7" sensor
        # width is unknown, so this branch is approximate. We report
        # it but do not trust it for absolute scale.
        f_px_mm = float(f_mm)  # raw value, in mm

    return f_px_35, f_px_mm


def main():
    jpgs = sorted(JPG_DIR.glob("*.jpg"))[:N]
    print("=" * 72)
    print("FOCAL AUDIT — SimpleFood45 bagels")
    print("=" * 72)
    print(f"  folder: {JPG_DIR}")
    print(f"  images: {len(jpgs)}")
    print()

    for i, p in enumerate(jpgs, 1):
        try:
            e = read_exif(p)
        except Exception as exc:
            print(f"[{i}/{N}] {p.name}: EXIF read failed: {exc}")
            continue

        f35 = e.get("FocalLengthIn35mmFilm")
        f_mm = e.get("FocalLength")
        w = e.get("ImageWidth")
        h = e.get("ImageLength")
        make = e.get("Make")
        model = e.get("Model")
        orient = e.get("Orientation")

        f_px_35, f_mm_raw = focal_px_from_exif(e)

        print(f"[{i}/{N}] {p.name}")
        print(f"    make / model:      {make}  {model}")
        print(f"    image size (WxH):  {w} x {h}")
        print(f"    Orientation tag:   {orient}")
        print(f"    FocalLength:       {f_mm}")
        print(f"    FocalLengthIn35:   {f35}")
        if f_px_35:
            print(f"    f_px (35mm-equiv): {f_px_35:.1f} px")
        else:
            print(f"    f_px (35mm-equiv): cannot compute (no 35mm tag)")
        print()

    print("=" * 72)
    print("COMPARE WITH PIPELINE'S REPORTED f_px = 2440.2")
    print("=" * 72)
    print()
    print("Interpretation rules:")
    print("  - if EXIF f_px ~= 2440 -> focal is correct, Depth Pro's")
    print("    depth is 2.6x too large (Interpretation A)")
    print("  - if EXIF f_px ~= 6300 -> pipeline is using a 2.6x too")
    print("    small focal (Interpretation B)")
    print("  - if EXIF f_px is something else, we have a third case")


if __name__ == "__main__":
    main()