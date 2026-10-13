"""
Depth quality measurement — a permanent benchmark.

The volume ablation harness (run.py) answers: "which input to volume
is wrong?"  It does not answer: "in what way is the depth wrong?"

This module answers the second question. It runs a depth predictor on
each item, compares the output to the dataset's ground-truth depth, and
reports metrics that distinguish scale error from shape error.

Metrics:

  bg_spread_pred_frac   (p90 - p10) / median on the outer border.
                        Fractional noise on a surface that should be flat.
                        Independent of absolute scale.
  bg_spread_gt_frac     Same, on the ground-truth depth.
  flatness_ratio        pred / gt. > 5 means the predictor invents depth
                        variation on a flat surface.
  shape_cv              std(pred/gt) / median(pred/gt) over all pixels.
                        Dimensionless. Measures whether pred is a constant
                        multiple of gt (scale only) or varies (shape error).
  food_height_ratio     (plate - food)_pred / (plate - food)_gt, both as
                        fractions of plate distance. 1.0 means the
                        predictor got food height right relative to plate.

Verdict per item:
  "good"   shape_cv < 0.10, flatness_ratio < 5, food_height in [0.7, 1.4]
  "scale"  shape_cv < 0.10 but food_height outside range
  "shape"  shape_cv >= 0.20 or flatness_ratio >= 5
  "unknown" insufficient data

Runs on Colab (Depth Pro needs a GPU).

    python -m benchmarks.ablation.depth_quality \
        --dataset metafood3d --predictor depth_pro --limit 20
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np


RESULTS_DIR = Path(__file__).resolve().parent / "results"


@dataclass
class DepthQualityRow:
    item_id: str
    category: str
    bg_spread_pred_frac: float
    bg_spread_gt_frac: float
    flatness_ratio: float
    shape_cv: float
    food_height_ratio: float
    pred_median: float
    gt_median: float
    verdict: str


DepthPredictor = Callable  # (item) -> (depth_map, focal_length_px)


# ----------------------------------------------------------------------
# Predictors
# ----------------------------------------------------------------------


def depth_pro_predictor(item):
    """
    Run Apple Depth Pro on the item's RGB render.

    Returns (depth_map, focal_length_px).

    Requires the checkpoint at /content/ml-depth-pro/checkpoints/depth_pro.pt
    and chdirs into /content/ml-depth-pro so Depth Pro's hardcoded
    relative checkpoint path resolves.
    """
    import torch
    import depth_pro

    repo_dir = Path("/content/ml-depth-pro")
    if not repo_dir.is_dir():
        raise RuntimeError(
            "ml-depth-pro not found at /content/ml-depth-pro. "
            "Clone it and download the checkpoint first."
        )
    if str(repo_dir / "src") not in sys.path:
        sys.path.insert(0, str(repo_dir / "src"))

    original_cwd = os.getcwd()
    os.chdir(repo_dir)
    try:
        model, transform = depth_pro.create_model_and_transforms()
        model.eval().to("cuda")
    finally:
        os.chdir(original_cwd)

    image, _, f_px = depth_pro.load_rgb(str(item.rgb_path))

    if f_px is None:
        # PNG has no EXIF. Fall back to a documented default FOV.
        h, w = np.asarray(image).shape[:2]
        f_px = 0.5 * w / np.tan(np.deg2rad(50.0 / 2))

    with torch.no_grad():
        pred = model.infer(transform(image).to("cuda"), f_px=f_px)
    depth_map = pred["depth"].detach().cpu().numpy().squeeze().astype(np.float32)

    return depth_map, float(f_px)


PREDICTORS = {"depth_pro": depth_pro_predictor}


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------


def _border_mask(shape, border=150):
    h, w = shape
    m = np.zeros((h, w), dtype=bool)
    m[:border, :] = True
    m[-border:, :] = True
    m[:, :border] = True
    m[:, -border:] = True
    return m


def _spread_frac(values):
    vals = values[np.isfinite(values)]
    if vals.size == 0:
        return float("nan")
    med = float(np.median(vals))
    if med <= 0:
        return float("nan")
    p10 = float(np.percentile(vals, 10))
    p90 = float(np.percentile(vals, 90))
    return (p90 - p10) / med


def measure_item(item, predictor):
    if item.gt_depth_map is None:
        raise ValueError("item has no gt_depth_map")

    pred, _ = predictor(item)
    gt = item.gt_depth_map

    if pred.shape != gt.shape:
        import cv2
        pred = cv2.resize(
            pred, (gt.shape[1], gt.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    valid = np.isfinite(pred) & np.isfinite(gt) & (pred > 0) & (gt > 0)
    bg = _border_mask(pred.shape, border=150) & valid

    if bg.any():
        pred_median = float(np.median(pred[bg]))
        gt_median = float(np.median(gt[bg]))
        pred_spread = _spread_frac(pred[bg])
        gt_spread = _spread_frac(gt[bg])
        if gt_spread > 0:
            flatness_ratio = pred_spread / gt_spread
        else:
            flatness_ratio = float("nan")
    else:
        pred_median = gt_median = pred_spread = gt_spread = flatness_ratio = float("nan")

    if valid.any():
        ratio = pred[valid] / gt[valid]
        med = float(np.median(ratio))
        shape_cv = float(np.std(ratio) / med) if med > 0 else float("nan")
    else:
        shape_cv = float("nan")

    food_height_ratio = float("nan")
    if item.gt_mask is not None:
        food = item.gt_mask.astype(bool) & valid
        if food.any() and bg.any():
            pred_food_med = float(np.median(pred[food]))
            gt_food_med = float(np.median(gt[food]))
            pred_h = (pred_median - pred_food_med) / pred_median
            gt_h = (gt_median - gt_food_med) / gt_median
            if abs(gt_h) > 1e-6:
                food_height_ratio = pred_h / gt_h

    verdict = _classify(shape_cv, flatness_ratio, food_height_ratio)

    return DepthQualityRow(
        item_id=item.id,
        category=item.id.split("/", 1)[0] if "/" in item.id else "",
        bg_spread_pred_frac=pred_spread,
        bg_spread_gt_frac=gt_spread,
        flatness_ratio=flatness_ratio,
        shape_cv=shape_cv,
        food_height_ratio=food_height_ratio,
        pred_median=pred_median,
        gt_median=gt_median,
        verdict=verdict,
    )


def _classify(shape_cv, flatness_ratio, food_h):
    if not np.isfinite(shape_cv):
        return "unknown"
    if shape_cv >= 0.20:
        return "shape"
    if np.isfinite(flatness_ratio) and flatness_ratio >= 5.0:
        return "shape"
    if np.isfinite(food_h) and not (0.7 <= food_h <= 1.4):
        return "scale"
    return "good"


# ----------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------


def get_adapter(name):
    if name == "metafood3d":
        from benchmarks.ablation.metafood3d import MetaFood3DAdapter
        return MetaFood3DAdapter()
    raise ValueError(f"unknown dataset: {name}")


def run(dataset, predictor_name, limit):
    adapter = get_adapter(dataset)
    predictor = PREDICTORS[predictor_name]
    items = adapter.load_items(limit=limit)

    rows = []
    for i, item in enumerate(items, 1):
        print(f"[{i}/{len(items)}] {item.id}", flush=True)
        try:
            row = measure_item(item, predictor)
        except Exception as exc:
            print(f"    FAILED: {exc}")
            continue
        rows.append(row)
        print(
            f"    flatness={row.flatness_ratio:.2f} "
            f"cv={row.shape_cv:.3f} "
            f"food_h={row.food_height_ratio:.2f} "
            f"-> {row.verdict}"
        )
    return rows


# ----------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------


CSV_COLUMNS = [
    "item_id", "category",
    "bg_spread_pred_frac", "bg_spread_gt_frac", "flatness_ratio",
    "shape_cv", "food_height_ratio",
    "pred_median", "gt_median", "verdict",
]


def _fmt(v):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return ""
    return f"{v:.6f}" if isinstance(v, float) else str(v)


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLUMNS)
        for r in rows:
            w.writerow([_fmt(getattr(r, c)) for c in CSV_COLUMNS])


def print_summary(rows):
    print()
    print("=" * 70)
    print("DEPTH QUALITY SUMMARY")
    print("=" * 70)
    print(f"Items: {len(rows)}")

    verdicts = [r.verdict for r in rows]
    for v in ("good", "scale", "shape", "unknown"):
        n = verdicts.count(v)
        if n:
            print(f"  {v:<8} {n}")

    def med(attr):
        vals = [getattr(r, attr) for r in rows
                if np.isfinite(getattr(r, attr))]
        return float(np.median(vals)) if vals else float("nan")

    print()
    print(f"  median flatness_ratio:     {med('flatness_ratio'):.4f}")
    print(f"  median shape_cv:           {med('shape_cv'):.4f}")
    print(f"  median food_height_ratio:  {med('food_height_ratio'):.4f}")
    print()
    print("Interpretation:")
    print("  flatness near 1, cv near 0  -> depth is correct")
    print("  flatness > 5, cv > 0.2      -> shape error. Replace model.")
    print("  flatness near 1, scale off  -> scale error. Rescale output.")
    print("=" * 70)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def main(argv=None):
    p = argparse.ArgumentParser(prog="depth_quality")
    p.add_argument("--dataset", required=True, choices=["metafood3d"])
    p.add_argument("--predictor", default="depth_pro", choices=list(PREDICTORS))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--output", default=None)
    args = p.parse_args(argv)

    print("=" * 70)
    print("DEPTH QUALITY BENCHMARK")
    print("=" * 70)
    print(f"  dataset:   {args.dataset}")
    print(f"  predictor: {args.predictor}")
    print(f"  limit:     {args.limit}")
    print("=" * 70)
    print()

    rows = run(args.dataset, args.predictor, args.limit)
    if not rows:
        print("No rows. Nothing to write.")
        return 1

    if args.output:
        out = Path(args.output)
    else:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = RESULTS_DIR / f"depth_quality_{args.dataset}_{ts}.csv"

    write_csv(rows, out)
    print(f"\nCSV written to: {out}")

    print_summary(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())