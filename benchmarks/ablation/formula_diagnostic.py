"""
Formula stability diagnostic.

The synthetic test (tests/test_volume_synthetic.py) proves the volume
formula is correct on perfect inputs. It does NOT test the formula on
real Depth Pro output, which has noise.

This benchmark feeds real Depth Pro output into estimate_volume_cm3
and compares the result to:

  - the same formula on GT depth
  - the dataset's ground-truth volume

For each item it runs four combinations:

  A: pred_depth + GT_mask + pred_focal
  B: pred_depth + GT_mask + true_focal
  C: GT_depth   + GT_mask + true_focal
  D: GT_depth   + GT_mask + pred_focal

and prints the volume, ratio to GT, and every intermediate value from
inside the formula (plane fit, food depth stats, ray distances, height).

The combination that produces a huge volume while its counterpart
produces a sane volume is the source of the error.

Runs on the laptop against a live Colab tunnel (predictor=remote) or
directly on Colab (predictor=depth_pro).

    python -m benchmarks.ablation.formula_diagnostic \
        --dataset metafood3d --predictor remote --limit 20
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from benchmarks.ablation.depth_quality import PREDICTORS
from vision_service.inference.volume import estimate_volume_cm3


RESULTS_DIR = Path(__file__).resolve().parent / "results"


COMBOS = [
    "A_pred_depth_gt_mask_pred_focal",
    "B_pred_depth_gt_mask_true_focal",
    "C_gt_depth_gt_mask_true_focal",
    "D_gt_depth_gt_mask_pred_focal",
]


def _safe_call(depth, mask, focal):
    """Call estimate_volume_cm3 with debug; return (result, error)."""
    try:
        res = estimate_volume_cm3(depth, mask, focal, return_debug=True)
        return res, None
    except Exception as exc:
        return None, str(exc)


def run(dataset: str, predictor_name: str, limit: int | None):
    from benchmarks.ablation.metafood3d import MetaFood3DAdapter

    adapter = MetaFood3DAdapter()
    predictor = PREDICTORS[predictor_name]
    items = adapter.load_items(limit=limit)

    rows = []

    for i, item in enumerate(items, 1):
        print(f"[{i}/{len(items)}] {item.id}", flush=True)

        if item.gt_depth_map is None or item.gt_mask is None:
            print("    skipped: no GT depth or mask")
            continue

        try:
            pred_depth, pred_focal = predictor(item)
        except Exception as exc:
            print(f"    predictor failed: {exc}")
            continue

        gt_depth = item.gt_depth_map
        gt_mask = item.gt_mask
        true_focal = float(item.focal_length_px)
        gt_volume = float(item.gt_volume_cm3)

        combos_input = {
            COMBOS[0]: (pred_depth, gt_mask, pred_focal),
            COMBOS[1]: (pred_depth, gt_mask, true_focal),
            COMBOS[2]: (gt_depth, gt_mask, true_focal),
            COMBOS[3]: (gt_depth, gt_mask, pred_focal),
        }

        row = {
            "item_id": item.id,
            "gt_volume_cm3": gt_volume,
            "pred_focal": float(pred_focal),
            "true_focal": true_focal,
        }

        for name in COMBOS:
            depth, mask, focal = combos_input[name]
            res, err = _safe_call(depth, mask, focal)

            if res is None:
                row[f"{name}__volume_cm3"] = ""
                row[f"{name}__ratio"] = ""
                row[f"{name}__support_inliers"] = ""
                row[f"{name}__support_ratio"] = ""
                row[f"{name}__food_z_median"] = ""
                row[f"{name}__surface_ray_distance_median"] = ""
                row[f"{name}__plane_ray_distance_median"] = ""
                row[f"{name}__height_median"] = ""
                row[f"{name}__valid_volume_pixels"] = ""
                row[f"{name}__error"] = err
                continue

            dbg = res.get("_debug", {})
            row[f"{name}__volume_cm3"] = res["volume_cm3"]
            row[f"{name}__ratio"] = res["volume_cm3"] / gt_volume if gt_volume > 0 else ""
            row[f"{name}__support_inliers"] = res["support_inliers"]
            row[f"{name}__support_ratio"] = res["support_ratio"]
            row[f"{name}__food_z_median"] = dbg.get("food_z_median", "")
            row[f"{name}__surface_ray_distance_median"] = dbg.get("surface_ray_distance_median", "")
            row[f"{name}__plane_ray_distance_median"] = dbg.get("plane_ray_distance_median", "")
            row[f"{name}__height_median"] = dbg.get("height_median", "")
            row[f"{name}__valid_volume_pixels"] = res["valid_volume_pixels"]
            row[f"{name}__error"] = ""

        rows.append(row)

        # Print a compact line for the four ratios
        ratios = []
        for name in COMBOS:
            v = row.get(f"{name}__ratio")
            if v == "":
                ratios.append("ERR")
            else:
                ratios.append(f"{v:.2f}")
        print(f"    ratios A/B/C/D: {ratios[0]} / {ratios[1]} / {ratios[2]} / {ratios[3]}")

    return rows


def write_csv(rows, path: Path):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def print_summary(rows):
    print()
    print("=" * 70)
    print("FORMULA DIAGNOSTIC SUMMARY")
    print("=" * 70)
    print(f"Items: {len(rows)}")

    for name in COMBOS:
        ratios = [r[f"{name}__ratio"] for r in rows if r[f"{name}__ratio"] != ""]
        if ratios:
            med = float(np.median(ratios))
            print(f"  {name:<45} median ratio = {med:.4f}")
        else:
            print(f"  {name:<45} all failed")

    # Also print median of debug fields for the first combo that succeeded
    for name in COMBOS:
        vals = [r[f"{name}__height_median"] for r in rows if r[f"{name}__height_median"] != ""]
        if vals:
            print(f"  {name} height_median (m): {float(np.median(vals)):.6f}")
            break

    print("=" * 70)


def main(argv=None):
    p = argparse.ArgumentParser(prog="formula_diagnostic")
    p.add_argument("--dataset", required=True, choices=["metafood3d"])
    p.add_argument("--predictor", default="remote", choices=list(PREDICTORS))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--output", default=None)
    args = p.parse_args(argv)

    print("=" * 70)
    print("FORMULA DIAGNOSTIC")
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
        out = RESULTS_DIR / f"formula_diagnostic_{args.dataset}_{ts}.csv"

    write_csv(rows, out)
    print(f"\nCSV written to: {out}")
    print_summary(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())