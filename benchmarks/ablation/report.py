"""
Reporting for the volume ablation harness.

Two outputs:

1. A CSV file with one row per item, one column per variant
   (volume + ratio). This is the raw evidence.

2. A summary printed to stdout:

     - MAPE per variant across all items.
     - Median ratio per variant across all items.
     - Per-category breakdown.
     - Ranked list of error contributions: which ratio
       (B/D, C/D, etc.) is largest on average.

The summary is the answer. It says exactly which input to the
volume formula contributes most to the error, and by how much.
"""

from __future__ import annotations

import csv
import statistics
from pathlib import Path

from benchmarks.ablation.run import Row
from benchmarks.ablation.variants import VARIANTS


# --- CSV output ------------------------------------------------------


CSV_COLUMNS = (
    ["item_id", "category", "gt_volume_cm3"]
    + [f"vol_{code}" for code in VARIANTS]
    + [f"ratio_{code}" for code in VARIANTS]
)


def write_csv(rows: list[Row], path: Path) -> None:
    """Write one row per item to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_COLUMNS)

        for row in rows:
            csv_row = [row.item_id, row.category, row.gt_volume_cm3]
            csv_row += [_fmt(row.volumes.get(c)) for c in VARIANTS]
            csv_row += [_fmt(row.ratios.get(c)) for c in VARIANTS]
            writer.writerow(csv_row)


def _fmt(v: float | None) -> str:
    if v is None:
        return ""
    return f"{v:.4f}"


# --- Summary ---------------------------------------------------------


def _mape(ratios: list[float]) -> float:
    """Mean absolute percentage error. Ratios are pred/gt."""
    if not ratios:
        return float("nan")
    return statistics.mean(abs(r - 1.0) for r in ratios) * 100.0


def _median_ratio(ratios: list[float]) -> float:
    if not ratios:
        return float("nan")
    return statistics.median(ratios)


def _collect(row: Row, code: str) -> float | None:
    """Return the ratio for one variant, or None if not runnable."""
    return row.ratios.get(code)


def summarize(rows: list[Row]) -> dict:
    """
    Return a summary dict with per-variant metrics and a ranked
    list of error contributions.
    """
    summary: dict = {"n_items": len(rows), "variants": {}}

    for code in VARIANTS:
        ratios = [
            r
            for r in (_collect(row, code) for row in rows)
            if r is not None
        ]
        summary["variants"][code] = {
            "n": len(ratios),
            "mape_pct": round(_mape(ratios), 2) if ratios else None,
            "median_ratio": round(_median_ratio(ratios), 4) if ratios else None,
        }

    # Rank: how much does each input contribute?
    #
    # Reference is D (GT depth + GT mask). The residual of D
    # itself is the formula + plane fit.
    #
    # Contribution of depth = B/D ratio (median across items).
    # Contribution of mask  = C/D ratio.
    # Everything else is small.
    ranked = _rank_contributions(rows)
    summary["ranked_contributions"] = ranked

    return summary


def _rank_contributions(rows: list[Row]) -> list[dict]:
    """
    Produce a ranked list of error contributions.

    Uses median ratios across items. Each entry:

        {
            "source": "depth" or "mask" or "formula_residual",
            "median_ratio": float,
            "excess_over_1": float,   # ratio - 1
        }

    Where "excess_over_1" is > 0 if the source inflates the error.
    """
    result = []

    def _med(code: str) -> float | None:
        vals = [r.ratios.get(code) for r in rows]
        vals = [v for v in vals if v is not None]
        if not vals:
            return None
        return statistics.median(vals)

    depth_median = _med("B")
    mask_median = _med("C")
    residual_median = _med("D")
    total_median = _med("A")

    for name, val in (
        ("depth", depth_median),
        ("mask", mask_median),
        ("formula_residual", residual_median),
        ("total_pipeline", total_median),
    ):
        if val is None:
            continue
        result.append({
            "source": name,
            "median_ratio": round(val, 4),
            "excess_over_1": round(val - 1.0, 4),
        })

    # Sort descending by excess
    result.sort(key=lambda x: x["excess_over_1"], reverse=True)
    return result


# --- Print summary ---------------------------------------------------


def print_summary(summary: dict) -> None:
    print()
    print("=" * 70)
    print("ABLATION SUMMARY")
    print("=" * 70)
    print(f"Items evaluated: {summary['n_items']}")
    print()

    print("Per-variant metrics (ratio = predicted / ground truth):")
    print(f"  {'variant':<10} {'n':>4} {'MAPE %':>10} {'median ratio':>15}")
    for code, m in summary["variants"].items():
        mape = "-" if m["mape_pct"] is None else f"{m['mape_pct']:.2f}"
        med = "-" if m["median_ratio"] is None else f"{m['median_ratio']:.4f}"
        print(f"  {code:<10} {m['n']:>4} {mape:>10} {med:>15}")

    print()
    print("Ranked error contributions (largest first):")
    print(f"  {'source':<20} {'median ratio':>15} {'excess over 1':>15}")
    for entry in summary["ranked_contributions"]:
        print(
            f"  {entry['source']:<20} "
            f"{entry['median_ratio']:>15.4f} "
            f"{entry['excess_over_1']:>15.4f}"
        )

    print()
    print("Interpretation:")
    print("  - The largest 'excess over 1' is the biggest error source.")
    print("  - If 'depth' dominates -> fix Depth Pro or depth scale.")
    print("  - If 'mask' dominates  -> fix SAM2 / box / mask filtering.")
    print("  - If 'formula_residual' > 0.5 -> bug in volume.py itself.")
    print("=" * 70)
    print()