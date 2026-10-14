"""
Command-line entry point for the volume ablation harness.

Usage:

    # GT-only variants (D..I). No Colab needed. Runs on laptop.
    python -m benchmarks.ablation.cli --dataset metafood3d --limit 20

    # All variants (A..I). Requires live Colab /detect endpoint.
    python -m benchmarks.ablation.cli --dataset metafood3d --limit 20 \
        --predictions metafood3d --endpoint https://<tunnel>/detect

Behavior:

    - Loads items from the chosen dataset adapter.
    - Runs the nine variants A..I per item.
    - Writes results to benchmarks/ablation/results/<dataset>_<timestamp>.csv
    - Prints a summary with per-variant MAPE and ranked error
      contributions.

Predictions (Depth Pro + SAM2 output) are optional. When they
cannot be produced -- for example, on a laptop without a GPU and
without a live Colab endpoint -- the harness still runs the
GT-only variants D..I. Variants A, B, C are skipped in that
case because they need live predictions.

Prediction source is chosen with --predictions:

    none        (default) skip A, B, C
    metafood3d  POST each item to Colab /detect
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from benchmarks.ablation.base import Item
from benchmarks.ablation.variants import Predictions
from benchmarks.ablation.run import run_ablation
from benchmarks.ablation.report import (
    print_summary,
    summarize,
    write_csv,
)


RESULTS_DIR = Path(__file__).resolve().parent / "results"


# --- Prediction sources ---------------------------------------------


def predictions_none(item: Item) -> Predictions:
    """
    Sentinel for the 'no predictions' path.

    Raising NotImplementedError tells run_ablation to skip
    variants A, B, C for this item and run only the GT-only
    variants D..I.
    """
    raise NotImplementedError("PREDICTIONS=none")


def predictions_metafood3d(item: Item) -> Predictions:
    """
    Run Depth Pro and SAM2 on the item's RGB image by POSTing
    to the Colab /detect endpoint.

    The endpoint must be provided via the --endpoint CLI arg
    or the VISION_INFERENCE_ENDPOINT env var. It is not needed
    when running with --predictions none.
    """
    endpoint = _resolve_endpoint()
    from benchmarks.ablation.remote import predictions_from_endpoint
    return predictions_from_endpoint(item, endpoint)


def _resolve_endpoint() -> str:
    """
    Return the full URL to the /detect endpoint.
    """
    raw = os.environ.get("VISION_INFERENCE_ENDPOINT", "").strip()
    if not raw:
        raise ValueError(
            "predictions=metafood3d requires the endpoint. "
            "Set VISION_INFERENCE_ENDPOINT in .env or pass --endpoint."
        )
    if raw.endswith("/detect"):
        return raw
    return raw.rstrip("/") + "/detect"


def _resolve_predictions_fn(name: str):
    if name == "none":
        return predictions_none
    if name == "metafood3d":
        return predictions_metafood3d
    raise ValueError(f"unknown predictions value: {name}")


# --- Adapters --------------------------------------------------------


def _get_adapter(dataset: str):
    if dataset == "metafood3d":
        from benchmarks.ablation.metafood3d import MetaFood3DAdapter
        return MetaFood3DAdapter()
    if dataset == "simplefood45":
        from benchmarks.ablation.simplefood45 import SimpleFood45Adapter
        return SimpleFood45Adapter()
    raise ValueError(f"unknown dataset: {dataset}")


# --- CLI -------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.ablation.cli",
        description="Volume ablation harness",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["metafood3d", "simplefood45"],
        help="Dataset adapter to use.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of items to process.",
    )
    parser.add_argument(
        "--category",
        default=None,
        help="Restrict to one category, e.g. 'Bagel'.",
    )
    parser.add_argument(
        "--predictions",
        default=os.environ.get("PREDICTIONS", "none"),
        help="Prediction source: none (default), metafood3d.",
    )
    parser.add_argument(
        "--endpoint",
        default=None,
        help=(
            "Colab /detect endpoint. Only needed when "
            "--predictions=metafood3d. Can also be set via "
            "VISION_INFERENCE_ENDPOINT."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "CSV path. Default: "
            "benchmarks/ablation/results/<dataset>_<ts>.csv"
        ),
    )

    args = parser.parse_args(argv)

    if args.endpoint:
        os.environ["VISION_INFERENCE_ENDPOINT"] = args.endpoint

    print("=" * 70)
    print("VOLUME ABLATION HARNESS")
    print("=" * 70)
    print(f"  dataset:     {args.dataset}")
    print(f"  category:    {args.category or '(all)'}")
    print(
        f"  limit:       "
        f"{args.limit if args.limit is not None else '(all)'}"
    )
    print(f"  predictions: {args.predictions}")
    print("=" * 70)
    print()

    adapter = _get_adapter(args.dataset)
    predictions_fn = _resolve_predictions_fn(args.predictions)

    rows = run_ablation(
        adapter=adapter,
        predictions_fn=predictions_fn,
        limit=args.limit,
        category=args.category,
    )

    if not rows:
        print("No items processed. Nothing to write.")
        return 1

    # --- CSV output ---

    if args.output:
        out_path = Path(args.output)
    else:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = RESULTS_DIR / f"{args.dataset}_{ts}.csv"

    write_csv(rows, out_path)
    print(f"\nCSV written to: {out_path}")

    # --- Summary ---

    summary = summarize(rows)
    print_summary(summary)

    return 0


if __name__ == "__main__":
    sys.exit(main())