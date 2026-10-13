"""
Main loop for the volume ablation harness.

For each item from the dataset adapter:

    1. Run the ML pipeline once per item (Depth Pro + SAM2) to
       get Predictions. This requires a GPU and only runs on the
       host that has it (Colab).

    2. Run all nine variants A..I using either Predictions or
       the ground truth stored in the Item.

    3. Record the volume each variant produced, and compute the
       ratios against the ground-truth volume.

The loop has no dependency on Colab. The Predictions are
supplied by a caller (a Colab cell or a test harness). When
Predictions cannot be computed, variants A, B, C still work
because B and C only need the GT side; only A needs live
Depth Pro + SAM2 output.

Two ways to use this module:

    - Directly from Python:

        from benchmarks.ablation.run import run_ablation
        rows = run_ablation(
            adapter,
            predictions_fn=my_fn,
            limit=20,
        )

    - From the CLI (cli.py) which wires adapter + predictions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from benchmarks.ablation.base import Adapter, Item
from benchmarks.ablation.variants import (
    VARIANTS,
    Predictions,
)


@dataclass
class Row:
    """One row of the ablation CSV."""

    item_id: str
    category: str
    gt_volume_cm3: float
    volumes: dict[str, float | None]   # variant -> volume or None
    ratios: dict[str, float | None]    # variant -> ratio vs GT


PredictionsFn = Callable[[Item], Predictions]


def _category_of(item_id: str) -> str:
    """Return the category part of an item id like 'Bagel/bagel_1'."""
    return item_id.split("/", 1)[0] if "/" in item_id else ""


def run_ablation(
    adapter: Adapter,
    predictions_fn: PredictionsFn,
    limit: int | None = None,
    category: str | None = None,
) -> list[Row]:
    """
    Run the nine-variant ablation over the items in the adapter.

    Args:
        adapter: dataset adapter that returns Item objects.
        predictions_fn: callable that maps an Item to a
            Predictions object. May raise NotImplementedError;
            in that case variants A, B, C are skipped for that
            item but the GT-only variants D..I still run.
        limit: max number of items to process.
        category: restrict to one category (optional).

    Returns:
        List of Row objects, one per item.
    """

    items = adapter.load_items(limit=limit, category=category)

    rows: list[Row] = []

    for i, item in enumerate(items, start=1):
        print(f"[{i}/{len(items)}] {item.id}", flush=True)

        # --- Try to run the ML pipeline ---

        predictions: Predictions | None = None
        try:
            predictions = predictions_fn(item)
        except NotImplementedError:
            print("    (no predictions: running GT-only variants)")
        except Exception as exc:
            print(f"    predictions failed: {exc}")

        # --- Run each variant ---

        volumes: dict[str, float | None] = {}
        for code, fn in VARIANTS.items():
            try:
                if predictions is None:
                    # Variants B and C use GT + predictions. D..I use
                    # only GT. A requires predictions.
                    # We pass a sentinel "None" predictions object that
                    # will raise on attribute access inside the variant.
                    # Simpler: skip A, B, C if no predictions.
                    if code in ("A", "B", "C"):
                        volumes[code] = None
                        continue

                    # D..I need only GT; pass a dummy Predictions
                    # object whose fields are not used.
                    dummy = Predictions(
                        depth_map=None,        # type: ignore[arg-type]
                        mask=None,             # type: ignore[arg-type]
                        focal_length_px=item.focal_length_px,
                    )
                    volumes[code] = float(fn(item, dummy))

                else:
                    volumes[code] = float(fn(item, predictions))
            except ValueError as exc:
                # Variant could not run (missing input, geometry
                # unreliable). Record as None.
                volumes[code] = None
                print(f"    variant {code}: skipped ({exc})")
            except Exception as exc:
                volumes[code] = None
                print(f"    variant {code}: ERROR {exc}")

        # --- Compute ratios against GT volume ---

        ratios: dict[str, float | None] = {}
        for code, v in volumes.items():
            if v is None or item.gt_volume_cm3 <= 0:
                ratios[code] = None
            else:
                ratios[code] = v / item.gt_volume_cm3

        rows.append(
            Row(
                item_id=item.id,
                category=_category_of(item.id),
                gt_volume_cm3=item.gt_volume_cm3,
                volumes=volumes,
                ratios=ratios,
            )
        )

    return rows