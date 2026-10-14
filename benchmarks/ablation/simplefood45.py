"""
SimpleFood45 adapter.

Real phone photos of 45 food classes, with ground-truth weight,
volume, and energy in labels.csv.

labels.csv columns: image,label,weight,volume,energy

No ground-truth depth. No ground-truth mask. Only variant A of the
ablation produces a number against this dataset -- the full real
pipeline, which is the production-distribution MAPE measurement.

The food name is exposed via Item.food_label so the remote
predictor sends the food name to Grounding DINO, not the filename.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

from PIL import Image

from benchmarks.ablation.base import Adapter, Item


DEFAULT_ROOT = (
    Path.home() / "Downloads" / "simple_food_45" / "ordered_dataset"
)


class SimpleFood45Adapter(Adapter):

    @property
    def name(self) -> str:
        return "simplefood45"

    def __init__(self, root: Path | None = None) -> None:
        env_root = os.environ.get("SF45_ROOT")
        self.root = (
            Path(root) if root is not None
            else Path(env_root) if env_root
            else DEFAULT_ROOT
        )
        self.labels_path = self.root / "labels.csv"
        if not self.labels_path.exists():
            raise FileNotFoundError(
                f"labels.csv not found at {self.labels_path}. "
                "Set SF45_ROOT to the ordered_dataset folder."
            )

    def supports_inputs(self) -> set[str]:
        return {"volume", "weight", "energy"}

    def load_items(self, limit=None, category=None):
        items: list[Item] = []
        with open(self.labels_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = (row.get("label") or "").strip()
                image_name = (row.get("image") or "").strip()
                if not label or not image_name:
                    continue
                if category is not None and label != category:
                    continue

                image_path = self.root / image_name
                if not image_path.exists():
                    continue

                try:
                    volume = float(row.get("volume") or "")
                except (TypeError, ValueError):
                    continue
                if not (volume > 0):
                    continue

                try:
                    weight = float(row.get("weight") or "")
                except (TypeError, ValueError):
                    weight = None

                try:
                    with Image.open(image_path) as im:
                        im.verify()
                except Exception:
                    continue

                items.append(Item(
                    id=f"{label}/{image_path.stem}",
                    rgb_path=image_path,
                    focal_length_px=2000.0,
                    gt_volume_cm3=volume,
                    gt_weight_g=weight,
                    gt_depth_map=None,
                    gt_mask=None,
                    food_label=label,
                ))

                if limit is not None and len(items) >= limit:
                    return items

        return items

