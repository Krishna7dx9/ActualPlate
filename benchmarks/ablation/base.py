"""
Adapter interface for the volume ablation harness.

The harness runs nine volume-estimation variants (A through I)
per food item. Each variant swaps one or more pipeline inputs
(depth, mask, dilation, scale) for a known reference value.

Two goals:

1. Isolate the contribution of each input to the total volume
   error: depth, mask, formula, dilation constant, scale
   sensitivity, mask-edge sensitivity.
2. Produce one CSV table with one row per item and one column
   per variant. Ratios A/D, B/D, ... I/D show which input
   contributes most.

The harness knows nothing about any specific dataset. Each
dataset provides an Adapter subclass. The adapter returns a
list of Item objects; everything else is common.

Datasets currently planned:

- MetaFood3D: Blender-rendered RGB + depth + transforms.json,
  with ground-truth volume in the bundled Excel. The depth is
  Blender's, the mask is derived from depth, the volume is from
  a 3D scan. All three inputs have ground truth.

- SimpleFood45: real phone photos with ground-truth weight and
  volume, but no ground-truth depth and no ground-truth mask.
  Only the full pipeline (variant A) can run. Used to measure
  real-world MAPE, not to isolate inputs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Item:
    """
    One food item, prepared for the ablation harness.

    Not every dataset provides every field. A field set to None
    means the dataset does not supply it; variants that need it
    will be skipped for this item, and the corresponding CSV
    column will be empty.

    Attributes:
        id: unique identifier, e.g. "Bagel/bagel_1"
        rgb_path: absolute path to the RGB image
        focal_length_px: camera focal length in pixels
        gt_volume_cm3: ground-truth volume from the dataset
        gt_weight_g: ground-truth weight, if available
        gt_depth_map: (H, W) float32 depth in meters, if available
        gt_mask: (H, W) bool food mask, if available
    """

    id: str
    rgb_path: Path
    focal_length_px: float
    gt_volume_cm3: float
    gt_weight_g: float | None = None
    gt_depth_map: np.ndarray | None = None
    gt_mask: np.ndarray | None = None
    food_label: str | None = None


class Adapter(ABC):
    """
    Abstract base class for a dataset adapter.

    A subclass knows how to read one specific dataset from disk
    and return a uniform list of Item objects. The harness does
    not import any dataset-specific code.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Short identifier used on the command line and in the CSV,
        e.g. "metafood3d" or "simplefood45".
        """

    @abstractmethod
    def load_items(
        self,
        limit: int | None = None,
        category: str | None = None,
    ) -> list[Item]:
        """
        Return a list of Item objects ready for ablation.

        Args:
            limit: if not None, return at most this many items.
            category: if not None, restrict to this category
                (e.g. "Bagel"). Adapters that do not have a
                category concept should ignore this argument.

        Raises:
            FileNotFoundError: if the dataset is not present at
                the expected location.
        """

    def supports_inputs(self) -> set[str]:
        """
        Declare which inputs this adapter can provide ground
        truth for.

        Returns a set containing any of:
            "depth"  - each Item has gt_depth_map
            "mask"   - each Item has gt_mask
            "volume" - each Item has gt_volume_cm3

        Used by the harness to decide which variants to run.
        Default assumes all three are supported. Override in
        subclasses that only support a subset.
        """
        return {"depth", "mask", "volume"}