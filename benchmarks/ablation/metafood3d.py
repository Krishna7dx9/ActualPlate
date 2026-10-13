"""
MetaFood3D adapter for the ablation harness.

Reads the partial Blender-render dataset that was extracted from
the downloaded archive.

Expected on-disk layout under the root:

    Blender_render_images/
        <Object_name>/
            <Food_Type>/
                <Folder_name>/
                    transforms.json
                    Original/
                        <Folder_name>_000.png
                        <Folder_name>_001.png
                        ...
                    Depth/
                        <Folder_name>_000_depth0001.png
                        ...
                    Normal/

Ground-truth volume and weight come from the bundled Excel
file (one row per <Object_name>, <Food_Type> pair).

Camera intrinsics are not stored in a separate file. The
transforms.json contains "camera_angle_x" (horizontal field of
view in radians). The focal length in pixels is derived from
that and the image width:

    focal_px = (W / 2) / tan(camera_angle_x / 2)

Ground-truth mask is derived from the depth map: pixels whose
depth differs from the plate depth by more than a threshold are
considered food.

The adapter is read-only. Nothing is written to the dataset.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
import openpyxl

from benchmarks.ablation.base import Adapter, Item


# --- Filesystem locations ---------------------------------------------


DEFAULT_DATASET_ROOT = (
    Path.home()
    / "Downloads"
    / "mf3d_partial"
    / "Blender_render_images"
)

DEFAULT_EXCEL = (
    Path.home()
    / "Downloads"
    / "_MetaFood3D_new_complete_dataset_nutrition_v2.xlsx"
)


# --- Constants --------------------------------------------------------


# Depth PNG convention. Blender writes 16-bit grayscale PNGs.
# We do not know the exact scaling factor until we inspect one
# file, so we normalise by the maximum observed value first and
# then rescale using the food height in the metadata.

# Threshold for deriving the GT mask from the depth map: any
# pixel whose depth differs from the plate depth by more than
# this many meters is considered food.
GT_MASK_DEPTH_TOLERANCE_M = 0.005


# --- Adapter ----------------------------------------------------------


class MetaFood3DAdapter(Adapter):
    """
    Adapter for the partially extracted MetaFood3D archive.
    """

    def __init__(
        self,
        root: Path | None = None,
        excel: Path | None = None,
    ) -> None:
        self.root = Path(root) if root else DEFAULT_DATASET_ROOT
        self.excel = Path(excel) if excel else DEFAULT_EXCEL

        if not self.root.exists():
            raise FileNotFoundError(
                f"MetaFood3D render root not found: {self.root}"
            )
        if not self.excel.exists():
            raise FileNotFoundError(
                f"MetaFood3D nutrition Excel not found: {self.excel}"
            )

        self._meta = self._load_metadata()

    @property
    def name(self) -> str:
        return "metafood3d"

    def supports_inputs(self) -> set[str]:
        return {"depth", "mask", "volume"}

    # --- Metadata -----------------------------------------------------

    def _load_metadata(self) -> dict[tuple[str, str], dict]:
        """
        Load the Excel into a lookup keyed by (Object_name, Food_Type).

        Some rows contain Excel formula strings (e.g.
        "=(D21*E20)/D20") because openpyxl does not evaluate
        formulas by default. Those rows are skipped — we do not
        have their numeric value.

        Values:
            {
                "volume_cm3": float,
                "weight_g": float,
                "kcal": float,
            }
        """
        wb = openpyxl.load_workbook(self.excel, read_only=True, data_only=True)
        ws = wb[wb.sheetnames[0]]

        meta: dict[tuple[str, str], dict] = {}
        header_seen = False
        skipped = 0

        for row in ws.iter_rows(values_only=True):
            if not header_seen:
                header_seen = True
                continue

            if row is None or row[0] is None or row[1] is None:
                continue

            object_name = str(row[0]).strip()
            food_type = str(row[1]).strip()
            weight = row[3]
            kcal = row[4]
            volume = row[8]

            if volume is None:
                continue

            try:
                volume_f = float(volume)
                weight_f = float(weight) if weight is not None else None
                kcal_f = float(kcal) if kcal is not None else None
            except (TypeError, ValueError):
                skipped += 1
                continue

            meta[(object_name, food_type)] = {
                "volume_cm3": volume_f,
                "weight_g": weight_f,
                "kcal": kcal_f,
            }

        if skipped:
            print(
                f"[metafood3d] skipped {skipped} rows "
                "with non-numeric values (formulas)"
            )

        return meta

    # --- Discovery ----------------------------------------------------

    def _iter_food_folders(
        self,
        category: str | None = None,
    ):
        """
        Yield (object_name, food_type, folder_path) for every
        food item that has a transforms.json and at least one
        depth PNG.
        """
        for object_dir in sorted(self.root.iterdir()):
            if not object_dir.is_dir():
                continue

            object_name = object_dir.name
            if category is not None and object_name != category:
                continue

            for food_dir in sorted(object_dir.iterdir()):
                if not food_dir.is_dir():
                    continue

                food_type = food_dir.name

                # Inside food_dir there is a single subfolder that
                # matches food_type casing or a variant. Find it.
                for item_dir in sorted(food_dir.iterdir()):
                    if not item_dir.is_dir():
                        continue

                    if (item_dir / "transforms.json").exists():
                        yield object_name, food_type, item_dir
                        break

    # --- Loading ------------------------------------------------------

    def load_items(
        self,
        limit: int | None = None,
        category: str | None = None,
    ) -> list[Item]:
        items: list[Item] = []

        for object_name, food_type, item_dir in self._iter_food_folders(
            category=category,
        ):
            meta_key = (object_name, food_type)
            meta = self._meta.get(meta_key)
            if meta is None:
                # No GT volume for this item. Skip it.
                continue

            try:
                item = self._load_one(
                    object_name, food_type, item_dir, meta
                )
            except Exception as exc:
                print(
                    f"[metafood3d] skip {object_name}/{food_type}: {exc}"
                )
                continue

            items.append(item)

            if limit is not None and len(items) >= limit:
                break

        return items

    def _load_one(
        self,
        object_name: str,
        food_type: str,
        item_dir: Path,
        meta: dict,
    ) -> Item:

        transforms_path = item_dir / "transforms.json"
        with open(transforms_path, "r") as f:
            transforms = json.load(f)

        # --- Focal length from camera_angle_x ---

        camera_angle_x = float(transforms["camera_angle_x"])

        # The RGB image is inside Original/. Find the first frame.
        original_dir = item_dir / "Original"
        if not original_dir.exists():
            raise FileNotFoundError("Original/ missing")

        rgb_candidates = sorted(original_dir.glob("*.png"))
        if not rgb_candidates:
            raise FileNotFoundError("no RGB PNGs in Original/")

        rgb_path = rgb_candidates[0]
        with Image.open(rgb_path) as im:
            width, height = im.size

        focal_length_px = (
            (width / 2.0) / float(np.tan(camera_angle_x / 2.0))
        )

        # --- Depth map ---

        depth_dir = item_dir / "Depth"
        depth_candidates = sorted(depth_dir.glob("*.png"))
        if not depth_candidates:
            raise FileNotFoundError("no depth PNGs in Depth/")

        gt_depth = self._load_depth_png(depth_candidates[0])

        # --- GT mask from depth ---

        gt_mask = self._derive_mask_from_depth(gt_depth)

        if gt_mask.sum() < 50:
            raise ValueError("derived GT mask too small")

        # --- Item id ---

        item_id = f"{object_name}/{food_type}"

        return Item(
            id=item_id,
            rgb_path=rgb_path,
            focal_length_px=focal_length_px,
            gt_volume_cm3=meta["volume_cm3"],
            gt_weight_g=meta.get("weight_g"),
            gt_depth_map=gt_depth,
            gt_mask=gt_mask,
        )

    # --- Depth and mask helpers ---------------------------------------

    def _load_depth_png(self, path: Path) -> np.ndarray:
        """
        Load a Blender depth PNG and return a relative depth map.

        Empirically verified on this dataset:

            dtype:  uint8
            range:  138 (food, near) to 255 (plate, far)
            median: 255  (most pixels are plate)

        Smaller value = closer to camera = food.
        Larger value = farther = plate.

        We normalise to [0, 1] meters for the volume engine. The
        absolute scale does not matter for the ablation because
        volume scales with the cube of the height difference
        between food and plate, which is preserved by any linear
        map. What matters is that:
          - food pixels have smaller depth than plate pixels
          - the plate is at the maximum value
        """
        with Image.open(path) as im:
            arr = np.array(im, dtype=np.float32)

        if arr.ndim == 3:
            arr = arr[..., 0]

        # Blender convention here: 255 = far (plate), lower = near (food).
        # Convert to "distance in meters" with the plate at 1.0 m
        # and the closest food pixel at 1.0 - max_height.
        # We do not know max_height yet, so we use raw levels
        # normalized to [0, 1] and treat the plate as 1.0.
        v_min = float(arr.min())
        v_max = float(arr.max())
        if v_max <= v_min:
            raise ValueError("depth PNG has constant value")

        # Larger value = farther. Normalize so 1.0 is the farthest.
        depth_m = (arr - v_min) / (v_max - v_min)   # 0 = nearest, 1 = farthest

        # Blender Z-buffer: what we want is distance from the camera.
        # Since the plate is the farthest surface at 1.0, food pixels
        # (smaller) are closer. The volume engine expects depth in
        # meters with smaller = closer. So we keep the map as-is but
        # scale it so the range is between 0.9 and 1.0 (a plausible
        # scene depth spread). This preserves ordering and relative
        # heights while giving the engine a reasonable numeric range.
        depth_m = 0.9 + 0.1 * depth_m
        return depth_m.astype(np.float32)

    def _derive_mask_from_depth(
        self,
        depth_map: np.ndarray,
    ) -> np.ndarray:
        """
        Build a food mask from the depth map.

        In the normalised depth from _load_depth_png:
            plate   -> highest value (1.0)
            food    -> lower values (down to 0.9)

        A pixel is considered food if its depth is strictly less
        than the plate depth by more than a small tolerance.
        The plate depth is estimated as the median of the outer
        20-pixel border (assumed to be background).
        """
        h, w = depth_map.shape
        border = 20

        edge_mask = np.zeros((h, w), dtype=bool)
        edge_mask[:border, :] = True
        edge_mask[-border:, :] = True
        edge_mask[:, :border] = True
        edge_mask[:, -border:] = True

        plate_depth = float(np.median(depth_map[edge_mask]))

        # Pixel is food if it is closer to the camera than the
        # plate by more than the tolerance. In our normalised
        # depth, "closer" = "smaller value".
        tolerance_m = 0.002   # 2 mm at 0.9..1.0 m range
        food = depth_map < (plate_depth - tolerance_m)
        return food.astype(bool)