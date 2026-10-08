"""
Vision inference orchestration.

Runs the full detection pipeline on a single image:

    image + labels
      -> Grounding DINO           (boxes, scores, text labels)
      -> SAM2                     (per-box masks)
      -> Apple Depth Pro          (metric depth map + focal length)
      -> per-food depth statistics

Returns an InferenceResult: a dataclass that separates JSON-safe
data from the internal arrays required by the volume engine.

No HTTP, no FastAPI, no request handling. That is api.py's job.

Debug visualizations:
    When SHOW_VISUALIZATIONS=true is set in the environment at the
    time of the call, this module writes PNG files to
    VIS_DIR/<request_id>/{boxes,masks,depth}.png.

    Files are used instead of plt.show() because matplotlib figures
    created inside a background thread (uvicorn runs in a thread)
    do not render in a notebook cell output channel. Files can be
    read from any cell, from any thread.

    The env var is read inside the function on every call, not at
    module import, so toggling it at runtime works without a
    process restart.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from vision_service.inference.models import (
    DEVICE,
    get_depth_pro,
    get_grounding_dino,
    get_sam2_predictor,
)


BOX_THRESHOLD = float(os.environ.get("BOX_THRESHOLD", "0.35"))
TEXT_THRESHOLD = float(os.environ.get("TEXT_THRESHOLD", "0.25"))


def _should_visualize() -> bool:
    """
    Read SHOW_VISUALIZATIONS on every call so that a running service
    can be toggled without a process restart.
    """
    return os.environ.get("SHOW_VISUALIZATIONS", "false").lower() == "true"


def _vis_dir() -> Path:
    return Path(os.environ.get("VIS_DIR", "/content/visualizations"))


@dataclass
class FoodDetection:
    """
    One detected food item.

    Every field here is JSON-safe. numpy types must be cast to Python
    scalars before being assigned.
    """

    label: str
    score: float
    mask_pixels: int
    depth_m: dict  # {"min": float | None, "median": ..., "max": ...}


@dataclass
class InferenceResult:
    """
    Result of one inference call.

    json_fields() returns the subset that is safe to serialize.
    depth_map and masks are used by the volume engine and never
    serialized by this module.
    """

    foods: list[FoodDetection]
    image_width: int
    image_height: int
    focal_length_px: float
    depth_unit: str

    depth_map: np.ndarray = field(repr=False)
    masks: list[np.ndarray] = field(repr=False)

    def json_fields(self) -> dict:
        return {
            "foods": [
                {
                    "label": f.label,
                    "score": f.score,
                    "mask_pixels": f.mask_pixels,
                    "depth_m": f.depth_m,
                }
                for f in self.foods
            ],
            "image_width": self.image_width,
            "image_height": self.image_height,
            "focal_length_px": self.focal_length_px,
            "depth_unit": self.depth_unit,
        }


# ---------------------------------------------------------------------------
# File-based debug visualizations.
#
# These write PNGs to disk. They do NOT call plt.show() because that
# does not render from the uvicorn background thread in a notebook.
# ---------------------------------------------------------------------------


def _save_boxes(
    image_np: np.ndarray,
    boxes: np.ndarray,
    labels: list[str],
    scores: np.ndarray,
    out_dir: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image_np)

    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        ax.add_patch(
            patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor="red",
                facecolor="none",
            )
        )
        ax.text(
            x1,
            max(y1 - 8, 5),
            f"{label} {score:.2f}",
            color="red",
            bbox={"facecolor": "white", "alpha": 0.8},
        )

    ax.axis("off")
    fig.savefig(out_dir / "boxes.png", bbox_inches="tight")
    plt.close(fig)


def _save_masks(
    image_np: np.ndarray,
    masks: list[np.ndarray],
    boxes: np.ndarray,
    labels: list[str],
    scores: np.ndarray,
    out_dir: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    count = len(masks)
    if count == 0:
        return

    fig, axes = plt.subplots(1, count, figsize=(5 * count, 5))
    if count == 1:
        axes = [axes]

    for ax, mask, box, label, score in zip(axes, masks, boxes, labels, scores):
        ax.imshow(image_np)
        ax.imshow(np.ma.masked_where(~mask, mask), alpha=0.6)
        x1, y1, x2, y2 = box
        ax.add_patch(
            patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor="red",
                facecolor="none",
            )
        )
        ax.set_title(f"{label} ({score:.2f})")
        ax.axis("off")

    fig.savefig(out_dir / "masks.png", bbox_inches="tight")
    plt.close(fig)


def _save_depth(depth_map: np.ndarray, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(depth_map, cmap="inferno")
    fig.colorbar(im, ax=ax, label="Depth (meters)")
    ax.set_title("Apple Depth Pro")
    ax.axis("off")
    fig.savefig(out_dir / "depth.png", bbox_inches="tight")
    plt.close(fig)


def _write_visualizations(
    request_id: str,
    image_np: np.ndarray,
    boxes: np.ndarray,
    detected_labels: list[str],
    scores: np.ndarray,
    masks: list[np.ndarray],
    depth_map: np.ndarray,
) -> None:
    """
    Write boxes.png, masks.png, depth.png to VIS_DIR/<request_id>/.

    Any failure here must not break the inference call. Visualizations
    are diagnostics, not part of the response contract.
    """
    try:
        out_dir = _vis_dir() / request_id
        out_dir.mkdir(parents=True, exist_ok=True)

        _save_boxes(image_np, boxes, detected_labels, scores, out_dir)
        _save_masks(image_np, masks, boxes, detected_labels, scores, out_dir)
        _save_depth(depth_map, out_dir)
    except Exception as exc:
        # Never let a visualization failure abort a real inference.
        print(f"[visualize] failed for {request_id}: {exc}", flush=True)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def detect_and_measure(
    image_path: str,
    labels: list[str],
    request_id: str = "no-request-id",
) -> InferenceResult:
    """
    Run the full inference pipeline on one image.

    Raises ValueError for malformed inputs. Other exceptions
    (model, CUDA, I/O) propagate to the caller.
    """
    if not labels:
        raise ValueError("At least one label is required.")

    processor, grounding_dino = get_grounding_dino()
    sam2_predictor = get_sam2_predictor()
    depth_pro, depth_model, depth_transform = get_depth_pro()

    image = Image.open(image_path).convert("RGB")
    image_np = np.asarray(image).copy()

    # --- Grounding DINO: image + text -> boxes + labels + scores.

    inputs = processor(
        images=image,
        text=". ".join(labels) + ".",
        return_tensors="pt",
    )
    inputs = {
        key: value.to(DEVICE) if torch.is_tensor(value) else value
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = grounding_dino(**inputs)

    detections = processor.post_process_grounded_object_detection(
        outputs,
        input_ids=inputs["input_ids"],
        threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        target_sizes=[image.size[::-1]],
    )[0]

    boxes = detections["boxes"].detach().cpu().numpy()
    scores = detections["scores"].detach().cpu().numpy()
    detected_labels = detections["text_labels"]

    # --- SAM2: image + boxes -> masks.

    sam2_predictor.set_image(image_np)

    masks = []
    for box in boxes:
        mask_result, _, _ = sam2_predictor.predict(
            box=box,
            multimask_output=False,
        )
        masks.append(np.asarray(mask_result[0], dtype=bool))

    # --- Depth Pro: image -> metric depth map + focal length.

    depth_image, _, f_px = depth_pro.load_rgb(image_path)
    depth_tensor = depth_transform(depth_image).to(DEVICE)

    with torch.no_grad():
        prediction = depth_model.infer(depth_tensor, f_px=f_px)

    depth_map = (
        prediction["depth"].detach().cpu().numpy().astype(np.float32)
    )

    focal_length_px = prediction["focallength_px"]
    if torch.is_tensor(focal_length_px):
        focal_length_px = focal_length_px.detach().cpu().item()
    else:
        focal_length_px = float(focal_length_px)

    if depth_map.shape != image_np.shape[:2]:
        depth_map = cv2.resize(
            depth_map,
            (image_np.shape[1], image_np.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    # --- Per-food aggregation: depth stats inside each mask.

    foods: list[FoodDetection] = []

    for label, score, mask in zip(detected_labels, scores, masks):
        if mask.shape != depth_map.shape:
            mask = cv2.resize(
                mask.astype(np.uint8),
                (depth_map.shape[1], depth_map.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

        food_depth = depth_map[mask]
        food_depth = food_depth[np.isfinite(food_depth)]

        depth_stats = {
            "min": float(np.min(food_depth)) if len(food_depth) else None,
            "median": float(np.median(food_depth)) if len(food_depth) else None,
            "max": float(np.max(food_depth)) if len(food_depth) else None,
        }

        foods.append(
            FoodDetection(
                label=str(label),
                score=float(score),
                mask_pixels=int(mask.sum()),
                depth_m=depth_stats,
            )
        )

    # --- Optional debug visualizations, written to disk.

    if _should_visualize():
        _write_visualizations(
            request_id=request_id,
            image_np=image_np,
            boxes=boxes,
            detected_labels=list(detected_labels),
            scores=scores,
            masks=masks,
            depth_map=depth_map,
        )

    return InferenceResult(
        foods=foods,
        image_width=image.width,
        image_height=image.height,
        focal_length_px=float(focal_length_px),
        depth_unit="meters",
        depth_map=depth_map,
        masks=masks,
    )