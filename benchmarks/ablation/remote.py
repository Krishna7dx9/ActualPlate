"""
Remote predictions for the ablation harness.

Sends an RGB image + labels to the Colab /detect endpoint with
include_arrays=true. Decodes the returned base64 arrays into
numpy. Returns a Predictions object.

The Colab server must be running, reachable at the given endpoint,
and must be the branch that includes the include_arrays change.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
import requests

from benchmarks.ablation.base import Item
from benchmarks.ablation.variants import Predictions


def _decode_npz(b64: str) -> np.ndarray:
    raw = base64.b64decode(b64)
    with np.load(io.BytesIO(raw)) as data:
        return data["arr"]


def predictions_from_endpoint(
    item: Item,
    endpoint: str,
    timeout_s: int = 300,
    focal_px: float | None = None,
) -> Predictions:
    """
    POST one item to Colab /detect and return Predictions.

    focal_px, if provided, is sent as the `focal_px` form field.
    The server forwards it to Depth Pro as a focal override, used
    by the diagnostic to test Depth Pro's internal focal guess.
    """
    if getattr(item, "food_label", None):
        labels = item.food_label
    else:
        labels = item.id.split("/", 1)[-1]
        labels = labels.replace("_", " ").strip()

    data = {
        "labels": labels,
        "include_arrays": "true",
    }
    if focal_px is not None:
        data["focal_px"] = str(float(focal_px))

    with open(item.rgb_path, "rb") as f:
        response = requests.post(
            endpoint,
            files={"file": (Path(item.rgb_path).name, f, "image/png")},
            data=data,
            timeout=timeout_s,
        )

    response.raise_for_status()
    body = response.json()

    if "error" in body:
        raise ValueError(f"Colab /detect error: {body['error']}")
    if "depth_map_b64" not in body:
        raise KeyError(
            "response missing depth_map_b64; is the Colab server "
            "on the branch that includes the include_arrays change?"
        )

    depth_map = _decode_npz(body["depth_map_b64"])

    foods = body.get("foods", [])
    if not foods:
        raise ValueError("Colab returned no foods.")
    if "mask_b64" not in foods[0]:
        raise KeyError(
            "response missing mask_b64; is the Colab server on "
            "the branch that includes include_arrays?"
        )

    mask = _decode_npz(foods[0]["mask_b64"]).astype(bool)

    return Predictions(
        depth_map=depth_map.astype(np.float32),
        mask=mask,
        focal_length_px=float(body["focal_length_px"]),
    )
    """
    POST one item to Colab /detect and return Predictions.

    Args:
        item: the Item to process.
        endpoint: full URL to /detect, e.g.
            "https://xyz.trycloudflare.com/detect".
        timeout_s: HTTP timeout. Depth Pro + SAM2 take 30-60 s per
            image on the first call (cold), less after warmup.

    Raises:
        requests.HTTPError: on non-2xx responses.
        KeyError: if the response is missing expected fields.
        ValueError: if the response reports no detections.
    """
    labels = item.id.split("/", 1)[-1]      # food_type from "Bagel/bagel_1"
    labels = labels.replace("_", " ").strip()

    with open(item.rgb_path, "rb") as f:
        response = requests.post(
            endpoint,
            files={"file": (Path(item.rgb_path).name, f, "image/png")},
            data={
                "labels": labels,
                "include_arrays": "true",
            },
            timeout=timeout_s,
        )

    response.raise_for_status()
    body = response.json()

    if "error" in body:
        raise ValueError(f"Colab /detect error: {body['error']}")
    if "depth_map_b64" not in body:
        raise KeyError(
            "response missing depth_map_b64; is the Colab server "
            "on the branch that includes the include_arrays change?"
        )

    depth_map = _decode_npz(body["depth_map_b64"])

    foods = body.get("foods", [])
    if not foods:
        raise ValueError("Colab returned no foods.")
    if "mask_b64" not in foods[0]:
        raise KeyError(
            "response missing mask_b64; is the Colab server on "
            "the branch that includes include_arrays?"
        )

    # Use the first food's mask. The ablation harness sends one
    # label per request, so there should be exactly one food.
    mask = _decode_npz(foods[0]["mask_b64"]).astype(bool)

    return Predictions(
        depth_map=depth_map.astype(np.float32),
        mask=mask,
        focal_length_px=float(body["focal_length_px"]),
    )