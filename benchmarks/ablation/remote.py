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


def _post_with_retry(url, *, files, data, timeout_s, attempts=4):
    """
    POST with retry on transient network errors.

    Cloudflare free tunnels drop connections during long GPU
    inference. Retry the request with exponential backoff. Only
    retry on network-level failures (5xx from Cloudflare, connection
    errors, timeouts). Do not retry on 4xx -- those are real bugs.
    """
    import time

    last_exc = None
    for i in range(attempts):
        try:
            r = requests.post(
                url, files=files, data=data, timeout=timeout_s
            )
            if r.status_code < 500:
                return r
            last_exc = requests.HTTPError(
                f"{r.status_code} from tunnel"
            )
        except (
            requests.ConnectionError,
            requests.Timeout,
        ) as exc:
            last_exc = exc

        wait = min(2 ** i, 30)
        print(f"    retry {i + 1}/{attempts} after {wait}s: {last_exc}")
        time.sleep(wait)
        files = {
            k: (
                v[0],
                open(v[1].name, "rb") if hasattr(v[1], "name") else v[1],
                v[2],
            )
            for k, v in files.items()
        }

    if last_exc:
        raise last_exc
    raise RuntimeError("retry loop exited without result")


def predictions_from_endpoint(
    item: Item,
    endpoint: str,
    timeout_s: int = 300,
    focal_px: float | None = None,
) -> Predictions:
    """
    POST one item to Colab /detect and return Predictions.

    Args:
        item: the Item to process.
        endpoint: full URL to /detect.
        timeout_s: HTTP timeout. Depth Pro + SAM2 take 30-60 s per
            image on the first call (cold), less after warmup.
        focal_px: if provided, sent as the `focal_px` form field.
            The server forwards it to Depth Pro as a focal override.

    Raises:
        requests.HTTPError: on non-2xx responses after retries.
        KeyError: if the response is missing expected fields.
        ValueError: if the response reports no detections.
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
        response = _post_with_retry(
            endpoint,
            files={"file": (Path(item.rgb_path).name, f, "image/png")},
            data=data,
            timeout_s=timeout_s,
            attempts=4,
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