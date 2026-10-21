"""
FastAPI app for the ActualPlate vision inference service.

Two endpoints:

    GET  /health    readiness check (are models loaded?)
    POST /detect    image + labels -> per-food volume + metadata

The app holds no model state of its own. Model loading is owned by
models.py and is triggered lazily by the first /detect request.

Auth: if VISION_API_TOKEN is set in the environment, /detect requires
the header X-API-Token to match. If unset, the endpoint is open.
This is intended for local/Colab development; set the token in
production.

Response format:

The default response includes only summary fields. If the request
sets `include_arrays=true`, the response additionally includes
base64-encoded numpy arrays, downsampled to a 512-pixel max side:

    depth_map_b64    one per request, (H, W) float32
    mask_b64         one per food, (H, W) uint8

Downsampling bounds the response size; Cloudflare free tunnels
drop the connection on multi-megabyte bodies.

If the request sets `debug=true`, each food record includes a
`_debug` sub-dict from estimate_volume_cm3: plane normal, plane
offset, support inliers, per-stage ray distances, and the height
distribution. This runs on the full-resolution depth map and
mask on the server, so the numbers reflect the pipeline's actual
geometry.
"""

from __future__ import annotations

import base64
import io
import os
import shutil
import tempfile
import uuid

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from vision_service.inference.inference import detect_and_measure
from vision_service.inference.models import models_loaded
from vision_service.inference.volume import estimate_volume_cm3


VISION_API_TOKEN = os.environ.get("VISION_API_TOKEN")

# Maximum side length for arrays returned when include_arrays=true.
MAX_ARRAY_SIDE = 512

app = FastAPI(
    title="ActualPlate Vision API",
    version="1.0.0",
)


def _check_auth(x_api_token: str | None) -> None:
    if VISION_API_TOKEN is None:
        return
    if x_api_token != VISION_API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid API token.")


def _encode_npz(array: np.ndarray) -> str:
    buf = io.BytesIO()
    np.savez_compressed(buf, arr=array)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _downsample_u8(arr: np.ndarray, max_side: int = MAX_ARRAY_SIDE) -> np.ndarray:
    h, w = arr.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return arr
    scale = max_side / float(longest)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_NEAREST)


def _downsample_f32(arr: np.ndarray, max_side: int = MAX_ARRAY_SIDE) -> np.ndarray:
    h, w = arr.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return arr
    scale = max_side / float(longest)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_AREA)


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "service": "ActualPlate Vision API",
        "models_loaded": models_loaded(),
    }


@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    labels: str = Form(...),
    include_arrays: bool = Form(False),
    debug: bool = Form(False),
    focal_px: float | None = Form(None),
    x_api_token: str | None = Header(default=None),
) -> JSONResponse:

    _check_auth(x_api_token)

    request_id = str(uuid.uuid4())

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid content type: {file.content_type}",
        )

    food_labels = [l.strip() for l in labels.split(",") if l.strip()]
    if not food_labels:
        raise HTTPException(status_code=400, detail="No labels provided.")

    image_path: str | None = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".jpg",
        ) as tmp:
            shutil.copyfileobj(file.file, tmp)
            image_path = tmp.name

        result = detect_and_measure(
            image_path=image_path,
            labels=food_labels,
            focal_px_override=focal_px,
        )

        foods = []

        for food, mask in zip(result.foods, result.masks):

            food_record = {
                "label": food.label,
                "score": food.score,
                "mask_pixels": food.mask_pixels,
                "depth_m": food.depth_m,
                "bbox": getattr(food, "bbox", None),
            }

            try:
                volume_result = estimate_volume_cm3(
                    depth_map=result.depth_map,
                    mask=mask,
                    focal_length_px=result.focal_length_px,
                    return_debug=debug,
                )
                if debug:
                    dbg = volume_result.pop("_debug", None)
                    if dbg is not None:
                        food_record["_debug"] = dbg
                food_record.update(volume_result)
                food_record["volume_status"] = "valid"

            except ValueError as exc:
                food_record["volume_cm3"] = None
                food_record["volume_status"] = "unreliable"
                food_record["volume_error"] = str(exc)

            if include_arrays:
                m = np.asarray(mask, dtype=np.uint8)
                m = _downsample_u8(m, MAX_ARRAY_SIDE)
                food_record["mask_b64"] = _encode_npz(m)

            foods.append(food_record)

        response = {
            "foods": foods,
            "image_width": result.image_width,
            "image_height": result.image_height,
            "focal_length_px": result.focal_length_px,
            "depth_unit": result.depth_unit,
            "request_id": request_id,
        }

        if include_arrays:
            dm = np.asarray(result.depth_map, dtype=np.float32)
            dm = _downsample_f32(dm, MAX_ARRAY_SIDE)
            response["depth_map_b64"] = _encode_npz(dm)

        return JSONResponse(content=response)

    finally:
        if image_path and os.path.exists(image_path):
            os.remove(image_path)