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
import json

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from vision_service.inference.inference import detect_and_measure
from vision_service.inference.models import models_loaded
from vision_service.inference.volume import estimate_volume_cm3


VISION_API_TOKEN = os.environ.get("VISION_API_TOKEN")
if not VISION_API_TOKEN:
    raise ValueError("VISION_API_TOKEN environment variable must be set.")

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
    scale_map: str | None = Form(None), # <--- Replaced individual floats with map
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

    # Parse the scale map JSON string
    parsed_scale_map = {}
    if scale_map:
        try:
            parsed_scale_map = json.loads(scale_map)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid scale_map JSON format.")

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
            
            # Look up the correct scale for THIS specific food
            food_label_lower = food.label.strip().lower()
            food_scale = parsed_scale_map.get(food_label_lower, {})
            estimated_thickness_cm = food_scale.get("thickness")
            estimated_diameter_cm = food_scale.get("diameter")

            food_record = {
                "label": food.label,
                "score": food.score,
                "mask_pixels": food.mask_pixels,
                "depth_m": food.depth_m,
                "bbox": getattr(food, "bbox", None),
                "mask_b64": getattr(food, "mask_b64", None),
            }

            # --- NEW 3D NORMALIZATION MATH ---
            if estimated_thickness_cm is not None and estimated_diameter_cm is not None:
                food_depth = result.depth_map[mask]
                
                if len(food_depth) > 0:
                    min_d = float(np.min(food_depth))
                    max_d = float(np.max(food_depth))
                    
                    if max_d > min_d:
                        # 1. Normalize to 0-1 (Compresses the shape)
                        normalized = (max_d - result.depth_map) / (max_d - min_d)
                        normalized = np.clip(normalized, 0, 1)
                    else:
                        normalized = np.zeros_like(result.depth_map)
                        
                    # 2. Multiply by Gemma's real-world thickness
                    real_height = normalized * float(estimated_thickness_cm)
                    
                    # 3. Calculate real-world pixel area
                    bbox = getattr(food, "bbox", [0,0,0,0])
                    bbox_w_px = max(1.0, bbox[2] - bbox[0])
                    cm_per_px = float(estimated_diameter_cm) / bbox_w_px
                    pixel_area_cm2 = cm_per_px ** 2
                    
                    # 4. Integrate Volume (Sum of Height * Area for each pixel)
                    vol = float(np.sum(real_height[mask]) * pixel_area_cm2)
                    
                    food_record["volume_cm3"] = vol
                    food_record["volume_status"] = "valid"
                    food_record["scale_factor_applied"] = cm_per_px
                else:
                    food_record["volume_cm3"] = 0.0
                    food_record["volume_status"] = "unreliable"
            else:
                # Fallback to old math if VLM didn't provide data for this food
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

        # Encode REAL depth map to Base64 for the UI
        depth_m = result.depth_map
        
        # 1. Sanitize NaN/Inf
        depth_m = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)

        # 2. Create a combined mask of ALL foods to focus the color scale
        combined_mask = np.zeros_like(depth_m, dtype=bool)
        for mask in result.masks:
            if mask.shape == depth_m.shape:
                combined_mask |= mask

        # 3. Clip outliers based ONLY on the food's depth range
        food_depths = depth_m[combined_mask]
        if len(food_depths) > 100:
            p1, p99 = np.percentile(food_depths, [1, 99])
        else:
            valid_depths = depth_m[depth_m > 0]
            p1, p99 = np.percentile(valid_depths, [1, 99]) if len(valid_depths) > 0 else (0, 1)
            
        depth_m = np.clip(depth_m, p1, p99)
            
        depth_norm = cv2.normalize(depth_m, None, 0, 255, cv2.NORM_MINMAX)
        depth_uint8 = depth_norm.astype(np.uint8)
        
        # 4. Bilateral Filter: smooths flat areas (background) but preserves edges (food)
        depth_uint8 = cv2.bilateralFilter(depth_uint8, 9, 75, 75)
        
        # 5. Apply colormap
        depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)
        
        # 6. Dim the background (outside the mask) so the food pops and the halo noise is hidden
        depth_color[~combined_mask] = (depth_color[~combined_mask] * 0.3).astype(np.uint8)
        
        _, depth_buffer = cv2.imencode('.png', depth_color)
        depth_b64 = base64.b64encode(depth_buffer).decode('utf-8')

        response = {
            "foods": foods,
            "image_width": result.image_width,
            "image_height": result.image_height,
            "focal_length_px": result.focal_length_px,
            "depth_unit": result.depth_unit,
            "request_id": request_id,
            "depth_b64": depth_b64,
        }

        if include_arrays:
            dm = np.asarray(result.depth_map, dtype=np.float32)
            dm = _downsample_f32(dm, MAX_ARRAY_SIDE)
            response["depth_map_b64"] = _encode_npz(dm)

        return JSONResponse(content=response)

    finally:
        if image_path and os.path.exists(image_path):
            os.remove(image_path)