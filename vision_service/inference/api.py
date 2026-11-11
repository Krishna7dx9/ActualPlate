"""
FastAPI app for the ActualPlate vision inference service.

Two endpoints:

    GET  /health    readiness check (are models loaded?)
    POST /detect    image + labels -> per-food volume + metadata

The app holds no model state of its own. Model loading is owned by
models.py and is triggered lazily by the first /detect request.

Auth: when VISION_API_TOKEN is configured, /detect requires the header
X-API-Token to match. Tokenless access is allowed only outside production.
Production startup fails if VISION_API_TOKEN is missing.

Response format:

The default response includes only summary fields. If the request
sets `include_arrays=true`, the response additionally includes
base64-encoded numpy arrays, downsampled to a 512-pixel max side:

    depth_map_b64    one per request, (H, W) float32
    mask_b64         one per food, (H, W) uint8

Downsampling bounds the response size; Cloudflare free tunnels
drop the connection on multi-megabyte bodies.

VLM-provided dimensions are not metric calibration evidence. The API
therefore returns `volume_cm3: null` until an independently validated
metric scale source is available. A separately named
`experimental_volume_cm3` may be returned for research diagnostics,
with `volume_status: "unvalidated"`; it must not be used for nutrition
or customer-facing physical measurements.
"""

from __future__ import annotations

import base64
import io
import os
import shutil
import tempfile
import uuid
import json
import secrets

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from vision_service.inference.inference import detect_and_measure
from vision_service.inference.models import models_loaded


APP_ENV = os.environ.get("APP_ENV", "development").strip().lower()
VISION_API_TOKEN = os.environ.get("VISION_API_TOKEN")
VISION_API_TOKEN = VISION_API_TOKEN.strip() if VISION_API_TOKEN else None


def _validate_auth_configuration(
    app_environment: str,
    api_token: str | None,
) -> None:
    """Require service-to-service authentication in production deployments."""
    if app_environment == "production" and not api_token:
        raise RuntimeError(
            "VISION_API_TOKEN must be configured when APP_ENV=production."
        )


_validate_auth_configuration(APP_ENV, VISION_API_TOKEN)

# Maximum side length for arrays returned when include_arrays=true.
MAX_ARRAY_SIDE = 512

app = FastAPI(
    title="ActualPlate Vision API",
    version="1.0.0",
)


def _check_auth(x_api_token: str | None) -> None:
    # Tokenless access is supported only outside production. Production
    # configuration is rejected during application initialization.
    if VISION_API_TOKEN is None:
        return
    if not secrets.compare_digest(x_api_token or "", VISION_API_TOKEN):
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
            food_scale = {}
            
            # 1. Try exact match
            if food_label_lower in parsed_scale_map:
                food_scale = parsed_scale_map[food_label_lower]
            else:
                # 2. Controlled Alias Match (Handles known DINO truncations safely)
                aliases = {
                    "corn": "corn on the cob",
                    "nuggets": "chicken nugget",
                    "fries": "french fries"
                }
                alias_target = aliases.get(food_label_lower, food_label_lower)
                
                if alias_target in parsed_scale_map:
                    food_scale = parsed_scale_map[alias_target]
                else:
                    # 3. Strict unambiguous substring match
                    # ONLY use it if there is exactly ONE match to prevent wrong scale assignment
                    possible_matches = [k for k in parsed_scale_map.keys() if food_label_lower in k or k in food_label_lower]
                    if len(possible_matches) == 1:
                        food_scale = parsed_scale_map[possible_matches[0]]
                        
            estimated_thickness_cm = food_scale.get("thickness")
            estimated_diameter_cm = food_scale.get("diameter")

            food_record = {
                "label": food.label,
                "score": food.score,
                "mask_pixels": food.mask_pixels,
                "depth_stats": food.depth_stats,
                "bbox": getattr(food, "bbox", None),
                "mask_b64": getattr(food, "mask_b64", None),
            }

            # VLM dimensions are visual estimates, not metric calibration evidence.
            # Keep the heuristic only as an explicitly experimental diagnostic.
            estimated_thickness = food_scale.get("thickness")
            estimated_diameter = food_scale.get("diameter")
            has_candidate_dimensions = (
                isinstance(estimated_thickness, (int, float))
                and np.isfinite(estimated_thickness)
                and estimated_thickness > 0
                and isinstance(estimated_diameter, (int, float))
                and np.isfinite(estimated_diameter)
                and estimated_diameter > 0
            )

            food_record["volume_cm3"] = None
            food_record["experimental_volume_cm3"] = None
            food_record["volume_status"] = "unavailable"
            food_record["volume_reason"] = (
                "Metric scale evidence is required before reporting physical volume."
            )

            if has_candidate_dimensions:
                food_record["volume_status"] = "unvalidated"
                food_record["volume_reason"] = (
                    "Volume uses VLM-estimated dimensions without "
                    "independent metric calibration."
                )

                food_depth = result.depth_map[mask]
                food_depth = food_depth[np.isfinite(food_depth)]

                bbox = getattr(food, "bbox", None)
                if (
                    food_depth.size > 0
                    and bbox is not None
                    and len(bbox) == 4
                    and float(bbox[2]) > float(bbox[0])
                ):
                    estimated_cm_per_pixel = (
                        float(estimated_diameter)
                        / float(bbox[2] - bbox[0])
                    )
                    food_record["experimental_scale_factor_cm_per_px"] = (
                        estimated_cm_per_pixel
                    )

                    min_depth = float(np.min(food_depth))
                    max_depth = float(np.max(food_depth))
                    if max_depth > min_depth:
                        normalized = (max_depth - result.depth_map) / (
                            max_depth - min_depth
                        )
                        normalized = np.clip(normalized, 0, 1)
                        estimated_height = (
                            normalized * float(estimated_thickness)
                        )
                        food_record["experimental_volume_cm3"] = float(
                            np.sum(estimated_height[mask])
                            * estimated_cm_per_pixel ** 2
                        )

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