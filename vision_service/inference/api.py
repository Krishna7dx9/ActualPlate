"""
FastAPI app for the ActualPlate vision inference service.

GET /health reports process liveness and whether models have loaded.
POST /detect accepts an image and labels and returns per-food metadata.

Models are loaded lazily on the first /detect request. Authentication is
required when VISION_API_TOKEN is configured; production startup rejects a
missing token.

The response always includes the PNG mask used by the backend visualization
pipeline. When include_arrays=true, downsampled NumPy arrays are additionally
returned as NPZ payloads under mask_array_b64 and depth_map_b64.

VLM-provided dimensions are not metric calibration evidence. The API returns
volume_cm3=null until an independently validated metric-scale source exists.
A separately named experimental_volume_cm3 may be returned for research
diagnostics with volume_status="unvalidated"; it must not be used for nutrition
or customer-facing physical measurements.
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
import secrets
import shutil
import tempfile
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from vision_service.inference.inference import detect_and_measure
from vision_service.inference.models import models_loaded
from vision_service.settings import settings


APP_ENV = settings.app_env
VISION_API_TOKEN = settings.api_token


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

MAX_ARRAY_SIDE = settings.max_array_side


def _load_label_aliases() -> dict[str, str]:
    configured_path = os.environ.get("VISION_LABEL_ALIASES_FILE")
    alias_path = (
        Path(configured_path).expanduser()
        if configured_path
        else Path(__file__).resolve().parents[1]
        / "recognition"
        / "label_aliases.json"
    )
    try:
        with alias_path.open("r", encoding="utf-8") as alias_file:
            raw_aliases = json.load(alias_file)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("Unable to load vision label alias configuration.") from exc

    if not isinstance(raw_aliases, dict):
        raise ValueError("Vision label aliases must be a JSON object.")

    aliases: dict[str, str] = {}
    for key, value in raw_aliases.items():
        if (
            not isinstance(key, str)
            or not key.strip()
            or not isinstance(value, str)
            or not value.strip()
        ):
            raise ValueError("Vision label aliases must map non-empty strings.")
        aliases[key.strip().lower()] = value.strip().lower()
    return aliases


LABEL_ALIASES = _load_label_aliases()

app = FastAPI(
    title="ActualPlate Vision API",
    version="1.0.0",
)


def _check_auth(x_api_token: str | None) -> None:
    """Compare service tokens without leaking timing information."""
    if VISION_API_TOKEN is None:
        return
    if not secrets.compare_digest(x_api_token or "", VISION_API_TOKEN):
        raise HTTPException(status_code=401, detail="Invalid API token.")


def _encode_npz(array: np.ndarray) -> str:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, arr=array)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _downsample_u8(arr: np.ndarray, max_side: int = MAX_ARRAY_SIDE) -> np.ndarray:
    height, width = arr.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return arr
    scale = max_side / float(longest)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    return cv2.resize(
        arr,
        (new_width, new_height),
        interpolation=cv2.INTER_NEAREST,
    )


def _downsample_f32(arr: np.ndarray, max_side: int = MAX_ARRAY_SIDE) -> np.ndarray:
    height, width = arr.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return arr
    scale = max_side / float(longest)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    return cv2.resize(
        arr,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA,
    )


def _parse_scale_map(raw_scale_map: str | None) -> dict[str, dict[str, float | None]]:
    """Parse and validate per-label dimension estimates from the backend."""
    if not raw_scale_map:
        return {}

    def reject_nonstandard_constant(value: str):
        raise ValueError(f"Invalid numeric constant: {value}")

    try:
        parsed = json.loads(
            raw_scale_map,
            parse_constant=reject_nonstandard_constant,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail="Invalid scale_map JSON format.",
        ) from exc

    if not isinstance(parsed, dict):
        raise HTTPException(
            status_code=400,
            detail="scale_map must be a JSON object keyed by food label.",
        )

    normalized: dict[str, dict[str, float | None]] = {}
    for label, dimensions in parsed.items():
        if not isinstance(label, str) or not label.strip():
            raise HTTPException(
                status_code=400,
                detail="scale_map labels must be non-empty strings.",
            )
        if not isinstance(dimensions, dict):
            raise HTTPException(
                status_code=400,
                detail="Each scale_map value must be an object.",
            )

        validated_dimensions: dict[str, float | None] = {}
        for field_name in ("thickness", "diameter"):
            value = dimensions.get(field_name)
            if value is None:
                validated_dimensions[field_name] = None
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value <= 0
            ):
                raise HTTPException(
                    status_code=400,
                    detail=f"scale_map {field_name} values must be positive finite numbers or null.",
                )
            validated_dimensions[field_name] = float(value)

        normalized[label.strip().lower()] = validated_dimensions

    return normalized


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
    scale_map: str | None = Form(None),
    x_api_token: str | None = Header(default=None),
) -> JSONResponse:
    _check_auth(x_api_token)

    request_id = str(uuid.uuid4())
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid content type: {file.content_type}",
        )

    food_labels = [label.strip() for label in labels.split(",") if label.strip()]
    if not food_labels:
        raise HTTPException(status_code=400, detail="No labels provided.")

    parsed_scale_map = _parse_scale_map(scale_map)
    image_path: str | None = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temporary:
            shutil.copyfileobj(file.file, temporary)
            image_path = temporary.name

        result = detect_and_measure(
            image_path=image_path,
            labels=food_labels,
            focal_px_override=focal_px,
        )

        foods = []
        for food, mask in zip(result.foods, result.masks):
            food_label_lower = str(food.label).strip().lower()
            food_scale = parsed_scale_map.get(food_label_lower, {})

            if not food_scale:
                alias_target = LABEL_ALIASES.get(
                    food_label_lower,
                    food_label_lower,
                )
                food_scale = parsed_scale_map.get(alias_target, {})

            if not food_scale:
                possible_matches = [
                    key
                    for key in parsed_scale_map
                    if food_label_lower in key or key in food_label_lower
                ]
                if len(possible_matches) == 1:
                    food_scale = parsed_scale_map[possible_matches[0]]

            estimated_thickness = food_scale.get("thickness")
            estimated_diameter = food_scale.get("diameter")
            has_candidate_dimensions = (
                estimated_thickness is not None
                and estimated_diameter is not None
            )

            food_record = {
                "label": food.label,
                "score": food.score,
                "mask_pixels": food.mask_pixels,
                "depth_stats": food.depth_stats,
                "bbox": getattr(food, "bbox", None),
                # Keep the PNG mask stable for the backend's visualization path.
                "mask_b64": getattr(food, "mask_b64", None),
                "volume_cm3": None,
                "experimental_volume_cm3": None,
                "volume_status": "unavailable",
                "volume_reason": (
                    "Metric scale evidence is required before reporting physical volume."
                ),
            }

            if has_candidate_dimensions:
                food_record["volume_status"] = "unvalidated"
                food_record["volume_reason"] = (
                    "Volume uses VLM-estimated dimensions without "
                    "independent metric calibration."
                )

                finite_food_mask = mask & np.isfinite(result.depth_map)
                food_depth = result.depth_map[finite_food_mask]
                bbox = getattr(food, "bbox", None)
                if (
                    food_depth.size > 0
                    and bbox is not None
                    and len(bbox) == 4
                    and float(bbox[2]) > float(bbox[0])
                ):
                    scale_cm_per_pixel = (
                        float(estimated_diameter)
                        / float(bbox[2] - bbox[0])
                    )
                    food_record["experimental_scale_factor_cm_per_px"] = (
                        scale_cm_per_pixel
                    )

                    min_depth = float(np.min(food_depth))
                    max_depth = float(np.max(food_depth))
                    if max_depth > min_depth:
                        finite_depth_map = np.nan_to_num(
                            result.depth_map,
                            nan=max_depth,
                            posinf=max_depth,
                            neginf=min_depth,
                        )
                        normalized = (max_depth - finite_depth_map) / (
                            max_depth - min_depth
                        )
                        normalized = np.clip(normalized, 0, 1)
                        estimated_height = normalized * float(estimated_thickness)
                        food_record["experimental_volume_cm3"] = float(
                            np.sum(estimated_height[finite_food_mask])
                            * scale_cm_per_pixel**2
                        )

            if include_arrays:
                mask_array = _downsample_u8(
                    np.asarray(mask, dtype=np.uint8),
                    MAX_ARRAY_SIDE,
                )
                food_record["mask_array_b64"] = _encode_npz(mask_array)

            foods.append(food_record)

        depth_values = np.nan_to_num(
            result.depth_map,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        combined_mask = np.zeros_like(depth_values, dtype=bool)
        for mask in result.masks:
            if mask.shape == depth_values.shape:
                combined_mask |= mask

        food_depths = depth_values[combined_mask]
        if len(food_depths) > 100:
            p1, p99 = np.percentile(food_depths, [1, 99])
        else:
            valid_depths = depth_values[depth_values > 0]
            p1, p99 = (
                np.percentile(valid_depths, [1, 99])
                if len(valid_depths) > 0
                else (0, 1)
            )

        display_depth = np.clip(depth_values, p1, p99)
        depth_normalized = cv2.normalize(
            display_depth,
            None,
            0,
            255,
            cv2.NORM_MINMAX,
        )
        depth_uint8 = depth_normalized.astype(np.uint8)
        depth_uint8 = cv2.bilateralFilter(depth_uint8, 9, 75, 75)
        depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)
        depth_color[~combined_mask] = (
            depth_color[~combined_mask] * 0.3
        ).astype(np.uint8)

        success, depth_buffer = cv2.imencode(".png", depth_color)
        if not success:
            raise RuntimeError("Unable to encode depth visualization.")
        depth_b64 = base64.b64encode(depth_buffer).decode("ascii")

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
            depth_array = _downsample_f32(
                np.asarray(result.depth_map, dtype=np.float32),
                MAX_ARRAY_SIDE,
            )
            response["depth_map_b64"] = _encode_npz(depth_array)

        return JSONResponse(content=response)
    finally:
        if image_path and os.path.exists(image_path):
            os.remove(image_path)
