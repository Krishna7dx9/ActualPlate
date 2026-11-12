"""Application orchestration for image analysis and nutrition enrichment."""

from __future__ import annotations

import base64
import json
import logging
import math
import os
import tempfile
from typing import Any

import cv2
import numpy as np
import requests
from fastapi import UploadFile
from starlette.concurrency import run_in_threadpool

from backend.app.nutrition.nutrition_aggregator import aggregate_nutrition
from backend.app.nutrition.nutrition_formatter import format_nutrition_response
from backend.app.nutrition.providers.fatsecret_client import search_food
from vision_service.detection.grounding_dino_detector import GroundingDINODetector
from vision_service.recognition.food_recognition import detect_food


logger = logging.getLogger(__name__)


def _write_temporary_image(image: bytes) -> str:
    """Persist validated image bytes for detector clients that require a path."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".image") as temporary:
        temporary.write(image)
        return temporary.name


def _build_scale_map(detected_foods: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Pass only positive finite visual estimates to the worker as unvalidated priors."""
    scale_map: dict[str, dict[str, float]] = {}
    for food in detected_foods:
        label = food.get("search_query")
        if not isinstance(label, str) or not label.strip():
            continue
        dimensions: dict[str, float] = {}
        for source_key, target_key in (
            ("estimated_thickness_cm", "thickness"),
            ("estimated_diameter_cm", "diameter"),
        ):
            value = food.get(source_key)
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                and value > 0
            ):
                dimensions[target_key] = float(value)
        scale_map[label.strip().lower()] = dimensions
    return scale_map


def _detect_geometry(image_path: str, detected_foods: list[dict[str, Any]]) -> dict:
    detector = GroundingDINODetector()
    return detector.detect(
        image_path=image_path,
        labels=[
            item["search_query"]
            for item in detected_foods
            if item.get("search_query")
        ],
        scale_map=_build_scale_map(detected_foods),
    )


def _merge_volumes(foods_data: list[dict[str, Any]]) -> dict[str, float]:
    """Merge only finite, positive authoritative metric volumes by label."""
    merged: dict[str, float] = {}
    for item in foods_data:
        label = str(item.get("label", "")).strip().lower()
        volume = item.get("volume_cm3")
        if (
            label
            and not isinstance(volume, bool)
            and isinstance(volume, (int, float))
            and math.isfinite(float(volume))
            and volume > 0
        ):
            merged[label] = merged.get(label, 0.0) + float(volume)
    return merged


def _lookup_nutrition(
    food_name: str,
    mass_grams: float,
) -> tuple[dict | None, str | None]:
    """Return nutrition data or a stable failure code; never silently discard errors."""
    try:
        provider_data = search_food(food_name)
        if not isinstance(provider_data, dict):
            return None, "invalid_provider_response"
        if provider_data.get("error"):
            return None, "provider_rejected_request"
        return format_nutrition_response(provider_data, mass_grams), None
    except requests.RequestException:
        logger.warning(
            "Nutrition provider request failed",
            extra={"food_query": food_name},
            exc_info=True,
        )
        return None, "provider_unavailable"
    except (KeyError, TypeError, ValueError):
        logger.exception(
            "Nutrition provider returned an invalid response",
            extra={"food_query": food_name},
        )
        return None, "invalid_provider_response"
    except Exception:
        logger.exception(
            "Unexpected nutrition enrichment failure",
            extra={"food_query": food_name},
        )
        return None, "nutrition_processing_error"


async def _enrich_nutrition(
    detected_foods: list[dict[str, Any]],
    volumes_by_label: dict[str, float],
) -> tuple[list[dict], list[dict[str, str]], str]:
    nutrition_results: list[dict] = []
    nutrition_errors: list[dict[str, str]] = []
    attempted = 0

    for item in detected_foods:
        query = item.get("search_query")
        density = item.get("density_g_cm3")
        if (
            not query
            or isinstance(density, bool)
            or not isinstance(density, (int, float))
            or not math.isfinite(float(density))
            or density <= 0
        ):
            continue

        volume = volumes_by_label.get(str(query).strip().lower())
        if (
            isinstance(volume, bool)
            or not isinstance(volume, (int, float))
            or not math.isfinite(float(volume))
            or volume <= 0
        ):
            continue

        mass_grams = float(volume) * float(density)
        if not math.isfinite(mass_grams) or mass_grams <= 0:
            continue

        attempted += 1
        nutrition, failure_code = await run_in_threadpool(
            _lookup_nutrition,
            str(query),
            mass_grams,
        )
        if nutrition is not None:
            nutrition_results.append(nutrition)
        elif failure_code is not None:
            nutrition_errors.append(
                {"food": str(query), "code": failure_code}
            )

    if attempted == 0:
        status = "not_available"
    elif nutrition_errors and nutrition_results:
        status = "partial"
    elif nutrition_errors:
        status = "unavailable"
    else:
        status = "complete"

    return nutrition_results, nutrition_errors, status


def _encode_jpeg(image: np.ndarray) -> str:
    success, buffer = cv2.imencode(".jpg", image)
    if not success:
        raise RuntimeError("Unable to encode visualization image.")
    return base64.b64encode(buffer).decode("ascii")


def _render_visualizations(
    image_path: str,
    foods_data: list[dict[str, Any]],
    depth_b64: str,
) -> tuple[str, str, str]:
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Validated upload could not be decoded by OpenCV.")

    colors = (
        (0, 255, 0),
        (0, 165, 255),
        (255, 0, 0),
        (255, 255, 0),
    )

    detection_image = image.copy()
    for index, item in enumerate(foods_data):
        bbox = item.get("bbox")
        if bbox and len(bbox) == 4:
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(
                detection_image,
                (x1, y1),
                (x2, y2),
                colors[index % len(colors)],
                2,
            )

    segmentation_image = image.copy()
    for index, item in enumerate(foods_data):
        mask_b64 = item.get("mask_b64")
        if not mask_b64:
            continue

        try:
            mask_bytes = base64.b64decode(mask_b64, validate=True)
            mask = cv2.imdecode(
                np.frombuffer(mask_bytes, np.uint8),
                cv2.IMREAD_GRAYSCALE,
            )
        except (ValueError, TypeError):
            logger.warning("Skipping malformed segmentation mask for item %s", index)
            continue
        if mask is None:
            logger.warning("Skipping undecodable segmentation mask for item %s", index)
            continue
        if mask.shape != image.shape[:2]:
            mask = cv2.resize(
                mask,
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        overlay = np.zeros_like(segmentation_image)
        overlay[mask > 127] = colors[index % len(colors)]
        segmentation_image = cv2.addWeighted(
            segmentation_image,
            1.0,
            overlay,
            0.35,
            0,
        )

    return (
        _encode_jpeg(detection_image),
        _encode_jpeg(segmentation_image),
        depth_b64,
    )


def _clean_geometry_payload(foods_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned = []
    for item in foods_data:
        clean_item = item.copy()
        clean_item.pop("mask_b64", None)
        clean_item.pop("mask_array_b64", None)
        cleaned.append(clean_item)
    return cleaned


async def process_image(file: UploadFile) -> dict:
    image = await file.read()
    image_path: str | None = None

    try:
        image_path = _write_temporary_image(image)
        detected_foods = await run_in_threadpool(detect_food, image)
        if not detected_foods:
            return {
                "foods": [],
                "geometry": [],
                "vlm_output": [],
                "total_detected": 0,
                "total_nutrition": aggregate_nutrition([]),
                "nutrition_status": "not_available",
                "nutrition_errors": [],
            }

        detections = await run_in_threadpool(
            _detect_geometry,
            image_path,
            detected_foods,
        )
        foods_data = detections.get("foods", [])
        nutrition_results, nutrition_errors, nutrition_status = (
            await _enrich_nutrition(detected_foods, _merge_volumes(foods_data))
        )

        return {
            "foods": nutrition_results,
            "geometry": foods_data,
            "vlm_output": detected_foods,
            "total_detected": len(foods_data),
            "total_nutrition": aggregate_nutrition(nutrition_results),
            "nutrition_status": nutrition_status,
            "nutrition_errors": nutrition_errors,
        }
    finally:
        if image_path and os.path.exists(image_path):
            os.remove(image_path)


async def process_image_stream(file: UploadFile):
    image = await file.read()
    image_path: str | None = None

    try:
        image_path = _write_temporary_image(image)

        yield f"data: {json.dumps({'step': 'gemma', 'status': 'running', 'message': 'Food recognition is running'})}\n\n"
        detected_foods = await run_in_threadpool(detect_food, image)
        yield f"data: {json.dumps({'step': 'gemma', 'status': 'done', 'message': 'Food recognition completed', 'data': detected_foods})}\n\n"

        if not detected_foods:
            yield f"data: {json.dumps({'step': 'complete', 'status': 'done', 'nutrition_status': 'not_available'})}\n\n"
            return

        yield f"data: {json.dumps({'step': 'dino', 'status': 'running', 'message': 'Detection and geometry are running'})}\n\n"
        detections = await run_in_threadpool(
            _detect_geometry,
            image_path,
            detected_foods,
        )
        foods_data = detections.get("foods", [])

        try:
            detection_image, segmentation_image, depth_image = await run_in_threadpool(
                _render_visualizations,
                image_path,
                foods_data,
                detections.get("depth_b64", ""),
            )
        except Exception:
            logger.exception("Image visualization rendering failed")
            yield f"data: {json.dumps({'step': 'error', 'status': 'error', 'message': 'Image rendering is temporarily unavailable.'})}\n\n"
            return
        yield f"data: {json.dumps({'step': 'dino', 'status': 'done', 'message': 'Detection completed', 'image': detection_image})}\n\n"
        yield f"data: {json.dumps({'step': 'sam2', 'status': 'done', 'message': 'Segmentation completed', 'image': segmentation_image})}\n\n"
        yield f"data: {json.dumps({'step': 'depth', 'status': 'done', 'message': 'Depth map generated', 'image': depth_image})}\n\n"
        yield f"data: {json.dumps({'step': 'volume', 'status': 'done', 'message': 'Volume processing completed'})}\n\n"

        yield f"data: {json.dumps({'step': 'nutrition', 'status': 'running', 'message': 'Nutrition lookup is running'})}\n\n"
        nutrition_results, nutrition_errors, nutrition_status = (
            await _enrich_nutrition(detected_foods, _merge_volumes(foods_data))
        )

        final_payload = {
            "foods": nutrition_results,
            "geometry": _clean_geometry_payload(foods_data),
            "vlm_output": detected_foods,
            "total_detected": len(foods_data),
            "total_nutrition": aggregate_nutrition(nutrition_results),
            "nutrition_status": nutrition_status,
            "nutrition_errors": nutrition_errors,
        }
        yield f"data: {json.dumps({'step': 'nutrition', 'status': 'done', 'message': 'Complete', 'final': final_payload})}\n\n"
    except Exception:
        logger.exception("Image processing stream failed")
        yield f"data: {json.dumps({'step': 'error', 'status': 'failed', 'message': 'Image processing failed'})}\n\n"
    finally:
        if image_path and os.path.exists(image_path):
            os.remove(image_path)
