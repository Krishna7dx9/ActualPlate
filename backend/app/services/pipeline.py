"""Application orchestration for image analysis and nutrition enrichment."""

from __future__ import annotations

import base64
import json
import logging
import os
import tempfile
from collections.abc import Iterator
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


def _detect_geometry(image_path: str, detected_foods: list[dict[str, Any]]) -> dict:
    scale_map = {
        food["search_query"].strip().lower(): {
            "thickness": food.get("estimated_thickness_cm"),
            "diameter": food.get("estimated_diameter_cm"),
        }
        for food in detected_foods
        if food.get("search_query")
    }
    detector = GroundingDINODetector()
    return detector.detect(
        image_path=image_path,
        labels=[
            item["search_query"]
            for item in detected_foods
            if item.get("search_query")
        ],
        scale_map=scale_map,
    )


def _merge_volumes(foods_data: list[dict[str, Any]]) -> dict[str, float]:
    merged: dict[str, float] = {}
    for item in foods_data:
        label = str(item.get("label", "")).strip().lower()
        volume = item.get("volume_cm3")
        if label and isinstance(volume, (int, float)) and volume > 0:
            merged[label] = merged.get(label, 0.0) + float(volume)
    return merged


def _lookup_nutrition(food_name: str, mass_grams: float) -> tuple[dict | None, str | None]:
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
        if not query or not isinstance(density, (int, float)) or density <= 0:
            continue

        volume = volumes_by_label.get(str(query).strip().lower())
        if volume is None:
            continue

        attempted += 1
        nutrition, failure_code = await run_in_threadpool(
            _lookup_nutrition,
            str(query),
            volume * float(density),
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

        mask_bytes = base64.b64decode(mask_b64)
        mask = cv2.imdecode(
            np.frombuffer(mask_bytes, np.uint8),
            cv2.IMREAD_GRAYSCALE,
        )
        if mask is None:
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


def _clean_geometry(foods_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned = []
    for item in foods_data:
        clean_item = item.copy()
        clean_item.pop("mask_b64", None)
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
            "geometry": _clean_geometry(foods_data),
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

        yield f"data: {json.dumps({'step': 'food_recognition', 'status': 'running'})}\n\n"
        detected_foods = await run_in_threadpool(detect_food, image)
        yield f"data: {json.dumps({'step': 'food_recognition', 'status': 'done', 'data': detected_foods})}\n\n"

        if not detected_foods:
            yield f"data: {json.dumps({'step': 'complete', 'status': 'done', 'nutrition_status': 'not_available'})}\n\n"
            return

        yield f"data: {json.dumps({'step': 'geometry', 'status': 'running'})}\n\n"
        detections = await run_in_threadpool(
            _detect_geometry,
            image_path,
            detected_foods,
        )
        foods_data = detections.get("foods", [])

        detection_image, segmentation_image, depth_image = await run_in_threadpool(
            _render_visualizations,
            image_path,
            foods_data,
            detections.get("depth_b64", ""),
        )
        yield f"data: {json.dumps({'step': 'detection', 'status': 'done', 'image': detection_image})}\n\n"
        yield f"data: {json.dumps({'step': 'segmentation', 'status': 'done', 'image': segmentation_image})}\n\n"
        yield f"data: {json.dumps({'step': 'depth', 'status': 'done', 'image': depth_image})}\n\n"

        yield f"data: {json.dumps({'step': 'nutrition', 'status': 'running'})}\n\n"
        nutrition_results, nutrition_errors, nutrition_status = (
            await _enrich_nutrition(detected_foods, _merge_volumes(foods_data))
        )

        final_payload = {
            "foods": nutrition_results,
            "geometry": _clean_geometry(foods_data),
            "vlm_output": detected_foods,
            "total_detected": len(foods_data),
            "total_nutrition": aggregate_nutrition(nutrition_results),
            "nutrition_status": nutrition_status,
            "nutrition_errors": nutrition_errors,
        }
        yield f"data: {json.dumps({'step': 'complete', 'status': 'done', 'final': final_payload})}\n\n"
    except Exception:
        logger.exception("Image processing stream failed")
        yield f"data: {json.dumps({'step': 'error', 'status': 'failed', 'message': 'Image processing failed'})}\n\n"
    finally:
        if image_path and os.path.exists(image_path):
            os.remove(image_path)
