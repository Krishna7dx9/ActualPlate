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
"""

from __future__ import annotations

import os
import shutil
import tempfile
import uuid

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from vision_service.inference.inference import detect_and_measure
from vision_service.inference.models import models_loaded
from vision_service.inference.volume import estimate_volume_cm3


VISION_API_TOKEN = os.environ.get("VISION_API_TOKEN")

app = FastAPI(
    title="ActualPlate Vision API",
    version="1.0.0",
)


def _check_auth(x_api_token: str | None) -> None:
    """
    Enforce X-API-Token when VISION_API_TOKEN is configured.

    No-op when the env var is unset (dev mode).
    """
    if VISION_API_TOKEN is None:
        return
    if x_api_token != VISION_API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid API token.")


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
        # Save uploaded image to a temp file. tempfile chooses a
        # platform-appropriate directory; we only need to keep the
        # handle alive long enough to read it.
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".jpg",
        ) as tmp:
            shutil.copyfileobj(file.file, tmp)
            image_path = tmp.name

        result = detect_and_measure(
            image_path=image_path,
            labels=food_labels,
        )

        # Compute volume per detection. The volume engine raises
        # ValueError for data-quality problems (not enough surface,
        # plane not planar, etc.). Those are expected and are reported
        # per food. Any other exception is a bug and propagates.
        foods = []

        for food, mask in zip(result.foods, result.masks):

            food_record = {
                "label": food.label,
                "score": food.score,
                "mask_pixels": food.mask_pixels,
                "depth_m": food.depth_m,
            }

            try:
                volume_result = estimate_volume_cm3(
                    depth_map=result.depth_map,
                    mask=mask,
                    focal_length_px=result.focal_length_px,
                )
                food_record.update(volume_result)
                food_record["volume_status"] = "valid"

            except ValueError as exc:
                food_record["volume_cm3"] = None
                food_record["volume_status"] = "unreliable"
                food_record["volume_error"] = str(exc)

            foods.append(food_record)

        response = {
            "foods": foods,
            "image_width": result.image_width,
            "image_height": result.image_height,
            "focal_length_px": result.focal_length_px,
            "depth_unit": result.depth_unit,
            "request_id": request_id,
        }

        return JSONResponse(content=response)

    finally:
        if image_path and os.path.exists(image_path):
            os.remove(image_path)