"""Bounded validation of uploaded image payloads."""

from __future__ import annotations

from io import BytesIO

from fastapi import HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError


SUPPORTED_IMAGE_FORMATS = frozenset({"JPEG", "PNG", "WEBP"})


async def validate_image_upload(
    file: UploadFile,
    *,
    max_bytes: int,
    max_pixels: int,
) -> None:
    """Validate payload size, decoded format, and pixel count.

    The payload is read with a strict upper bound and the upload cursor is
    rewound so downstream processing can consume the same validated bytes.
    """
    if max_bytes <= 0 or max_pixels <= 0:
        raise ValueError("Upload limits must be positive.")

    try:
        payload = await file.read(max_bytes + 1)
    finally:
        await file.seek(0)

    if len(payload) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail="Uploaded file exceeds the configured size limit.",
        )

    try:
        with Image.open(BytesIO(payload)) as image:
            if image.format not in SUPPORTED_IMAGE_FORMATS:
                raise HTTPException(
                    status_code=415,
                    detail="Unsupported image format.",
                )

            width, height = image.size
            if width <= 0 or height <= 0 or width * height > max_pixels:
                raise HTTPException(
                    status_code=413,
                    detail="Image dimensions exceed the configured pixel limit.",
                )
            image.verify()

        # Verify the encoded image can be decoded, not just identified.
        with Image.open(BytesIO(payload)) as image:
            image.load()
    except HTTPException:
        raise
    except (
        UnidentifiedImageError,
        OSError,
        SyntaxError,
        Image.DecompressionBombError,
    ) as exc:
        raise HTTPException(
            status_code=400,
            detail="Uploaded content is not a valid supported image.",
        ) from exc
