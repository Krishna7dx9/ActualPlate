"""Bounded validation and canonical re-encoding of uploaded image payloads."""
from __future__ import annotations

from io import BytesIO
from typing import BinaryIO

from fastapi import HTTPException, UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool


SUPPORTED_IMAGE_FORMATS = frozenset({"JPEG", "PNG", "WEBP"})


def _decode_and_reencode(payload: bytes, max_pixels: int) -> tuple[bytes, str]:
    try:
        with Image.open(BytesIO(payload)) as image:
            image_format = image.format
            if image_format not in SUPPORTED_IMAGE_FORMATS:
                raise HTTPException(status_code=415, detail="Unsupported image format.")
            if getattr(image, "is_animated", False):
                raise HTTPException(status_code=415, detail="Animated images are not supported.")
            width, height = image.size
            if width <= 0 or height <= 0 or width * height > max_pixels:
                raise HTTPException(
                    status_code=413,
                    detail="Image dimensions exceed the configured pixel limit.",
                )
            image.verify()

        # Decode again, apply orientation to pixels, and save without EXIF, XMP,
        # comments, ICC payloads, or trailing bytes from the original upload.
        with Image.open(BytesIO(payload)) as image:
            image.load()
            normalized = ImageOps.exif_transpose(image)
            output = BytesIO()
            has_alpha = "A" in normalized.getbands() or "transparency" in normalized.info
            if image_format == "JPEG":
                clean_image = normalized.convert("RGB")
                clean_image.info.clear()
                clean_image.save(output, format="JPEG", quality=95, optimize=True)
            elif image_format == "PNG":
                clean_image = normalized.convert("RGBA" if has_alpha else "RGB")
                clean_image.info.clear()
                clean_image.save(output, format="PNG", optimize=True)
            else:
                clean_image = normalized.convert("RGBA" if has_alpha else "RGB")
                clean_image.info.clear()
                clean_image.save(output, format="WEBP", quality=95, method=4)
            return output.getvalue(), image_format
    except HTTPException:
        raise
    except (
        UnidentifiedImageError,
        OSError,
        SyntaxError,
        ValueError,
        Image.DecompressionBombError,
    ) as exc:
        raise HTTPException(
            status_code=400,
            detail="Uploaded content is not a valid supported image.",
        ) from exc


def _replace_payload(file_obj: BinaryIO, payload: bytes) -> None:
    file_obj.seek(0)
    file_obj.truncate(0)
    file_obj.write(payload)
    file_obj.seek(0)


async def validate_image_upload(
    file: UploadFile,
    *,
    max_bytes: int,
    max_pixels: int,
) -> None:
    """Bound, decode, sanitize, and rewind the image for downstream inference."""
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

    sanitized, _image_format = await run_in_threadpool(
        _decode_and_reencode,
        payload,
        max_pixels,
    )
    if len(sanitized) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail="Sanitized image exceeds the configured size limit.",
        )

    await run_in_threadpool(_replace_payload, file.file, sanitized)
    file.size = len(sanitized)
    await file.seek(0)
