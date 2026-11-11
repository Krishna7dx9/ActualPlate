from io import BytesIO

import pytest
from fastapi import HTTPException, UploadFile
from PIL import Image

from backend.app.services.upload_validation import validate_image_upload


def _png_bytes(width: int = 8, height: int = 8) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (width, height)).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.mark.asyncio
async def test_accepts_supported_decodable_image_and_rewinds_upload():
    payload = _png_bytes()
    upload = UploadFile(filename="image.png", file=BytesIO(payload))

    await validate_image_upload(upload, max_bytes=1024, max_pixels=1024)

    assert await upload.read() == payload


@pytest.mark.asyncio
async def test_rejects_payload_over_configured_byte_limit():
    upload = UploadFile(filename="image.png", file=BytesIO(_png_bytes()))

    with pytest.raises(HTTPException) as error:
        await validate_image_upload(upload, max_bytes=4, max_pixels=1024)

    assert error.value.status_code == 413


@pytest.mark.asyncio
async def test_rejects_image_over_configured_pixel_limit():
    upload = UploadFile(filename="image.png", file=BytesIO(_png_bytes(20, 20)))

    with pytest.raises(HTTPException) as error:
        await validate_image_upload(upload, max_bytes=1024, max_pixels=100)

    assert error.value.status_code == 413


@pytest.mark.asyncio
async def test_rejects_non_image_payload_even_when_client_claims_image():
    upload = UploadFile(filename="fake.jpg", file=BytesIO(b"not an image"))

    with pytest.raises(HTTPException) as error:
        await validate_image_upload(upload, max_bytes=1024, max_pixels=1024)

    assert error.value.status_code == 400
