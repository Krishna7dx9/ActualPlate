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

    sanitized = await upload.read()
    assert sanitized
    with Image.open(BytesIO(sanitized)) as image:
        assert image.format == "PNG"
        assert image.size == (8, 8)
    assert upload.size == len(sanitized)


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


@pytest.mark.asyncio
async def test_reencodes_jpeg_strips_exif_and_applies_orientation():
    buffer = BytesIO()
    image = Image.new("RGB", (8, 4), color=(90, 120, 150))
    exif = Image.Exif()
    exif[274] = 6
    exif[270] = "private metadata marker"
    image.save(buffer, format="JPEG", exif=exif)
    original = buffer.getvalue()
    upload = UploadFile(filename="private.jpg", file=BytesIO(original))

    await validate_image_upload(upload, max_bytes=1024 * 10, max_pixels=1024)

    sanitized = await upload.read()
    assert b"private metadata marker" not in sanitized
    with Image.open(BytesIO(sanitized)) as result:
        assert result.format == "JPEG"
        assert result.getexif().get(274) is None
        assert result.getexif().get(270) is None
        assert result.size == (4, 8)


@pytest.mark.asyncio
async def test_rejects_animated_webp():
    frames = [Image.new("RGB", (4, 4), color=(index * 20, 20, 20)) for index in range(2)]
    buffer = BytesIO()
    frames[0].save(buffer, format="WEBP", save_all=True, append_images=frames[1:], duration=100, loop=0)
    upload = UploadFile(filename="animated.webp", file=BytesIO(buffer.getvalue()))

    with pytest.raises(HTTPException) as error:
        await validate_image_upload(upload, max_bytes=1024 * 10, max_pixels=1024)
    assert error.value.status_code == 415
