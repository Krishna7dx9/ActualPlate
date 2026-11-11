import pytest

from backend.app.middleware.request_size_limit import RequestSizeLimitMiddleware


@pytest.mark.asyncio
async def test_rejects_oversized_content_length_before_application_runs():
    application_called = False

    async def application(scope, receive, send):
        nonlocal application_called
        application_called = True

    middleware = RequestSizeLimitMiddleware(application, max_bytes=8)
    sent = []

    async def receive():
        raise AssertionError("The request body should not be read.")

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/upload-image",
        "headers": [(b"content-length", b"9")],
    }

    await middleware(scope, receive, send)

    assert application_called is False
    assert sent[0]["status"] == 413


@pytest.mark.asyncio
async def test_rejects_oversized_chunked_body_without_content_length():
    async def application(scope, receive, send):
        await receive()
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    middleware = RequestSizeLimitMiddleware(application, max_bytes=8)
    sent = []
    chunks = [
        {
            "type": "http.request",
            "body": b"123456789",
            "more_body": False,
        }
    ]

    async def receive():
        return chunks.pop(0)

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/upload-image",
        "headers": [],
    }

    await middleware(scope, receive, send)

    assert sent[0]["status"] == 413


@pytest.mark.asyncio
async def test_allows_request_within_configured_limit():
    async def application(scope, receive, send):
        message = await receive()
        assert message["body"] == b"small"
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    middleware = RequestSizeLimitMiddleware(application, max_bytes=8)
    sent = []
    chunks = [{"type": "http.request", "body": b"small", "more_body": False}]

    async def receive():
        return chunks.pop(0)

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/upload-image",
        "headers": [],
    }

    await middleware(scope, receive, send)

    assert sent[0]["status"] == 200
