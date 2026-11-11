import pytest

from backend.app.config import Settings


def test_cors_origins_are_parsed_from_environment(monkeypatch):
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("MAX_UPLOAD_BYTES", "10485760")
    monkeypatch.setenv("MAX_IMAGE_PIXELS", "40000000")
    monkeypatch.setenv("MAX_REQUEST_BODY_BYTES", "12582912")
    monkeypatch.setenv(
        "CORS_ALLOWED_ORIGINS",
        "https://app.example, https://admin.example/",
    )

    settings = Settings.from_environment()

    assert settings.cors_allowed_origins == (
        "https://app.example",
        "https://admin.example",
    )


def test_empty_cors_configuration_denies_cross_origin_by_default(monkeypatch):
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.delenv("CORS_ALLOWED_ORIGINS", raising=False)

    settings = Settings.from_environment()

    assert settings.cors_allowed_origins == ()


def test_production_rejects_wildcard_cors_origin(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("MAX_UPLOAD_BYTES", "10485760")
    monkeypatch.setenv("MAX_IMAGE_PIXELS", "40000000")
    monkeypatch.setenv("MAX_REQUEST_BODY_BYTES", "12582912")
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "*")

    with pytest.raises(ValueError, match="Wildcard CORS"):
        Settings.from_environment()



def test_request_body_limit_cannot_be_smaller_than_upload_limit(monkeypatch):
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("MAX_UPLOAD_BYTES", "10485760")
    monkeypatch.setenv("MAX_REQUEST_BODY_BYTES", "1048576")
    monkeypatch.setenv("MAX_IMAGE_PIXELS", "40000000")

    with pytest.raises(ValueError, match="MAX_REQUEST_BODY_BYTES"):
        Settings.from_environment()



def test_vision_endpoint_is_required_and_validated(monkeypatch):
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("MAX_UPLOAD_BYTES", "10485760")
    monkeypatch.setenv("MAX_REQUEST_BODY_BYTES", "12582912")
    monkeypatch.setenv("MAX_IMAGE_PIXELS", "40000000")
    monkeypatch.setenv("VISION_INFERENCE_ENDPOINT", "not-a-url")

    with pytest.raises(ValueError, match="VISION_INFERENCE_ENDPOINT"):
        Settings.from_environment()


def test_production_requires_vision_token(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("MAX_UPLOAD_BYTES", "10485760")
    monkeypatch.setenv("MAX_REQUEST_BODY_BYTES", "12582912")
    monkeypatch.setenv("MAX_IMAGE_PIXELS", "40000000")
    monkeypatch.setenv("VISION_INFERENCE_ENDPOINT", "http://localhost:8001")
    monkeypatch.delenv("VISION_API_TOKEN", raising=False)

    with pytest.raises(ValueError, match="VISION_API_TOKEN"):
        Settings.from_environment()

@pytest.mark.parametrize("value", ["prod", "staging", "production-like", ""])
def test_rejects_unknown_app_environment(value, monkeypatch):
    monkeypatch.setenv("APP_ENV", value)
    monkeypatch.setenv("MAX_UPLOAD_BYTES", "10485760")
    monkeypatch.setenv("MAX_REQUEST_BODY_BYTES", "12582912")
    monkeypatch.setenv("MAX_IMAGE_PIXELS", "40000000")
    monkeypatch.setenv("VISION_INFERENCE_ENDPOINT", "http://localhost:8001")

    with pytest.raises(ValueError, match="APP_ENV"):
        Settings.from_environment()


def test_production_requires_shared_redis_rate_limiter(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("MAX_UPLOAD_BYTES", "10485760")
    monkeypatch.setenv("MAX_REQUEST_BODY_BYTES", "12582912")
    monkeypatch.setenv("MAX_IMAGE_PIXELS", "40000000")
    monkeypatch.setenv("VISION_INFERENCE_ENDPOINT", "https://vision.example.com")
    monkeypatch.setenv("VISION_API_TOKEN", "production-test-token")
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "https://app.example.com")
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://user:pass@db.example.com/actualplate")
    monkeypatch.setenv("JWT_SECRET_KEY", "production-test-jwt-secret-at-least-32-bytes")
    monkeypatch.setenv("API_KEY_HASH_SECRET", "production-test-api-key-hash-at-least-32")
    monkeypatch.delenv("REDIS_URL", raising=False)

    with pytest.raises(ValueError, match="REDIS_URL"):
        Settings.from_environment()


def test_rate_limit_setting_is_validated(monkeypatch):
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("MAX_UPLOAD_BYTES", "10485760")
    monkeypatch.setenv("MAX_REQUEST_BODY_BYTES", "12582912")
    monkeypatch.setenv("MAX_IMAGE_PIXELS", "40000000")
    monkeypatch.setenv("VISION_INFERENCE_ENDPOINT", "http://localhost:8001")
    monkeypatch.setenv("RATE_LIMIT_REQUESTS_PER_MINUTE", "0")

    with pytest.raises(ValueError, match="RATE_LIMIT_REQUESTS_PER_MINUTE"):
        Settings.from_environment()
