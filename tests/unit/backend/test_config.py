import pytest

from backend.app.config import Settings


def test_cors_origins_are_parsed_from_environment(monkeypatch):
    monkeypatch.setenv("APP_ENV", "development")
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
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "*")

    with pytest.raises(ValueError, match="Wildcard CORS"):
        Settings.from_environment()
