"""Password, JWT, and API-key primitives for B2B authentication."""
from __future__ import annotations

import hashlib
import hmac
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any

import bcrypt
import jwt

from backend.app.config import settings

JWT_ISSUER = "actualplate-api"
JWT_AUDIENCE = "actualplate-client"
JWT_ALGORITHM = "HS256"


class AuthenticationConfigurationError(RuntimeError):
    """Raised when signing or API-key hashing secrets are not configured."""


def _secret(value: str | None, name: str) -> str:
    if not value or len(value.encode("utf-8")) < 32:
        raise AuthenticationConfigurationError(
            f"{name} must contain at least 32 bytes before authentication can be used."
        )
    return value


def hash_password(password: str) -> str:
    if not isinstance(password, str):
        raise ValueError("Password must be text.")
    encoded = password.encode("utf-8")
    if len(encoded) < 12:
        raise ValueError("Password must contain at least 12 UTF-8 bytes.")
    if len(encoded) > 72:
        raise ValueError("Password must not exceed 72 UTF-8 bytes.")
    return bcrypt.hashpw(encoded, bcrypt.gensalt(rounds=12)).decode("ascii")


def verify_password(password: str, password_hash: str) -> bool:
    if not isinstance(password, str) or not isinstance(password_hash, str):
        return False
    encoded = password.encode("utf-8")
    if not 1 <= len(encoded) <= 72:
        return False
    try:
        return bcrypt.checkpw(encoded, password_hash.encode("ascii"))
    except (ValueError, TypeError):
        return False


def create_access_token(*, user_id: str, organization_id: str) -> tuple[str, int]:
    signing_key = _secret(settings.jwt_secret_key, "JWT_SECRET_KEY")
    now = datetime.now(timezone.utc)
    ttl = settings.access_token_ttl_seconds
    claims: dict[str, Any] = {
        "sub": user_id,
        "org_id": organization_id,
        "iss": JWT_ISSUER,
        "aud": JWT_AUDIENCE,
        "iat": now,
        "nbf": now,
        "exp": now + timedelta(seconds=ttl),
        "jti": secrets.token_urlsafe(18),
        "token_type": "access",
    }
    return jwt.encode(claims, signing_key, algorithm=JWT_ALGORITHM), ttl


def decode_access_token(token: str) -> dict[str, Any]:
    signing_key = _secret(settings.jwt_secret_key, "JWT_SECRET_KEY")
    claims = jwt.decode(
        token,
        signing_key,
        algorithms=[JWT_ALGORITHM],
        issuer=JWT_ISSUER,
        audience=JWT_AUDIENCE,
        options={"require": ["sub", "org_id", "iat", "nbf", "exp", "jti", "token_type"]},
    )
    if claims.get("token_type") != "access":
        raise jwt.InvalidTokenError("Unexpected token type.")
    return claims


def generate_api_key(environment: str) -> str:
    prefix = "ap_live_" if environment == "production" else "ap_test_" if environment == "test" else "ap_dev_"
    return prefix + secrets.token_urlsafe(32)


def hash_api_key(raw_key: str) -> str:
    key = _secret(settings.api_key_hash_secret, "API_KEY_HASH_SECRET")
    return hmac.new(key.encode("utf-8"), raw_key.encode("utf-8"), hashlib.sha256).hexdigest()


def verify_api_key(raw_key: str, stored_hash: str) -> bool:
    try:
        candidate = hash_api_key(raw_key)
    except AuthenticationConfigurationError:
        return False
    return hmac.compare_digest(candidate, stored_hash)
