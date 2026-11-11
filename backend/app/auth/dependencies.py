"""Resolve JWT and API-key principals and enforce organization membership."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.auth.security import (
    AuthenticationConfigurationError,
    decode_access_token,
    hash_api_key,
)
from backend.app.database import get_db
from backend.app.models import APIKey, Membership, Organization, User

bearer_scheme = HTTPBearer(auto_error=False)
api_key_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)


@dataclass(frozen=True)
class CurrentPrincipal:
    user_id: str
    organization_id: str
    role: str
    scopes: frozenset[str]
    auth_method: str
    api_key_id: str | None = None


def _expired(value: datetime | None) -> bool:
    if value is None:
        return False
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value <= datetime.now(timezone.utc)


def _unauthorized() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Valid authentication credentials are required.",
        headers={"WWW-Authenticate": "Bearer"},
    )


def get_current_principal(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    raw_api_key: str | None = Depends(api_key_scheme),
    db: Session = Depends(get_db),
) -> CurrentPrincipal:
    if raw_api_key:
        try:
            key_hash = hash_api_key(raw_api_key)
        except AuthenticationConfigurationError as exc:
            raise HTTPException(status_code=503, detail="API-key authentication is not configured.") from exc
        key = db.scalar(
            select(APIKey).where(
                APIKey.key_hash == key_hash,
                APIKey.revoked_at.is_(None),
            )
        )
        if key is None or _expired(key.expires_at):
            raise _unauthorized()

        user = db.get(User, key.created_by_user_id)
        organization = db.get(Organization, key.organization_id)
        membership = db.scalar(
            select(Membership).where(
                Membership.user_id == key.created_by_user_id,
                Membership.organization_id == key.organization_id,
                Membership.is_active.is_(True),
            )
        )
        if (
            user is None or not user.is_active
            or organization is None or not organization.is_active
            or membership is None
        ):
            raise _unauthorized()

        key.last_used_at = datetime.now(timezone.utc)
        db.commit()
        return CurrentPrincipal(
            user_id=user.id,
            organization_id=organization.id,
            role=membership.role,
            scopes=frozenset(key.scopes or []),
            auth_method="api_key",
            api_key_id=key.id,
        )

    if credentials is None or credentials.scheme.lower() != "bearer":
        raise _unauthorized()
    try:
        claims = decode_access_token(credentials.credentials)
    except AuthenticationConfigurationError as exc:
        raise HTTPException(status_code=503, detail="JWT authentication is not configured.") from exc
    except jwt.PyJWTError as exc:
        raise _unauthorized() from exc

    user_id = claims.get("sub")
    organization_id = claims.get("org_id")
    if not isinstance(user_id, str) or not isinstance(organization_id, str):
        raise _unauthorized()

    user = db.get(User, user_id)
    organization = db.get(Organization, organization_id)
    membership = db.scalar(
        select(Membership).where(
            Membership.user_id == user_id,
            Membership.organization_id == organization_id,
            Membership.is_active.is_(True),
        )
    )
    if (
        user is None or not user.is_active
        or organization is None or not organization.is_active
        or membership is None
    ):
        raise _unauthorized()

    if settings_environment_is_production() and user.email_verified_at is None:
        raise HTTPException(status_code=403, detail="Account email must be verified.")
    return CurrentPrincipal(
        user_id=user.id,
        organization_id=organization.id,
        role=membership.role,
        scopes=frozenset({"scans:write", "usage:read", "keys:manage"}),
        auth_method="jwt",
    )


def settings_environment_is_production() -> bool:
    from backend.app.config import settings
    return settings.environment == "production"


def require_roles(*allowed_roles: str) -> Callable:
    def dependency(principal: CurrentPrincipal = Depends(get_current_principal)) -> CurrentPrincipal:
        if principal.role not in allowed_roles:
            raise HTTPException(status_code=403, detail="Your organization role does not allow this action.")
        return principal
    return dependency


def require_scope(scope: str) -> Callable:
    def dependency(principal: CurrentPrincipal = Depends(get_current_principal)) -> CurrentPrincipal:
        if scope not in principal.scopes:
            raise HTTPException(status_code=403, detail="The credential does not grant the required scope.")
        return principal
    return dependency
