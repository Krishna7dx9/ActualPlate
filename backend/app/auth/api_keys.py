"""Organization-scoped API-key creation, listing, and revocation."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.auth.dependencies import CurrentPrincipal, require_roles, require_scope
from backend.app.auth.schemas import APIKeyCreateRequest, APIKeyCreatedResponse, APIKeySummary
from backend.app.auth.security import AuthenticationConfigurationError, generate_api_key, hash_api_key
from backend.app.config import settings
from backend.app.database import get_db
from backend.app.models import APIKey, AuditLog, new_id

router = APIRouter(prefix="/v1/api-keys", tags=["API keys"])


@router.post("", response_model=APIKeyCreatedResponse, status_code=status.HTTP_201_CREATED)
def create_api_key(
    payload: APIKeyCreateRequest,
    principal: CurrentPrincipal = Depends(require_roles("owner", "admin")),
    _scope: CurrentPrincipal = Depends(require_scope("keys:manage")),
    db: Session = Depends(get_db),
) -> APIKeyCreatedResponse:
    try:
        raw_key = generate_api_key(settings.environment)
        key_hash = hash_api_key(raw_key)
    except AuthenticationConfigurationError as exc:
        raise HTTPException(status_code=503, detail="API-key creation is not configured.") from exc

    now = datetime.now(timezone.utc)
    record = APIKey(
        id=new_id(),
        organization_id=principal.organization_id,
        created_by_user_id=principal.user_id,
        name=payload.name,
        key_prefix=raw_key[:16],
        key_hash=key_hash,
        scopes=payload.scopes,
        created_at=now,
        expires_at=now + timedelta(days=payload.expires_in_days),
    )
    db.add(record)
    db.add(AuditLog(
        organization_id=principal.organization_id,
        actor_user_id=principal.user_id,
        action="api_key.created",
        target_type="api_key",
        target_id=record.id,
        attributes={"key_prefix": record.key_prefix, "scopes": payload.scopes},
    ))
    db.commit()
    db.refresh(record)
    return APIKeyCreatedResponse(
        id=record.id,
        name=record.name,
        key=raw_key,
        key_prefix=record.key_prefix,
        scopes=record.scopes,
        created_at=record.created_at,
        expires_at=record.expires_at,
    )


@router.get("", response_model=list[APIKeySummary])
def list_api_keys(
    principal: CurrentPrincipal = Depends(require_roles("owner", "admin")),
    _scope: CurrentPrincipal = Depends(require_scope("keys:manage")),
    db: Session = Depends(get_db),
) -> list[APIKeySummary]:
    records = db.scalars(
        select(APIKey)
        .where(APIKey.organization_id == principal.organization_id)
        .order_by(APIKey.created_at.desc())
    ).all()
    return [
        APIKeySummary(
            id=item.id,
            name=item.name,
            key_prefix=item.key_prefix,
            scopes=item.scopes or [],
            created_at=item.created_at,
            expires_at=item.expires_at,
            revoked_at=item.revoked_at,
            last_used_at=item.last_used_at,
        )
        for item in records
    ]


@router.delete("/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
def revoke_api_key(
    key_id: str,
    principal: CurrentPrincipal = Depends(require_roles("owner", "admin")),
    _scope: CurrentPrincipal = Depends(require_scope("keys:manage")),
    db: Session = Depends(get_db),
) -> Response:
    record = db.scalar(
        select(APIKey).where(
            APIKey.id == key_id,
            APIKey.organization_id == principal.organization_id,
        )
    )
    if record is None:
        raise HTTPException(status_code=404, detail="API key not found.")
    if record.revoked_at is None:
        record.revoked_at = datetime.now(timezone.utc)
        db.add(AuditLog(
            organization_id=principal.organization_id,
            actor_user_id=principal.user_id,
            action="api_key.revoked",
            target_type="api_key",
            target_id=record.id,
            attributes={"key_prefix": record.key_prefix},
        ))
        db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
