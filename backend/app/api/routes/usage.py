"""Tenant-scoped usage and plan entitlement endpoint."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.app.auth.dependencies import CurrentPrincipal, require_scope
from backend.app.database import get_db
from backend.app.services.usage_metering import OrganizationUnavailable, current_usage

router = APIRouter()


@router.get("/v1/usage", tags=["usage"])
def get_usage(
    principal: CurrentPrincipal = Depends(require_scope("usage:read")),
    db: Session = Depends(get_db),
):
    try:
        return current_usage(db, principal.organization_id)
    except OrganizationUnavailable as exc:
        raise HTTPException(status_code=403, detail="Organization usage is unavailable.") from exc
