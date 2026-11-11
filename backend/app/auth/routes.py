"""Signup, login, and authenticated identity endpoints."""
from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone

import jwt
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.app.auth.dependencies import CurrentPrincipal, get_current_principal
from backend.app.auth.schemas import (
    AuthResponse,
    LoginRequest,
    OrganizationSummary,
    SignupRequest,
    UserSummary,
)
from backend.app.auth.security import (
    AuthenticationConfigurationError,
    create_access_token,
    hash_password,
    verify_password,
)
from backend.app.config import settings
from backend.app.database import get_db
from backend.app.models import AuditLog, Membership, Organization, Plan, User, new_id

router = APIRouter(prefix="/v1/auth", tags=["authentication"])


def _slug(name: str) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", name.casefold()).strip("-")[:150] or "organization"
    return f"{base}-{uuid.uuid4().hex[:8]}"


def _auth_response(db: Session, user: User, organization: Organization, role: str) -> AuthResponse:
    try:
        token, ttl = create_access_token(user_id=user.id, organization_id=organization.id)
    except AuthenticationConfigurationError as exc:
        raise HTTPException(status_code=503, detail="Authentication is not configured.") from exc
    plan = db.get(Plan, organization.plan_id)
    if plan is None:
        raise HTTPException(status_code=503, detail="Organization plan is not configured.")
    return AuthResponse(
        access_token=token,
        expires_in=ttl,
        user=UserSummary(id=user.id, email=user.email),
        organization=OrganizationSummary(
            id=organization.id,
            name=organization.name,
            slug=organization.slug,
            plan_code=plan.code,
            role=role,
        ),
    )


@router.post("/signup", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
def signup(payload: SignupRequest, db: Session = Depends(get_db)) -> AuthResponse:
    if settings.environment == "production":
        raise HTTPException(
            status_code=503,
            detail="Public signup is disabled until email verification and signup abuse controls are configured.",
        )

    email = str(payload.email).strip().lower()
    if db.scalar(select(User.id).where(User.email == email)) is not None:
        raise HTTPException(status_code=409, detail="An account with this email already exists.")

    free_plan = db.scalar(select(Plan).where(Plan.code == "free", Plan.is_active.is_(True)))
    if free_plan is None:
        raise HTTPException(status_code=503, detail="The default organization plan is not configured.")

    user = User(id=new_id(), email=email, password_hash=hash_password(payload.password))
    organization = Organization(
        id=new_id(),
        name=payload.organization_name,
        slug=_slug(payload.organization_name),
        plan_id=free_plan.id,
    )
    membership = Membership(
        id=new_id(),
        user_id=user.id,
        organization_id=organization.id,
        role="owner",
    )
    audit = AuditLog(
        organization_id=organization.id,
        actor_user_id=user.id,
        action="account.signup",
        target_type="organization",
        target_id=organization.id,
        attributes={"plan": "free"},
    )
    try:
        db.add_all([user, organization, membership, audit])
        db.flush()
        response = _auth_response(db, user, organization, "owner")
        db.commit()
        return response
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="Account could not be created with the supplied details.") from exc
    except Exception:
        db.rollback()
        raise


@router.post("/login", response_model=AuthResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)) -> AuthResponse:
    email = str(payload.email).strip().lower()
    user = db.scalar(select(User).where(User.email == email))
    if user is None or not user.is_active or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    if settings.environment == "production" and user.email_verified_at is None:
        raise HTTPException(status_code=403, detail="Account email must be verified.")

    membership = db.scalar(
        select(Membership).where(Membership.user_id == user.id, Membership.is_active.is_(True)).order_by(Membership.created_at)
    )
    if membership is None:
        raise HTTPException(status_code=403, detail="No active organization membership is available.")
    organization = db.get(Organization, membership.organization_id)
    if organization is None or not organization.is_active:
        raise HTTPException(status_code=403, detail="Organization access is disabled.")
    return _auth_response(db, user, organization, membership.role)


@router.get("/me")
def me(
    principal: CurrentPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    user = db.get(User, principal.user_id)
    organization = db.get(Organization, principal.organization_id)
    plan = db.get(Plan, organization.plan_id) if organization else None
    if user is None or organization is None or plan is None:
        raise HTTPException(status_code=401, detail="Authenticated account is no longer available.")
    return {
        "user": {"id": user.id, "email": user.email},
        "organization": {
            "id": organization.id,
            "name": organization.name,
            "slug": organization.slug,
            "plan": plan.code,
            "role": principal.role,
        },
        "auth_method": principal.auth_method,
    }
