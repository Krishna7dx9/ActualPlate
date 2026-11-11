"""Public request/response schemas for identity and API-key management."""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, EmailStr, Field, field_validator

ALLOWED_API_SCOPES = frozenset({"scans:write", "usage:read", "keys:manage"})


class SignupRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=12, max_length=72)
    organization_name: str = Field(min_length=2, max_length=160)

    @field_validator("password")
    @classmethod
    def validate_password_bytes(cls, value: str) -> str:
        if not 12 <= len(value.encode("utf-8")) <= 72:
            raise ValueError("Password must contain 12 to 72 UTF-8 bytes.")
        return value

    @field_validator("organization_name")
    @classmethod
    def normalize_organization_name(cls, value: str) -> str:
        value = value.strip()
        if len(value) < 2:
            raise ValueError("Organization name must contain at least two characters.")
        return value


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=72)


class UserSummary(BaseModel):
    id: str
    email: EmailStr


class OrganizationSummary(BaseModel):
    id: str
    name: str
    slug: str
    plan_code: str
    role: Literal["owner", "admin", "member"]


class AuthResponse(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_in: int
    user: UserSummary
    organization: OrganizationSummary


class APIKeyCreateRequest(BaseModel):
    name: str = Field(min_length=2, max_length=100)
    scopes: list[str] = Field(default_factory=lambda: ["scans:write"])
    expires_in_days: int = Field(default=90, ge=1, le=365)

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        value = value.strip()
        if len(value) < 2:
            raise ValueError("API key name must contain at least two characters.")
        return value

    @field_validator("scopes")
    @classmethod
    def validate_scopes(cls, scopes: list[str]) -> list[str]:
        if not scopes:
            raise ValueError("At least one API key scope is required.")
        if len(set(scopes)) != len(scopes):
            raise ValueError("API key scopes must not contain duplicates.")
        unknown = set(scopes) - ALLOWED_API_SCOPES
        if unknown:
            raise ValueError("One or more API key scopes are not supported.")
        return sorted(scopes)


class APIKeyCreatedResponse(BaseModel):
    id: str
    name: str
    key: str
    key_prefix: str
    scopes: list[str]
    created_at: datetime
    expires_at: datetime | None


class APIKeySummary(BaseModel):
    id: str
    name: str
    key_prefix: str
    scopes: list[str]
    created_at: datetime
    expires_at: datetime | None
    revoked_at: datetime | None
    last_used_at: datetime | None
