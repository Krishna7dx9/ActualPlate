from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.app.database import Base, get_db
from backend.app.main import app
from backend.app.models import Plan


@pytest.fixture()
def client():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSessionLocal = sessionmaker(
        bind=engine,
        autoflush=False,
        expire_on_commit=False,
    )
    Base.metadata.create_all(bind=engine)
    with TestingSessionLocal() as db:
        db.add_all([
            Plan(
                id="plan-free-v1",
                code="free",
                daily_request_limit=100,
                features={"api_keys": True},
                is_active=True,
            ),
            Plan(
                id="plan-pro-v1",
                code="pro",
                daily_request_limit=5000,
                features={"api_keys": True},
                is_active=True,
            ),
        ])
        db.commit()

    def override_get_db():
        with TestingSessionLocal() as db:
            yield db

    app.dependency_overrides[get_db] = override_get_db
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.clear()
        Base.metadata.drop_all(bind=engine)
        engine.dispose()


def _signup(client: TestClient, email="owner@example.com"):
    response = client.post(
        "/v1/auth/signup",
        json={
            "email": email,
            "password": "correct-horse-battery-42",
            "organization_name": "Acme Food Systems",
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def test_signup_login_and_authenticated_me(client):
    created = _signup(client)
    assert created["organization"]["role"] == "owner"
    assert created["organization"]["plan_code"] == "free"
    assert created["access_token"]
    assert created["expires_in"] > 0

    login = client.post(
        "/v1/auth/login",
        json={"email": "OWNER@example.com", "password": "correct-horse-battery-42"},
    )
    assert login.status_code == 200
    token = login.json()["access_token"]

    me = client.get("/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200
    assert me.json()["user"]["email"] == "owner@example.com"
    assert me.json()["organization"]["role"] == "owner"
    assert me.json()["auth_method"] == "jwt"


def test_duplicate_signup_and_bad_password_are_rejected(client):
    _signup(client)
    duplicate = client.post(
        "/v1/auth/signup",
        json={
            "email": "owner@example.com",
            "password": "correct-horse-battery-42",
            "organization_name": "Another Company",
        },
    )
    assert duplicate.status_code == 409

    bad_login = client.post(
        "/v1/auth/login",
        json={"email": "owner@example.com", "password": "wrong-password"},
    )
    assert bad_login.status_code == 401
    assert bad_login.json()["detail"] == "Invalid email or password."


def test_api_key_is_shown_once_scoped_and_revocable(client):
    created = _signup(client)
    token = created["access_token"]
    jwt_headers = {"Authorization": f"Bearer {token}"}

    create = client.post(
        "/v1/api-keys",
        headers=jwt_headers,
        json={"name": "CI integration", "scopes": ["keys:manage"], "expires_in_days": 30},
    )
    assert create.status_code == 201, create.text
    key_payload = create.json()
    raw_key = key_payload["key"]
    assert raw_key.startswith("ap_dev_")
    assert key_payload["key_prefix"]
    assert "key_hash" not in key_payload

    api_headers = {"X-API-Key": raw_key}
    me = client.get("/v1/auth/me", headers=api_headers)
    assert me.status_code == 200
    assert me.json()["auth_method"] == "api_key"

    listed = client.get("/v1/api-keys", headers=api_headers)
    assert listed.status_code == 200
    assert len(listed.json()) == 1
    assert "key" not in listed.json()[0]
    assert "key_hash" not in listed.json()[0]

    revoked = client.delete(f"/v1/api-keys/{key_payload['id']}", headers=api_headers)
    assert revoked.status_code == 204

    rejected = client.get("/v1/auth/me", headers=api_headers)
    assert rejected.status_code == 401


def test_api_key_management_requires_role_and_scope(client):
    created = _signup(client)
    token = created["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    key_response = client.post(
        "/v1/api-keys",
        headers=headers,
        json={"name": "scan-only", "scopes": ["scans:write"]},
    )
    assert key_response.status_code == 201

    response = client.get(
        "/v1/api-keys",
        headers={"X-API-Key": key_response.json()["key"]},
    )
    assert response.status_code == 403


def test_api_keys_are_isolated_to_their_organization(client):
    first = _signup(client, "first@example.com")
    second = _signup(client, "second@example.com")
    first_key = client.post(
        "/v1/api-keys",
        headers={"Authorization": f"Bearer {first['access_token']}"},
        json={"name": "first org key", "scopes": ["keys:manage"]},
    ).json()

    response = client.delete(
        f"/v1/api-keys/{first_key['id']}",
        headers={"Authorization": f"Bearer {second['access_token']}"},
    )
    assert response.status_code == 404


def test_authentication_is_required_for_me(client):
    response = client.get("/v1/auth/me")
    assert response.status_code == 401
