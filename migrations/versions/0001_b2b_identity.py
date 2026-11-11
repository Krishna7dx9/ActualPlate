"""Create identity, organization, API key, usage, and audit tables.

Revision ID: 0001_b2b_identity
Revises:
Create Date: 2026-11-12
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0001_b2b_identity"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("email", sa.String(length=320), nullable=False),
        sa.Column("password_hash", sa.String(length=255), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("email_verified_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("email", name="uq_users_email"),
    )
    op.create_index("ix_users_email_active", "users", ["email", "is_active"])

    op.create_table(
        "plans",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("code", sa.String(length=32), nullable=False),
        sa.Column("daily_request_limit", sa.Integer(), nullable=False),
        sa.Column("features", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("code", name="uq_plans_code"),
    )

    op.create_table(
        "organizations",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("name", sa.String(length=160), nullable=False),
        sa.Column("slug", sa.String(length=180), nullable=False),
        sa.Column("plan_id", sa.String(length=36), sa.ForeignKey("plans.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("slug", name="uq_organizations_slug"),
    )
    op.create_index("ix_organizations_active_plan", "organizations", ["is_active", "plan_id"])

    op.create_table(
        "memberships",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("user_id", sa.String(length=36), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("organization_id", sa.String(length=36), sa.ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("role", sa.String(length=16), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("user_id", "organization_id", name="uq_membership_user_org"),
        sa.CheckConstraint("role IN ('owner', 'admin', 'member')", name="ck_membership_role"),
    )
    op.create_index("ix_memberships_org_active", "memberships", ["organization_id", "is_active"])

    op.create_table(
        "api_keys",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("organization_id", sa.String(length=36), sa.ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("created_by_user_id", sa.String(length=36), sa.ForeignKey("users.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("key_prefix", sa.String(length=24), nullable=False),
        sa.Column("key_hash", sa.String(length=64), nullable=False),
        sa.Column("scopes", sa.JSON(), nullable=False, server_default=sa.text("'[]'")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("key_hash", name="uq_api_keys_hash"),
    )
    op.create_index("ix_api_keys_org_revoked", "api_keys", ["organization_id", "revoked_at"])
    op.create_index("ix_api_keys_prefix", "api_keys", ["key_prefix"])

    op.create_table(
        "usage_logs",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("organization_id", sa.String(length=36), sa.ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("api_key_id", sa.String(length=36), sa.ForeignKey("api_keys.id", ondelete="SET NULL"), nullable=True),
        sa.Column("request_id", sa.String(length=100), nullable=True),
        sa.Column("endpoint", sa.String(length=200), nullable=False),
        sa.Column("status_code", sa.Integer(), nullable=True),
        sa.Column("units", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("attributes", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
    )
    op.create_index("ix_usage_logs_org_created", "usage_logs", ["organization_id", "created_at"])
    op.create_index("ix_usage_logs_key_created", "usage_logs", ["api_key_id", "created_at"])

    op.create_table(
        "audit_logs",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("organization_id", sa.String(length=36), sa.ForeignKey("organizations.id", ondelete="SET NULL"), nullable=True),
        sa.Column("actor_user_id", sa.String(length=36), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("action", sa.String(length=100), nullable=False),
        sa.Column("target_type", sa.String(length=100), nullable=True),
        sa.Column("target_id", sa.String(length=100), nullable=True),
        sa.Column("request_id", sa.String(length=100), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("attributes", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
    )
    op.create_index("ix_audit_logs_org_created", "audit_logs", ["organization_id", "created_at"])
    op.create_index("ix_audit_logs_action_created", "audit_logs", ["action", "created_at"])

    plans = sa.table(
        "plans",
        sa.column("id", sa.String()),
        sa.column("code", sa.String()),
        sa.column("daily_request_limit", sa.Integer()),
        sa.column("features", sa.JSON()),
        sa.column("is_active", sa.Boolean()),
    )
    op.bulk_insert(
        plans,
        [
            {"id": "plan-free-v1", "code": "free", "daily_request_limit": 100, "features": {"api_keys": True}, "is_active": True},
            {"id": "plan-pro-v1", "code": "pro", "daily_request_limit": 5000, "features": {"api_keys": True}, "is_active": True},
        ],
    )


def downgrade() -> None:
    op.drop_index("ix_audit_logs_action_created", table_name="audit_logs")
    op.drop_index("ix_audit_logs_org_created", table_name="audit_logs")
    op.drop_table("audit_logs")
    op.drop_index("ix_usage_logs_key_created", table_name="usage_logs")
    op.drop_index("ix_usage_logs_org_created", table_name="usage_logs")
    op.drop_table("usage_logs")
    op.drop_index("ix_api_keys_prefix", table_name="api_keys")
    op.drop_index("ix_api_keys_org_revoked", table_name="api_keys")
    op.drop_table("api_keys")
    op.drop_index("ix_memberships_org_active", table_name="memberships")
    op.drop_table("memberships")
    op.drop_index("ix_organizations_active_plan", table_name="organizations")
    op.drop_table("organizations")
    op.drop_table("plans")
    op.drop_index("ix_users_email_active", table_name="users")
    op.drop_table("users")
