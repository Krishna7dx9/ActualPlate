"""Transactional per-organization daily quota accounting."""
from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
from math import ceil

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from backend.app.database import SessionLocal
from backend.app.models import Organization, Plan, UsageLog, new_id


class DailyQuotaExceeded(RuntimeError):
    def __init__(self, *, used: int, limit: int, resets_at: datetime):
        super().__init__("Daily organization quota exceeded.")
        self.used = used
        self.limit = limit
        self.resets_at = resets_at
        self.retry_after_seconds = max(
            1,
            ceil((resets_at - datetime.now(timezone.utc)).total_seconds()),
        )


class OrganizationUnavailable(RuntimeError):
    pass


def _day_window(now: datetime | None = None) -> tuple[datetime, datetime]:
    current = now or datetime.now(timezone.utc)
    start = datetime.combine(current.date(), time.min, tzinfo=timezone.utc)
    return start, start + timedelta(days=1)


def _used_today(db: Session, organization_id: str, start: datetime, now: datetime) -> int:
    # Pending requests remain reserved for the full UTC day. Expiring an
    # in-flight reservation after a fixed timeout lets long-running scans
    # evade the organization's quota. Failed server-side requests (5xx) are
    # excluded once finalized; client errors and completed requests count.
    used = db.scalar(
        select(func.coalesce(func.sum(UsageLog.units), 0)).where(
            UsageLog.organization_id == organization_id,
            UsageLog.created_at >= start,
            or_(
                UsageLog.status_code < 500,
                UsageLog.status_code.is_(None),
            ),
        )
    )
    return int(used or 0)


def current_usage(db: Session, organization_id: str) -> dict:
    organization = db.get(Organization, organization_id)
    if organization is None or not organization.is_active:
        raise OrganizationUnavailable()
    plan = db.get(Plan, organization.plan_id)
    if plan is None or not plan.is_active:
        raise OrganizationUnavailable()
    start, resets_at = _day_window()
    used = _used_today(db, organization_id, start, datetime.now(timezone.utc))
    return {
        "plan": plan.code,
        "date": start.date().isoformat(),
        "used": used,
        "limit": plan.daily_request_limit,
        "remaining": max(0, plan.daily_request_limit - used),
        "resets_at": resets_at.isoformat(),
    }


def reserve_scan_usage(
    db: Session,
    *,
    organization_id: str,
    api_key_id: str | None,
    request_id: str,
    endpoint: str,
) -> tuple[str, int, int]:
    """Lock the organization row, check quota, and persist a pending usage row atomically."""
    organization = db.scalar(
        select(Organization)
        .where(Organization.id == organization_id, Organization.is_active.is_(True))
        .with_for_update()
    )
    if organization is None:
        raise OrganizationUnavailable()
    plan = db.get(Plan, organization.plan_id)
    if plan is None or not plan.is_active:
        raise OrganizationUnavailable()

    now = datetime.now(timezone.utc)
    start, resets_at = _day_window(now)
    used = _used_today(db, organization_id, start, now)
    limit = plan.daily_request_limit
    if used >= limit:
        raise DailyQuotaExceeded(used=used, limit=limit, resets_at=resets_at)

    record = UsageLog(
        id=new_id(),
        organization_id=organization_id,
        api_key_id=api_key_id,
        request_id=request_id,
        endpoint=endpoint,
        status_code=None,
        units=1,
        created_at=now,
        attributes={},
    )
    db.add(record)
    db.commit()
    return record.id, limit, max(0, limit - used - 1)


def finalize_usage(usage_id: str, status_code: int) -> None:
    """Finalize a reserved request using a short-lived independent DB session."""
    with SessionLocal() as db:
        record = db.get(UsageLog, usage_id)
        if record is None or record.status_code is not None:
            return
        record.status_code = status_code
        db.commit()
