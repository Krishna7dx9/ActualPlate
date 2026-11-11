"""SQLAlchemy engine, session, and declarative base."""
from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.engine import make_url
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from backend.app.config import settings


class Base(DeclarativeBase):
    pass


def _engine_url(url: str) -> str:
    parsed = make_url(url)
    if parsed.drivername == "postgresql":
        return parsed.set(drivername="postgresql+psycopg").render_as_string(
            hide_password=False
        )
    return url


database_url = _engine_url(settings.database_url)
sqlite = database_url.startswith("sqlite")
engine = create_engine(
    database_url,
    connect_args={"check_same_thread": False} if sqlite else {},
    pool_pre_ping=True,
)
SessionLocal = sessionmaker(
    bind=engine,
    class_=Session,
    autoflush=False,
    expire_on_commit=False,
)


if sqlite:
    @event.listens_for(engine, "connect")
    def _enable_sqlite_foreign_keys(connection, _record):
        cursor = connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
