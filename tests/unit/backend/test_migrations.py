from dataclasses import replace

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text

import backend.app.config as config_module


def test_initial_migration_creates_b2b_schema_and_default_plans(tmp_path, monkeypatch):
    database_path = tmp_path / "migration-smoke.db"
    database_url = f"sqlite:///{database_path}"
    monkeypatch.setattr(
        config_module,
        "settings",
        replace(config_module.settings, database_url=database_url),
    )

    alembic_config = Config("alembic.ini")
    command.upgrade(alembic_config, "head")

    engine = create_engine(database_url)
    try:
        tables = set(inspect(engine).get_table_names())
        assert {
            "users",
            "plans",
            "organizations",
            "memberships",
            "api_keys",
            "usage_logs",
            "audit_logs",
            "alembic_version",
        }.issubset(tables)
        with engine.connect() as connection:
            plans = connection.execute(
                text("SELECT code, daily_request_limit FROM plans ORDER BY code")
            ).all()
        assert plans == [("free", 100), ("pro", 5000)]
    finally:
        engine.dispose()
