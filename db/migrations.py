"""Schema migrations applied at startup."""

from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import inspect

from core.logger import get_logger
from db.database import engine

logger = get_logger(__name__)

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "alembic.ini"


def run_migrations() -> None:
    """Bring the database up to the latest revision.

    Databases created before Alembic have the tables but no version record, so
    a plain upgrade would try to recreate them. Those are stamped at the initial
    revision first, which adopts the existing schema while leaving any later
    migrations to run normally.
    """
    config = Config(_CONFIG_PATH)
    tables = inspect(engine).get_table_names()

    if "alembic_version" not in tables and "users" in tables:
        logger.info("Adopting pre-Alembic database at the initial revision")
        command.stamp(config, ScriptDirectory.from_config(config).get_base())

    command.upgrade(config, "head")
