"""Add stress_explanation to heart_rate_records

Revision ID: b2c9f1a7d340
Revises: 1f32c67d450e
Create Date: 2026-09-09

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "b2c9f1a7d340"
down_revision: Union[str, Sequence[str], None] = "1f32c67d450e"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("heart_rate_records", sa.Column("stress_explanation", sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column("heart_rate_records", "stress_explanation")
