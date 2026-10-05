"""Store stress_level as an integer

Revision ID: d7a3c9e1f204
Revises: c4e8a2f6b915
Create Date: 2026-10-05

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "d7a3c9e1f204"
down_revision: Union[str, Sequence[str], None] = "c4e8a2f6b915"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column(
        "heart_rate_records",
        "stress_level",
        type_=sa.Integer(),
        postgresql_using="nullif(regexp_replace(stress_level, '[^0-9]', '', 'g'), '')::integer",
    )


def downgrade() -> None:
    op.alter_column("heart_rate_records", "stress_level", type_=sa.String())
