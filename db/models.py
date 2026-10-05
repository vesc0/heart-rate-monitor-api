import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from db.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=True) # Legacy field kept for backward compatibility with existing databases.
    name = Column(String, nullable=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    token_version = Column(Integer, nullable=False, default=0, server_default="0")
    age = Column(Integer, nullable=True)
    gender = Column(String, nullable=True)
    height_cm = Column(Integer, nullable=True)
    weight_kg = Column(Integer, nullable=True)
    health_issues = Column(String, nullable=True)

    heart_rate_records = relationship("HeartRateRecord", back_populates="user", cascade="all, delete-orphan")


class HeartRateRecord(Base):
    __tablename__ = "heart_rate_records"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    bpm = Column(Integer, nullable=False)
    recorded_at = Column(DateTime(timezone=True), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    stress_level = Column(String, nullable=True)
    stress_explanation = Column(String, nullable=True)
    activity_state = Column(String, nullable=True)

    user = relationship("User", back_populates="heart_rate_records")