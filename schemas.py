import re
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr, Field, field_validator


# ── Validators ────────────────────────────────────────

_USERNAME_RE = re.compile(r"^[a-zA-Z0-9_-]{3,30}$")
_PASSWORD_MIN = 8
_PASSWORD_MAX = 72


def _validate_password(v: str) -> str:
    if len(v) < _PASSWORD_MIN:
        raise ValueError(f"Password must be at least {_PASSWORD_MIN} characters")
    if len(v) > _PASSWORD_MAX:
        raise ValueError(f"Password must be at most {_PASSWORD_MAX} characters")
    if not re.search(r"[A-Z]", v):
        raise ValueError("Password must contain at least one uppercase letter")
    if not re.search(r"[a-z]", v):
        raise ValueError("Password must contain at least one lowercase letter")
    if not re.search(r"\d", v):
        raise ValueError("Password must contain at least one digit")
    return v


# ── Auth ──────────────────────────────────────────────


class UserRegister(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=_PASSWORD_MIN, max_length=_PASSWORD_MAX)
    username: Optional[str] = None  # auto-derived from email if omitted

    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        return _validate_password(v)

    @field_validator("username")
    @classmethod
    def username_format(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not _USERNAME_RE.match(v):
            raise ValueError(
                "Username must be 3-30 characters and contain only letters, digits, hyphens, or underscores"
            )
        return v


class UserLogin(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=1, max_length=_PASSWORD_MAX)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class MessageResponse(BaseModel):
    message: str
    username: str


class UserProfile(BaseModel):
    username: str
    email: str
    age: Optional[int] = None
    health_issues: Optional[str] = None

    class Config:
        from_attributes = True


class UserProfileUpdate(BaseModel):
    username: Optional[str] = Field(None, min_length=3, max_length=30)
    email: Optional[EmailStr] = None
    age: Optional[int] = Field(None, ge=1, le=150)
    health_issues: Optional[str] = Field(None, max_length=500)

    @field_validator("username")
    @classmethod
    def username_format(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not _USERNAME_RE.match(v):
            raise ValueError(
                "Username must be 3-30 characters and contain only letters, digits, hyphens, or underscores"
            )
        return v


# ── Heart-rate entries ────────────────────────────────


class HeartRateCreate(BaseModel):
    id: Optional[str] = None  # client-generated UUID (optional)
    bpm: int = Field(..., ge=30, le=250)
    recorded_at: datetime
    stress_level: Optional[str] = None


class HeartRateResponse(BaseModel):
    id: str
    bpm: int
    recorded_at: datetime
    created_at: datetime
    stress_level: Optional[str] = None

    class Config:
        from_attributes = True


class HeartRateBulkDelete(BaseModel):
    ids: list[str] = Field(..., min_length=1, max_length=500)


# ── Stress prediction ────────────────────────────────


class StressPredictRequest(BaseModel):
    """HRV features computed from a 60-second PPG capture window."""
    mean_rr: float = Field(..., description="Mean RR interval (ms)")
    sdnn: float = Field(..., description="Std dev of RR intervals (ms)")
    median_rr: float = Field(..., description="Median RR interval (ms)")
    cv_rr: float = Field(..., description="Coefficient of variation of RR")
    rmssd: float = Field(..., description="Root mean square of successive differences (ms)")
    sdsd: float = Field(..., description="Std dev of successive differences (ms)")
    pnn50: float = Field(..., description="% of successive diffs > 50ms")
    pnn20: float = Field(..., description="% of successive diffs > 20ms")
    mean_hr: float = Field(..., description="Mean heart rate (BPM)")
    std_hr: float = Field(..., description="Std dev of heart rate (BPM)")
    min_hr: float = Field(..., description="Minimum heart rate (BPM)")
    max_hr: float = Field(..., description="Maximum heart rate (BPM)")
    hr_range: float = Field(..., description="Heart rate range (BPM)")
    num_beats: float = Field(..., description="Number of beats in window")


class StressPredictResponse(BaseModel):
    is_stressed: bool
    stress_level: str
