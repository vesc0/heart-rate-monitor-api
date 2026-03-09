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
    gender: Optional[str] = None
    height_cm: Optional[int] = None
    weight_kg: Optional[int] = None
    health_issues: Optional[str] = None

    class Config:
        from_attributes = True


class UserProfileUpdate(BaseModel):
    username: Optional[str] = Field(None, min_length=3, max_length=30)
    email: Optional[EmailStr] = None
    age: Optional[int] = Field(None, ge=1, le=150)
    gender: Optional[str] = Field(None, pattern=r'^(male|female)$')
    height_cm: Optional[int] = Field(None, ge=50, le=300)
    weight_kg: Optional[int] = Field(None, ge=20, le=500)
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

# HRV features computed from a 60-second PPG capture window,
# plus optional demographics for improved prediction.
class StressPredictRequest(BaseModel):
    # HRV features (required)
    sdnn: float = Field(..., description="Std dev of RR intervals (ms)")
    median_rr: float = Field(..., description="Median RR interval (ms)")
    cv_rr: float = Field(..., description="Coefficient of variation of RR")
    rmssd: float = Field(..., description="Root mean square of successive differences (ms)")
    pnn50: float = Field(..., description="% of successive diffs > 50ms")
    pnn20: float = Field(..., description="% of successive diffs > 20ms")
    mean_hr: float = Field(..., description="Mean heart rate (BPM)")
    std_hr: float = Field(..., description="Std dev of heart rate (BPM)")
    min_hr: float = Field(..., description="Minimum heart rate (BPM)")
    max_hr: float = Field(..., description="Maximum heart rate (BPM)")
    hr_range: float = Field(..., description="Heart rate range (BPM)")
    # Frequency-domain HRV
    lf_power: float = Field(0, description="Low-frequency power")
    hf_power: float = Field(0, description="High-frequency power")
    lf_hf_ratio: float = Field(0, description="LF/HF ratio")
    total_power: float = Field(0, description="Total spectral power")
    lf_norm: float = Field(0, description="Normalized LF power (%)")
    # Nonlinear HRV
    sd1: float = Field(0, description="Poincaré SD1")
    sd2: float = Field(0, description="Poincaré SD2")
    sd_ratio: float = Field(0, description="SD2/SD1 ratio")
    # Demographics (optional — medians used when missing)
    age: Optional[float] = Field(None, description="Age in years")
    gender_male: Optional[float] = Field(None, description="1=male, 0=female")
    height_cm: Optional[float] = Field(None, description="Height in cm")
    weight_kg: Optional[float] = Field(None, description="Weight in kg")


class StressPredictResponse(BaseModel):
    stress_level_pct: float
    is_stressed: bool
