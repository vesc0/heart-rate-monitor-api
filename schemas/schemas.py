import re
from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator


# ── Validators ────────────────────────────────────────
_PASSWORD_MIN = 8
_PASSWORD_MAX = 72

ActivityState = Literal["resting", "activity", "recovery"]


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

    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        return _validate_password(v)


class UserLogin(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=1, max_length=_PASSWORD_MAX)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class MessageResponse(BaseModel):
    message: str
    email: EmailStr
    name: Optional[str] = None


class UserProfile(BaseModel):
    name: Optional[str] = None
    email: EmailStr
    age: Optional[int] = None
    gender: Optional[str] = None
    height_cm: Optional[int] = None
    weight_kg: Optional[int] = None
    health_issues: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class UserProfileUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=120)
    email: Optional[EmailStr] = None
    age: Optional[int] = Field(None, ge=1, le=150)
    gender: Optional[str] = Field(None, pattern=r'^(male|female|other)$')
    height_cm: Optional[int] = Field(None, ge=50, le=300)
    weight_kg: Optional[int] = Field(None, ge=20, le=500)
    health_issues: Optional[str] = Field(None, max_length=500)


# ── Heart-rate entries ────────────────────────────────
class HeartRateCreate(BaseModel):
    id: Optional[str] = None  # client-generated UUID (optional)
    bpm: int = Field(..., ge=30, le=250)
    recorded_at: datetime
    stress_level: Optional[str] = None
    stress_explanation: Optional[str] = Field(None, max_length=4000)
    activity_state: Optional[ActivityState] = None


class HeartRateResponse(BaseModel):
    id: str
    bpm: int
    recorded_at: datetime
    created_at: datetime
    stress_level: Optional[str] = None
    stress_explanation: Optional[str] = None
    activity_state: Optional[ActivityState] = None

    model_config = ConfigDict(from_attributes=True)


class HeartRateBulkDelete(BaseModel):
    ids: list[str] = Field(..., min_length=1, max_length=500)


# ── Stress prediction ────────────────────────────────
# HRV features computed from a 60-second PPG capture window.
class StressPredictRequest(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    sdnn: float = Field(..., ge=0, le=2000, description="Std dev of RR intervals (ms)")
    median_rr: float = Field(..., ge=240, le=2000, description="Median RR interval (ms)")
    cv_rr: float = Field(..., ge=0, le=5, description="Coefficient of variation of RR")
    rmssd: float = Field(..., ge=0, le=2000, description="Root mean square of successive differences (ms)")
    pnn50: float = Field(..., ge=0, le=100, description="% of successive diffs > 50ms")
    pnn20: float = Field(..., ge=0, le=100, description="% of successive diffs > 20ms")
    mean_hr: float = Field(..., ge=30, le=250, description="Mean heart rate (BPM)")
    std_hr: float = Field(..., ge=0, le=250, description="Std dev of heart rate (BPM)")
    min_hr: float = Field(..., ge=30, le=250, description="Minimum heart rate (BPM)")
    max_hr: float = Field(..., ge=30, le=250, description="Maximum heart rate (BPM)")
    hr_range: float = Field(..., ge=0, le=250, description="Heart rate range (BPM)")
    # Frequency-domain HRV
    lf_power: float = Field(0, ge=0, le=1e9, description="Low-frequency power")
    hf_power: float = Field(0, ge=0, le=1e9, description="High-frequency power")
    lf_hf_ratio: float = Field(0, ge=0, le=1e9, description="LF/HF ratio")
    total_power: float = Field(0, ge=0, le=1e9, description="Total spectral power")
    lf_norm: float = Field(0, ge=0, le=100, description="Normalized LF power (%)")
    # Nonlinear HRV
    sd1: float = Field(0, ge=0, le=2000, description="Poincaré SD1")
    sd2: float = Field(0, ge=0, le=2000, description="Poincaré SD2")
    sd_ratio: float = Field(0, ge=0, le=1e9, description="SD2/SD1 ratio")


class StressPredictResponse(BaseModel):
    stress_level_pct: float
    is_stressed: bool


ShapDirection = Literal["increases_stress", "decreases_stress", "neutral"]


class ShapFeatureContribution(BaseModel):
    feature: str = Field(..., description="Model feature name")
    display_name: str = Field(..., description="Human-readable feature name")
    value: float = Field(..., description="Input value used by the model")
    shap_value: float = Field(..., description="SHAP contribution on probability scale")
    contribution_pct: float = Field(..., description="SHAP contribution in percentage points")
    direction: ShapDirection


class ShapExplanation(BaseModel):
    model_output: Literal["stress_probability"]
    base_value: float = Field(..., description="Baseline stress probability")
    base_value_pct: float = Field(..., description="Baseline stress probability percentage")
    prediction_value: float = Field(..., description="Predicted stress probability")
    prediction_pct: float = Field(..., description="Predicted stress percentage")
    top_contributions: list[ShapFeatureContribution]


class StressExplainResponse(StressPredictResponse):
    explanation: ShapExplanation


class RetrievedContextItem(BaseModel):
    id: str
    title: str
    source: str
    text: str
    score: float


class StressAnalysisResponse(StressPredictResponse):
    important_features: list[ShapFeatureContribution]
    retrieved_context: list[RetrievedContextItem]
    explanation: Optional[str] = None
