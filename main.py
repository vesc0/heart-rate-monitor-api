import logging
import uuid

import joblib
import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session

from database import SessionLocal, engine, Base
from models import HeartRateRecord, User
from schemas import (
    HeartRateBulkDelete,
    HeartRateCreate,
    HeartRateResponse,
    StressPredictRequest,
    StressPredictResponse,
    UserLogin,
    UserProfile,
    UserProfileUpdate,
    UserRegister,
)
from utils import (
    create_access_token,
    get_current_user_id,
    hash_password,
    verify_password,
)
from datetime import timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Bootstrap ─────────────────────────────────────────
Base.metadata.create_all(bind=engine)

# Add missing columns (lightweight schema migration)
try:
    with engine.connect() as conn:
        from sqlalchemy import text, inspect as sa_inspect
        inspector = sa_inspect(engine)
        columns = [c["name"] for c in inspector.get_columns("heart_rate_records")]
        if "stress_level" not in columns:
            conn.execute(text("ALTER TABLE heart_rate_records ADD COLUMN stress_level VARCHAR"))
            conn.commit()
            logger.info("Added stress_level column to heart_rate_records")
        user_columns = [c["name"] for c in inspector.get_columns("users")]
        for col_name, col_type in [("gender", "VARCHAR"), ("height_cm", "INTEGER"), ("weight_kg", "INTEGER")]:
            if col_name not in user_columns:
                conn.execute(text(f"ALTER TABLE users ADD COLUMN {col_name} {col_type}"))
                conn.commit()
                logger.info("Added %s column to users", col_name)
except Exception as e:
    logger.warning("Schema migration check failed (non-fatal): %s", e)

app = FastAPI(title="Heart Rate Monitor API")

# ── Load ML models at startup ─────────────────────────
_ML_DIR = Path(__file__).parent / "ml_models"
_ml_artifacts = None
try:
    _ml_artifacts = joblib.load(_ML_DIR / "all_artifacts.joblib")
    logger.info("ML models loaded from %s", _ML_DIR)
except Exception as e:
    logger.warning("ML models not found at %s — /stress-predict will be unavailable: %s", _ML_DIR, e)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Global exception handlers ────────────────────────


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = []
    for err in exc.errors():
        field = " -> ".join(str(loc) for loc in err.get("loc", []))
        errors.append({"field": field, "message": err.get("msg", "")})
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": "Validation error", "errors": errors},
    )


@app.exception_handler(OperationalError)
async def db_exception_handler(request: Request, exc: OperationalError):
    logger.error("Database error: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": "Database is currently unavailable. Please try again later."},
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An unexpected error occurred. Please try again later."},
    )


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ── Auth ──────────────────────────────────────────────


@app.post("/register", status_code=status.HTTP_201_CREATED)
def register(body: UserRegister, db: Session = Depends(get_db)):
    username = body.username or body.email.split("@")[0]
    user = User(
        username=username,
        email=body.email,
        hashed_password=hash_password(body.password),
    )
    db.add(user)
    try:
        db.commit()
        db.refresh(user)
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email or username already registered",
        )
    token = create_access_token(data={"sub": user.id})
    return {
        "message": "User registered",
        "username": user.username,
        "access_token": token,
        "token_type": "bearer",
    }


@app.post("/login")
def login(body: UserLogin, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == body.email).first()
    if not user or not verify_password(body.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    access_token = create_access_token(data={"sub": user.id})
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "username": user.username,
        "email": user.email,
        "age": user.age,
        "gender": user.gender,
        "height_cm": user.height_cm,
        "weight_kg": user.weight_kg,
        "health_issues": user.health_issues,
    }


# ── Profile ───────────────────────────────────────────


@app.get("/me", response_model=UserProfile)
def get_profile(
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )
    return user


@app.put("/me", response_model=UserProfile)
def update_profile(
    body: UserProfileUpdate,
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )
    if body.username is not None:
        user.username = body.username
    if body.email is not None:
        user.email = body.email
    if body.age is not None:
        user.age = body.age
    if body.gender is not None:
        user.gender = body.gender
    if body.height_cm is not None:
        user.height_cm = body.height_cm
    if body.weight_kg is not None:
        user.weight_kg = body.weight_kg
    if body.health_issues is not None:
        user.health_issues = body.health_issues
    try:
        db.commit()
        db.refresh(user)
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email or username already taken",
        )
    return user


@app.post("/logout", status_code=status.HTTP_200_OK)
def logout(user_id: int = Depends(get_current_user_id)):
    """JWT is stateless; the client clears the token.
    This endpoint exists for API completeness and future token-blacklist support."""
    return {"message": "Logged out successfully"}


# ── Heart-rate CRUD ───────────────────────────────────


@app.post(
    "/heart-rate",
    response_model=HeartRateResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_heart_rate(
    entry: HeartRateCreate,
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    # Normalize incoming timestamp to UTC and ensure tz-aware
    rec_dt = entry.recorded_at
    if rec_dt.tzinfo is None:
        rec_dt = rec_dt.replace(tzinfo=timezone.utc)
    else:
        rec_dt = rec_dt.astimezone(timezone.utc)

    record = HeartRateRecord(
        id=entry.id or str(uuid.uuid4()),
        user_id=user_id,
        bpm=entry.bpm,
        recorded_at=rec_dt,
        stress_level=entry.stress_level,
    )
    db.add(record)
    try:
        db.commit()
        db.refresh(record)
    except IntegrityError:
        db.rollback()
        # Duplicate id → likely a client retry; return the existing record
        existing = (
            db.query(HeartRateRecord)
            .filter(HeartRateRecord.id == record.id, HeartRateRecord.user_id == user_id)
            .first()
        )
        if existing:
            return existing
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="Duplicate entry"
        )
    return record


@app.get("/heart-rate", response_model=list[HeartRateResponse])
def list_heart_rate(
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db),
    limit: int = Query(500, ge=1, le=5000),
    offset: int = Query(0, ge=0),
):
    records = (
        db.query(HeartRateRecord)
        .filter(HeartRateRecord.user_id == user_id)
        .order_by(HeartRateRecord.recorded_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    return records


@app.delete("/heart-rate/{entry_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_heart_rate(
    entry_id: str,
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    record = (
        db.query(HeartRateRecord)
        .filter(HeartRateRecord.id == entry_id, HeartRateRecord.user_id == user_id)
        .first()
    )
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Record not found"
        )
    db.delete(record)
    db.commit()


@app.post("/heart-rate/batch-delete", status_code=status.HTTP_200_OK)
def batch_delete_heart_rate(
    body: HeartRateBulkDelete,
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    deleted = (
        db.query(HeartRateRecord)
        .filter(
            HeartRateRecord.id.in_(body.ids),
            HeartRateRecord.user_id == user_id,
        )
        .delete(synchronize_session=False)
    )
    db.commit()
    return {"deleted": deleted}


# ── Stress prediction ────────────────────────────────


@app.post("/stress-predict", response_model=StressPredictResponse)
def predict_stress(
    body: StressPredictRequest,
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    """Run the trained stress model on HRV features (+ optional demographics)
    from a 60-second PPG capture window.  Returns stress_level_pct 0–100."""
    if _ml_artifacts is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Stress prediction unavailable.",
        )

    feature_cols = _ml_artifacts["feature_columns"]
    model = _ml_artifacts["model"]
    demo_defaults = _ml_artifacts.get("demo_defaults", {})

    # Build feature vector: start with demographic defaults, override with request
    features = dict(demo_defaults)

    # Map all HRV fields from the request body
    hrv_map = {
        "sdnn": body.sdnn, "median_rr": body.median_rr,
        "cv_rr": body.cv_rr, "rmssd": body.rmssd,
        "pnn50": body.pnn50, "pnn20": body.pnn20, "mean_hr": body.mean_hr,
        "std_hr": body.std_hr, "min_hr": body.min_hr, "max_hr": body.max_hr,
        "hr_range": body.hr_range,
        "lf_power": body.lf_power, "hf_power": body.hf_power,
        "lf_hf_ratio": body.lf_hf_ratio, "total_power": body.total_power,
        "lf_norm": body.lf_norm,
        "sd1": body.sd1, "sd2": body.sd2, "sd_ratio": body.sd_ratio,
    }
    features.update(hrv_map)

    # Override demographics from request if provided
    for demo_key, body_val in [
        ("age", body.age), ("gender_male", body.gender_male),
        ("height_cm", body.height_cm), ("weight_kg", body.weight_kg),
    ]:
        if body_val is not None:
            features[demo_key] = float(body_val)

    # Fallback: fill remaining defaults from user profile demographics
    user = db.query(User).filter(User.id == user_id).first()
    if user:
        if body.age is None and user.age:
            features["age"] = float(user.age)
        if body.gender_male is None and user.gender:
            features["gender_male"] = 1.0 if user.gender == "male" else 0.0
        if body.height_cm is None and user.height_cm:
            features["height_cm"] = float(user.height_cm)
        if body.weight_kg is None and user.weight_kg:
            features["weight_kg"] = float(user.weight_kg)

    X = np.array([[features.get(c, 0.0) for c in feature_cols]])

    proba = model.predict_proba(X)[0]
    stress_pct = round(float(proba[1]) * 100, 1)

    return StressPredictResponse(
        stress_level_pct=stress_pct,
        is_stressed=stress_pct >= 50,
    )