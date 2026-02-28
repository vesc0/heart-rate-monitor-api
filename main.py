import logging
import uuid

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

logger = logging.getLogger(__name__)

# ── Bootstrap ─────────────────────────────────────────
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Heart Rate Monitor API")

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