import logging
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.exc import OperationalError

from db.database import engine, Base
from api.routes import auth, profile, heart_rate, stress

# Configure logging
logger = logging.getLogger(__name__)

# Bootstrap Database
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
        if "activity_state" not in columns:
            conn.execute(text("ALTER TABLE heart_rate_records ADD COLUMN activity_state VARCHAR"))
            conn.commit()
            logger.info("Added activity_state column to heart_rate_records")
        user_columns = [c["name"] for c in inspector.get_columns("users")]
        for col_name, col_type in [("gender", "VARCHAR"), ("height_cm", "INTEGER"), ("weight_kg", "INTEGER")]:
            if col_name not in user_columns:
                conn.execute(text(f"ALTER TABLE users ADD COLUMN {col_name} {col_type}"))
                conn.commit()
                logger.info("Added %s column to users", col_name)
        has_name = "name" in user_columns
        if not has_name:
            conn.execute(text("ALTER TABLE users ADD COLUMN name VARCHAR"))
            conn.commit()
            logger.info("Added name column to users")
            has_name = True
        if has_name and "username" in user_columns:
            conn.execute(text("UPDATE users SET name = username WHERE name IS NULL AND username IS NOT NULL"))
            conn.commit()
except Exception as e:
    logger.warning("Schema migration check failed (non-fatal): %s", e)

# FastAPI application instance
app = FastAPI(title="Heart Rate Monitor API")

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception handlers
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

# Routers
app.include_router(auth.router)
app.include_router(profile.router)
app.include_router(heart_rate.router)
app.include_router(stress.router)
