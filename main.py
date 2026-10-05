import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from slowapi.errors import RateLimitExceeded
from sqlalchemy.exc import OperationalError

from db.migrations import run_migrations
from api.routes import auth, profile, heart_rate, stress
from services.stress_service import rag_pipeline

from core.config import settings
from core.logger import setup_logging, get_logger

# Configure logging
setup_logging()
logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.RUN_MIGRATIONS:
        run_migrations()
    # Loading the embedding model takes seconds; do it off the first request.
    threading.Thread(target=rag_pipeline.warm_up, daemon=True).start()
    yield

# FastAPI application instance
app = FastAPI(title="Heart Rate Monitor API", lifespan=lifespan)

# Exception handlers
@app.exception_handler(RateLimitExceeded)
async def rate_limit_exception_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={"detail": "Too many requests. Please try again later."},
    )

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
