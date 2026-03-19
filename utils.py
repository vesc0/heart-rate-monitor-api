import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

from dotenv import load_dotenv
from fastapi import Depends, HTTPException, status
import bcrypt
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
import json
import re
try:
    import openai
except Exception:
    openai = None

load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    raise RuntimeError("SECRET_KEY must be set in the environment or .env")

ALGORITHM = os.getenv("ALGORITHM")
if not ALGORITHM:
    raise RuntimeError("ALGORITHM must be set in the environment or .env")

try:
    ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES"))
    if ACCESS_TOKEN_EXPIRE_MINUTES <= 0:
        raise ValueError("ACCESS_TOKEN_EXPIRE_MINUTES must be positive")
except (TypeError, ValueError):
    raise RuntimeError("ACCESS_TOKEN_EXPIRE_MINUTES must be a positive integer set in the environment or .env")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))


# Create JWT token
def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    if "sub" in to_encode:
        to_encode["sub"] = str(to_encode["sub"])
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def verify_access_token(token: str) -> Optional[int]:
    """Decode a JWT and return the user-id (``sub`` claim), or *None*."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        if user_id is None:
            return None
        return int(user_id)
    except (JWTError, ValueError):
        return None


# ── FastAPI dependency ────────────────────────────────

def get_current_user_id(token: str = Depends(oauth2_scheme)) -> int:
    """Dependency that extracts & validates the user-id from a Bearer token."""
    user_id = verify_access_token(token)
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user_id


def call_openai_for_stress(features: dict, model: str = "gpt-4o-mini", timeout: int = 10) -> dict:
    """Call an OpenAI-compatible chat API to obtain a stress prediction as strict JSON.

    Returns a dict with keys: `stress_level_pct` (float) and `is_stressed` (bool).
    Raises RuntimeError if the OpenAI client isn't configured or ValueError on parse errors.
    """
    if openai is None:
        raise RuntimeError("openai package not installed")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment")

    # Optional for providers like Groq: https://api.groq.com/openai/v1
    base_url = os.getenv("OPENAI_BASE_URL")
    model_name = os.getenv("OPENAI_MODEL", model)

    system = (
        "You estimate human stress level from physiological HRV features.\n"
        "Inputs include heart rate variability metrics, heart rate statistics, "
        "frequency-domain features, nonlinear HRV features, and demographics.\n\n"

        "Reason about autonomic nervous system balance and the relationships "
        "between variability, heart rate patterns, and signal stability.\n\n"

        "Perform reasoning internally but DO NOT output the reasoning.\n\n"

        "Carefully calculate the stress level, "
        "if metrics indicate high stress: feel free to use high values, "
        "and if they indicate low stress: feel free to use low values.\n\n"

        "Return ONLY valid JSON with exactly these fields:\n"
        "{\n"
        '  "stress_level_pct": float between 0 and 100,\n'
        '  "is_stressed": boolean\n'
        "}\n\n"

        "is_stressed must be true if stress_level_pct >= 50.\n"
        "No explanations, markdown, or extra text.\n"
        "It's very important to treat each prediction as an independent evaluation."
    )

    user_msg = (
        "Features JSON:\n" + json.dumps(features, separators=(",", ":")) + "\n\n"
        "Respond with valid JSON exactly as described. Example: {\"stress_level_pct\": 42.1, \"is_stressed\": false}"
    )

    # openai>=1.0 client API
    try:
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        client = openai.OpenAI(**client_kwargs)
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user_msg}],
            max_tokens=40,
            temperature = 0.5,
            top_p = 0.9,
            timeout=timeout,
        )
    except Exception as e:
        raise ValueError(f"OpenAI-compatible request failed: {e}")

    text = ""
    try:
        text = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        raise ValueError(f"Unexpected OpenAI response format: {e}")

    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError("OpenAI response did not contain a JSON object")

    parsed = json.loads(m.group(0))

    if "stress_level_pct" not in parsed or "is_stressed" not in parsed:
        raise ValueError("OpenAI JSON missing required keys")

    return {"stress_level_pct": float(parsed["stress_level_pct"]), "is_stressed": bool(parsed["is_stressed"])}