from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
from jose import JWTError, jwt

from core.config import settings

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))

def create_access_token(user_id: int, token_version: int) -> str:
    claims = {
        "sub": str(user_id),
        "ver": token_version,
        "exp": datetime.now(timezone.utc) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES),
    }
    return jwt.encode(claims, settings.SECRET_KEY, algorithm=settings.ALGORITHM)

def verify_access_token(token: str) -> Optional[tuple[int, int]]:
    """Return the token's user id and version, or None if it is invalid."""
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        # Tokens issued before versioning have no "ver" and count as version 0.
        return int(payload["sub"]), int(payload.get("ver", 0))
    except (JWTError, KeyError, TypeError, ValueError):
        return None
