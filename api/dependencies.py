from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from sqlalchemy.orm import Session

from core.security import verify_access_token
from db.database import SessionLocal
from db.models import User

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_current_user_id(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> int:
    claims = verify_access_token(token)
    user = db.get(User, claims[0]) if claims else None
    # A token outlives a logout or a deleted account, so check it against the account.
    if user is None or user.token_version != claims[1]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user.id
