from fastapi import APIRouter, Depends, Request, status
from sqlalchemy.orm import Session

from api.dependencies import get_db, get_current_user_id
from core.limiter import limiter
from schemas.schemas import UserRegister, UserLogin
from services.auth_service import AuthService

router = APIRouter(tags=["Auth"])

@router.post("/register", status_code=status.HTTP_201_CREATED)
@limiter.limit("10/minute")
def register(request: Request, body: UserRegister, db: Session = Depends(get_db)):
    return AuthService.register(body, db)

@router.post("/login")
@limiter.limit("10/minute")
def login(request: Request, body: UserLogin, db: Session = Depends(get_db)):
    return AuthService.login(body, db)

@router.post("/logout", status_code=status.HTTP_200_OK)
def logout(
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    AuthService.logout(user_id, db)
    return {"message": "Logged out successfully"}
