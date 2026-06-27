from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from api.dependencies import get_db
from schemas.schemas import UserRegister, UserLogin
from services.auth_service import AuthService

router = APIRouter(tags=["Auth"])

@router.post("/register", status_code=status.HTTP_201_CREATED)
def register(body: UserRegister, db: Session = Depends(get_db)):
    return AuthService.register(body, db)

@router.post("/login")
def login(body: UserLogin, db: Session = Depends(get_db)):
    return AuthService.login(body, db)

@router.post("/logout", status_code=status.HTTP_200_OK)
def logout():
    """JWT is stateless; the client clears the token."""
    return {"message": "Logged out successfully"}
