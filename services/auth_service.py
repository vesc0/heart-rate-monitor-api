from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from fastapi import HTTPException, status
from db.models import User
from schemas.schemas import UserRegister, UserLogin
from core.security import hash_password, verify_password, create_access_token

class AuthService:
    @staticmethod
    def register(body: UserRegister, db: Session):
        user = User(
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
                detail="Email already registered",
            )
        token = create_access_token(data={"sub": user.id})
        return {
            "message": "User registered",
            "email": user.email,
            "name": user.name,
            "access_token": token,
            "token_type": "bearer",
        }

    @staticmethod
    def login(body: UserLogin, db: Session):
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
            "name": user.name,
            "email": user.email,
            "age": user.age,
            "gender": user.gender,
            "height_cm": user.height_cm,
            "weight_kg": user.weight_kg,
            "health_issues": user.health_issues,
        }
