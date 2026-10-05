from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from fastapi import HTTPException, status
from db.models import User
from schemas.schemas import UserProfileUpdate

class ProfileService:
    @staticmethod
    def get_profile(user_id: int, db: Session):
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
            )
        return user

    @staticmethod
    def delete_account(user_id: int, db: Session):
        user = db.get(User, user_id)
        if user:
            db.delete(user)
            db.commit()

    @staticmethod
    def update_profile(user_id: int, body: UserProfileUpdate, db: Session):
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
            )
        if body.name is not None:
            user.name = body.name
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
                detail="Email already taken",
            )
        return user
