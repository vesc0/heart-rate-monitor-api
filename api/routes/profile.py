from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from api.dependencies import get_db, get_current_user_id
from schemas.schemas import UserProfile, UserProfileUpdate
from services.profile_service import ProfileService

router = APIRouter(tags=["Profile"])

@router.get("/me", response_model=UserProfile)
def get_profile(
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    return ProfileService.get_profile(user_id, db)

@router.put("/me", response_model=UserProfile)
def update_profile(
    body: UserProfileUpdate,
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    return ProfileService.update_profile(user_id, body, db)

@router.delete("/me", status_code=status.HTTP_204_NO_CONTENT)
def delete_account(
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    ProfileService.delete_account(user_id, db)
