from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.orm import Session

from api.dependencies import get_db, get_current_user_id
from schemas.schemas import HeartRateResponse, HeartRateCreate, HeartRateBulkDelete
from services.heart_rate_service import HeartRateService

router = APIRouter(prefix="/heart-rate", tags=["Heart Rate"])

@router.post("", response_model=HeartRateResponse, status_code=status.HTTP_201_CREATED)
def create_heart_rate(
    entry: HeartRateCreate,
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    return HeartRateService.create_record(user_id, entry, db)

@router.get("", response_model=list[HeartRateResponse])
def list_heart_rate(
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db),
    limit: int = Query(500, ge=1, le=5000),
    offset: int = Query(0, ge=0),
):
    return HeartRateService.list_records(user_id, limit, offset, db)

@router.delete("/{entry_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_heart_rate(
    entry_id: str,
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    HeartRateService.delete_record(user_id, entry_id, db)

@router.post("/batch-delete", status_code=status.HTTP_200_OK)
def batch_delete_heart_rate(
    body: HeartRateBulkDelete,
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    return HeartRateService.batch_delete(user_id, body, db)
