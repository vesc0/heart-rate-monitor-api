from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from fastapi import HTTPException, status
from datetime import timezone
import uuid

from db.models import HeartRateRecord
from schemas.schemas import HeartRateCreate, HeartRateBulkDelete

class HeartRateService:
    @staticmethod
    def create_record(user_id: int, entry: HeartRateCreate, db: Session):
        rec_dt = entry.recorded_at
        if rec_dt.tzinfo is None:
            rec_dt = rec_dt.replace(tzinfo=timezone.utc)
        else:
            rec_dt = rec_dt.astimezone(timezone.utc)

        record = HeartRateRecord(
            id=entry.id or str(uuid.uuid4()),
            user_id=user_id,
            bpm=entry.bpm,
            recorded_at=rec_dt,
            stress_level=entry.stress_level,
            stress_explanation=entry.stress_explanation,
            activity_state=entry.activity_state,
        )
        db.add(record)
        try:
            db.commit()
            db.refresh(record)
        except IntegrityError:
            db.rollback()
            existing = (
                db.query(HeartRateRecord)
                .filter(HeartRateRecord.id == record.id, HeartRateRecord.user_id == user_id)
                .first()
            )
            if existing:
                existing.bpm = entry.bpm
                existing.recorded_at = rec_dt
                existing.stress_level = entry.stress_level
                existing.stress_explanation = entry.stress_explanation
                existing.activity_state = entry.activity_state
                db.commit()
                db.refresh(existing)
                return existing
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT, detail="Duplicate entry"
            )
        return record

    @staticmethod
    def list_records(user_id: int, limit: int, offset: int, db: Session):
        return (
            db.query(HeartRateRecord)
            .filter(HeartRateRecord.user_id == user_id)
            .order_by(HeartRateRecord.recorded_at.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )

    @staticmethod
    def delete_record(user_id: int, entry_id: str, db: Session):
        record = (
            db.query(HeartRateRecord)
            .filter(HeartRateRecord.id == entry_id, HeartRateRecord.user_id == user_id)
            .first()
        )
        if not record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Record not found"
            )
        db.delete(record)
        db.commit()

    @staticmethod
    def batch_delete(user_id: int, body: HeartRateBulkDelete, db: Session):
        deleted = (
            db.query(HeartRateRecord)
            .filter(
                HeartRateRecord.id.in_(body.ids),
                HeartRateRecord.user_id == user_id,
            )
            .delete(synchronize_session=False)
        )
        db.commit()
        return {"deleted": deleted}
