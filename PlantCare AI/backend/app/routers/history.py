from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.models.db import get_db, PredictionRecord
from app.schemas.prediction import HistoryItem
from app.services import disease_info as disease_info_service
from app.schemas.prediction import PredictionResponse, DiseaseInfo, TopKPrediction

router = APIRouter(prefix="/api", tags=["history"])


@router.get("/history", response_model=list[HistoryItem])
def list_history(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    records = (
        db.query(PredictionRecord)
        .order_by(PredictionRecord.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    return [
        HistoryItem(
            id=r.id,
            filename=r.filename,
            predicted_class=r.predicted_class,
            confidence=r.confidence,
            plant=r.plant,
            disease=r.disease,
            is_healthy=bool(r.is_healthy),
            created_at=r.created_at,
        )
        for r in records
    ]


@router.get("/history/{record_id}", response_model=PredictionResponse)
def get_history_item(record_id: int, db: Session = Depends(get_db)):
    record = db.query(PredictionRecord).filter(PredictionRecord.id == record_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Record not found")

    info = disease_info_service.get_disease_info(record.predicted_class)
    return PredictionResponse(
        id=record.id,
        filename=record.filename,
        predicted_class=record.predicted_class,
        confidence=record.confidence,
        top_k=[TopKPrediction(label=record.predicted_class, confidence=record.confidence)],
        info=DiseaseInfo(**info),
        created_at=record.created_at,
    )


@router.delete("/history/{record_id}", status_code=204)
def delete_history_item(record_id: int, db: Session = Depends(get_db)):
    record = db.query(PredictionRecord).filter(PredictionRecord.id == record_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Record not found")
    db.delete(record)
    db.commit()
