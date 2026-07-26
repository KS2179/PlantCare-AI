import uuid
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session

from app.config import UPLOAD_DIR
from app.models.db import get_db, PredictionRecord
from app.schemas.prediction import PredictionResponse, DiseaseInfo, TopKPrediction
from app.services import inference, disease_info as disease_info_service

router = APIRouter(prefix="/api", tags=["prediction"])

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_UPLOAD_BYTES = 8 * 1024 * 1024  # 8 MB


@router.post("/predict", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}")

    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="File too large (max 8MB).")

    try:
        result = inference.predict(raw)
    except inference.ModelNotAvailableError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    info = disease_info_service.get_disease_info(result["predicted_class"])

    # Persist the original upload so predictions can be reviewed later.
    ext = Path(file.filename or "upload.jpg").suffix or ".jpg"
    stored_name = f"{uuid.uuid4().hex}{ext}"
    (UPLOAD_DIR / stored_name).write_bytes(raw)

    record = PredictionRecord(
        filename=stored_name,
        predicted_class=result["predicted_class"],
        confidence=result["confidence"],
        plant=info.get("plant"),
        disease=info.get("disease"),
        is_healthy=1 if info.get("is_healthy") else 0,
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    return PredictionResponse(
        id=record.id,
        filename=record.filename,
        predicted_class=record.predicted_class,
        confidence=record.confidence,
        top_k=[TopKPrediction(**k) for k in result["top_k"]],
        info=DiseaseInfo(**info),
        created_at=record.created_at,
    )
