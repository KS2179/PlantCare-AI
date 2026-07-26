from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class DiseaseInfo(BaseModel):
    plant: Optional[str] = None
    disease: Optional[str] = None
    is_healthy: bool = False
    symptoms: Optional[str] = None
    causes: Optional[str] = None
    prevention: Optional[str] = None
    treatment: Optional[str] = None


class TopKPrediction(BaseModel):
    label: str
    confidence: float


class PredictionResponse(BaseModel):
    id: int
    filename: str
    predicted_class: str
    confidence: float
    top_k: list[TopKPrediction]
    info: DiseaseInfo
    created_at: datetime

    class Config:
        from_attributes = True


class HistoryItem(BaseModel):
    id: int
    filename: str
    predicted_class: str
    confidence: float
    plant: Optional[str] = None
    disease: Optional[str] = None
    is_healthy: bool
    created_at: datetime

    class Config:
        from_attributes = True
