"""
API Data Schemas.
Defines Pydantic request and response models for strict API contracts.
Responsible Team Member: Member 4 (Backend API & Service Layer)
"""

from pydantic import BaseModel
from typing import Optional, Dict, Any

class PredictionRequest(BaseModel):
    include_explainability: bool = True

class PredictionResponse(BaseModel):
    filename: str
    is_ai_generated: bool
    confidence_score: float
    label: str
    explainability: Optional[Dict[str, Any]] = None
