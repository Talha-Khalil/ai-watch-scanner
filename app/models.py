"""
Pydantic models for request/response validation
"""
from pydantic import BaseModel
from typing import Optional, List


class WatchCandidate(BaseModel):
    name: Optional[str] = None
    brand: Optional[str] = None
    model: Optional[str] = None
    ref: Optional[str] = None
    mvmt: Optional[str] = None
    bracem: Optional[str] = None
    yop: Optional[str] = None
    price: Optional[str] = None
    similarity_score: Optional[float] = None


class ScanResponse(BaseModel):
    # Core identification
    identified_brand: str
    identified_model: str
    identified_ref: Optional[str] = None

    # Visual features extracted from image
    visual_features: str

    # Price
    projected_price: str
    price_range: Optional[str] = None
    price_basis: Optional[str] = None   # e.g. "Based on 3 matching records"

    # Confidence
    confidence: str  # High / Medium / Low
    confidence_reason: str

    # Movement & details
    movement: Optional[str] = None
    bracelet: Optional[str] = None
    year_of_production: Optional[str] = None

    # Top candidates from vector search
    top_candidates: List[WatchCandidate] = []

    # Full analysis narrative
    analysis: str


class HealthResponse(BaseModel):
    status: str
    watches_indexed: int