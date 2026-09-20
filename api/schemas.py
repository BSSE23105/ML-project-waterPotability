# Request and response shapes for the API

from typing import Optional
from pydantic import BaseModel, Field


class WaterSample(BaseModel):
    """The 9 chemical readings we need from the user."""
    ph: float = Field(..., ge=0, le=14, description="pH value (0-14)")
    Hardness: float = Field(..., ge=0, description="Hardness (mg/L)")
    Solids: float = Field(..., ge=0, description="Total Dissolved Solids (mg/L)")
    Chloramines: float = Field(..., ge=0, description="Chloramines (mg/L)")
    Sulfate: float = Field(..., ge=0, description="Sulfate (mg/L)")
    Conductivity: float = Field(..., ge=0, description="Conductivity (uS/cm)")
    Organic_carbon: float = Field(..., ge=0, description="Organic Carbon (mg/L)")
    Trihalomethanes: float = Field(..., ge=0, description="Trihalomethanes (ppm)")
    Turbidity: float = Field(..., ge=0, description="Turbidity (NTU)")


class BatchRequest(BaseModel):
    """A list of samples to score in one call."""
    samples: list[WaterSample] = Field(..., min_length=1, max_length=1000)
    explain: bool = True


class GuidelineWarning(BaseModel):
    feature: str
    value: float
    limit: str
    message: str


class Reason(BaseModel):
    feature: str
    value: float
    contribution: float
    direction: str


class PredictionResponse(BaseModel):
    id: Optional[int] = None
    prediction: int
    result: str
    confidence: float
    probability_potable: float
    input: dict
    warnings: list[GuidelineWarning] = []
    reasons: list[Reason] = []


class BatchSummary(BaseModel):
    total: int
    potable: int
    not_potable: int
    unsafe_share: float
    samples_with_warnings: int


class BatchResponse(BaseModel):
    summary: BatchSummary
    results: list[PredictionResponse]


class HistoryStats(BaseModel):
    total: int
    potable: int
    not_potable: int
    unsafe_share: float
    average_probability_potable: float
    most_common_warnings: list[dict]
    predictions_per_day: list[dict]
