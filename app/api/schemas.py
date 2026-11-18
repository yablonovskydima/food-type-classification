from pydantic import BaseModel, Field


class PredictionItem(BaseModel):
    category: str = Field(..., serialization_alias="class")
    probability: float
    index: int


class PredictResponse(BaseModel):
    filename: str
    predictions: list[PredictionItem]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    checkpoint: str
    error: str | None = None
