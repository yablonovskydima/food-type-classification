import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Query, UploadFile, status

from app.api.schemas import HealthResponse, PredictionItem, PredictResponse
from src.inference.checkpoints import DEFAULT_CHECKPOINT
from src.inference.images import decode_uploaded_image
from src.inference.predictor import FoodPredictor

predictor: FoodPredictor | None = None
load_error: str | None = None


def checkpoint_path() -> Path:
    return Path(os.getenv("CHECKPOINT_PATH", str(DEFAULT_CHECKPOINT)))


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor, load_error
    try:
        predictor, load_error = FoodPredictor(checkpoint_path=checkpoint_path()), None
    except (OSError, ValueError, RuntimeError) as error:
        predictor, load_error = None, str(error)
    yield
    predictor = None


app = FastAPI(
    title="Food-11 Image Classification API",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=predictor is not None,
        checkpoint=str(checkpoint_path()),
        error=load_error,
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(
    file: Annotated[UploadFile, File(...)],
    top_k: Annotated[int, Query(ge=1, le=11)] = 3,
) -> PredictResponse:
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"No model loaded: {load_error}",
        )
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is required",
        )
    try:
        image = decode_uploaded_image(await file.read())
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from None

    predictions = [
        PredictionItem(category=item["class"], probability=item["probability"], index=item["index"])
        for item in predictor.predict_image(image, top_k=top_k)
    ]
    return PredictResponse(filename=file.filename, predictions=predictions)
