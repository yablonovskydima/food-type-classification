import io
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Query, UploadFile, status
from PIL import Image, UnidentifiedImageError

from app.api.model import create_predictor
from app.api.schemas import HealthResponse, PredictionItem, PredictResponse
from src.inference.predictor import FoodPredictor

predictor: FoodPredictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    predictor = create_predictor()

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
        model_loaded=predictor is not None and predictor.model is not None,
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(
    file: Annotated[UploadFile, File(...)],
    top_k: Annotated[int, Query(ge=1, le=11)] = 3,
) -> PredictResponse:
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is required",
        )

    content = await file.read()
    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty",
        )

    try:
        image = Image.open(io.BytesIO(content))
        image.verify()
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image file or unsupported image format",
        ) from None

    current_predictor = predictor
    if current_predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded"
        )

    raw_results = current_predictor.predict_image(image, top_k=top_k)

    predictions = [
        PredictionItem(
            category=item["class"],
            probability=item["probability"],
            index=item["index"],
        )
        for item in raw_results
    ]

    return PredictResponse(filename=file.filename, predictions=predictions)
