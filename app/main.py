"""
FastAPI prediction microservice for Lucknow Speed Parcel Hub (MP012).
"""

import json
import logging
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

# Setup structured JSON logging to stdout (never to a file, no PII)
logger = logging.getLogger("mp012_api")
logger.setLevel(logging.INFO)

if not logger.handlers:
    stream_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(message)s")
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.propagate = False


def log_json_event(event_type: str, data: Dict[str, Any]):
    """Emit a single-line JSON log to stdout."""
    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "event": event_type,
        **data,
    }
    logger.info(json.dumps(payload))


MODEL_PATH = Path("models/model.joblib")
model = None


def load_model_artifact():
    """Load model artifact from disk."""
    global model
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
    else:
        logger.warning(json.dumps({"event": "model_not_found", "path": str(MODEL_PATH)}))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    load_model_artifact()
    yield


app = FastAPI(
    title="Lucknow Parcel Delivery Days API",
    description="Inference microservice for predicting delivery duration (days).",
    version="1.0.0",
    lifespan=lifespan,
)


class PredictionRequest(BaseModel):
    """
    Client schema for POST /predict.
    Enforces the client's two operational rules via Pydantic Field:
      1. transfers <= 10 (transfers above 10 is impossible; status 422)
      2. declared_value_rs >= 50 (below 50 is always a typing mistake; status 422)
    """
    distance_km: float = Field(..., description="Distance to destination in km")
    weight_kg: float = Field(..., description="Parcel weight in kg")
    hub_backlog: int = Field(..., description="Parcels waiting at hub")
    pin_code_tier: int = Field(..., description="Delivery area tier")
    declared_value_rs: float = Field(..., ge=50.0, description="Declared value in INR (minimum 50)")
    transfers: int = Field(..., le=10, description="Hub transfers count (maximum 10)")
    service: str = Field(..., description="Service chosen: standard, express, economy")
    destination: str = Field(..., description="Destination type: Metro, Tier-2, Rural")


class PredictionResponse(BaseModel):
    prediction: float


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """Liveness and readiness health check probe."""
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
def predict(request: PredictionRequest):
    """
    Predict delivery days given parcel booking features.
    Logs structured JSON with latency and request payload to stdout.
    """
    start_time = time.perf_counter()

    if model is None:
        if MODEL_PATH.exists():
            load_model_artifact()
        else:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model artifact models/model.joblib is not available.",
            )

    input_data = request.model_dump()
    df_input = pd.DataFrame([input_data])

    try:
        raw_pred = model.predict(df_input)[0]
        prediction_val = float(round(raw_pred, 4))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference pipeline execution error: {str(e)}",
        )

    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

    # Structured JSON log per prediction to stdout (P10)
    log_json_event(
        event_type="prediction",
        data={
            "inputs": input_data,
            "prediction": prediction_val,
            "latency_ms": latency_ms,
        },
    )

    return PredictionResponse(prediction=prediction_val)
