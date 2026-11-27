from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="Parcel Delivery Days API")


class PredictionRequest(BaseModel):
    distance_km: float
    weight_kg: float
    hub_backlog: float
    pin_code_tier: int
    declared_value_rs: float = Field(ge=50)
    transfers: int = Field(ge=0, le=10)
    service: str
    destination: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictionRequest):
    return {
        "message": "Request received successfully",
        "input": request.model_dump()
    }