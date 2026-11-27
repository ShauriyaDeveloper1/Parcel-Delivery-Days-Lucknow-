"""
Tests for FastAPI prediction service using TestClient (Stage 3).
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    """GET /health must return status 200 and {'status': 'ok'}."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_predict_valid_request(client):
    """POST /predict with valid payload returns 200 and numeric prediction."""
    payload = {
        "distance_km": 450.0,
        "weight_kg": 3.2,
        "hub_backlog": 120,
        "pin_code_tier": 2,
        "declared_value_rs": 850.0,
        "transfers": 2,
        "service": "standard",
        "destination": "Tier-2",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert isinstance(data["prediction"], (int, float))
    assert data["prediction"] > 0


def test_predict_rejects_transfers_above_10(client):
    """Rule 1: transfers above 10 is impossible; must return HTTP 422."""
    payload = {
        "distance_km": 450.0,
        "weight_kg": 3.2,
        "hub_backlog": 120,
        "pin_code_tier": 2,
        "declared_value_rs": 850.0,
        "transfers": 11,  # Invalid: > 10
        "service": "standard",
        "destination": "Tier-2",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_rejects_declared_value_below_50(client):
    """Rule 2: declared_value_rs below 50 is a typing mistake; must return HTTP 422."""
    payload = {
        "distance_km": 450.0,
        "weight_kg": 3.2,
        "hub_backlog": 120,
        "pin_code_tier": 2,
        "declared_value_rs": 49.0,  # Invalid: < 50
        "transfers": 2,
        "service": "standard",
        "destination": "Tier-2",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
