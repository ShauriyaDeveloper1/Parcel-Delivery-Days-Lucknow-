"""
Unit and integration tests for Stage 2 data cleaning, preprocessing, and model pipeline.
"""

import numpy as np
import pandas as pd
import pytest
import joblib

from src.parcel_ml.cleaning import clean_dataset, clean_service_value
from src.parcel_ml.preprocessing import build_preprocessor, prepare_features, NUMERIC_FEATURES, CATEGORICAL_FEATURES


def test_clean_service_value():
    assert clean_service_value("Express") == "express"
    assert clean_service_value(" EXPRESS ") == "express"
    assert clean_service_value("exp.") == "express"
    assert clean_service_value("eco.") == "economy"
    assert clean_service_value("sta.") == "standard"
    assert clean_service_value("standard") == "standard"


def test_clean_dataset_unit_mixup():
    df = pd.DataFrame({
        "destination": ["Metro", "Rural", "Tier-2"],
        "distance_km": [1500000.0, 1500.0, 1200.0],
        "weight_kg": [5.0, 10.0, 15.0],
        "hub_backlog": [100, 200, 300],
        "date": ["2026-02-01", "2026-02-01", "2026-02-01"],
        "service": ["express", "economy", "standard"],
        "delivery_days": [5.5, 6.0, 7.0],
    })
    cleaned = clean_dataset(df, is_train=True)
    assert cleaned.loc[0, "distance_km"] == 1500.0
    assert cleaned.loc[1, "distance_km"] == 1500.0
    assert cleaned.loc[2, "distance_km"] == 1200.0


def test_clean_dataset_sentinels_and_stuck_sensor():
    df = pd.DataFrame({
        "destination": ["Rural", "Rural"],
        "distance_km": [500.0, 600.0],
        "weight_kg": [9999.0, 12.5],
        "hub_backlog": [2011, 500],
        "date": ["2026-05-01", "2026-06-01"],
        "service": ["standard", "standard"],
        "delivery_days": [6.0, 6.5],
    })
    cleaned = clean_dataset(df, is_train=True)
    assert pd.isna(cleaned.loc[0, "weight_kg"])
    assert cleaned.loc[1, "weight_kg"] == 12.5
    assert pd.isna(cleaned.loc[0, "hub_backlog"])
    assert cleaned.loc[1, "hub_backlog"] == 500


def test_clean_dataset_target_typos():
    df = pd.DataFrame({
        "destination": ["Rural", "Rural"],
        "distance_km": [500.0, 600.0],
        "weight_kg": [10.0, 12.5],
        "hub_backlog": [500, 500],
        "date": ["2026-02-01", "2026-02-01"],
        "service": ["standard", "standard"],
        "delivery_days": [64.1, 7.2],
    })
    cleaned = clean_dataset(df, is_train=True)
    assert abs(cleaned.loc[0, "delivery_days"] - 6.41) < 1e-4
    assert cleaned.loc[1, "delivery_days"] == 7.2


def test_no_target_leakage_in_preprocessing():
    df = pd.DataFrame({
        "record_id": ["R001", "R002"],
        "date": ["2026-01-01", "2026-01-02"],
        "distance_km": [500.0, 1200.0],
        "weight_kg": [np.nan, 15.0],
        "hub_backlog": [np.nan, 300],
        "pin_code_tier": [1, 2],
        "declared_value_rs": [1000.0, 2500.0],
        "transfers": [1, 3],
        "service": ["express", "economy"],
        "destination": ["Metro", "Rural"],
        "delivery_days": [4.5, 7.8],
    })

    X, y = prepare_features(df, target_col="delivery_days")
    assert "delivery_days" not in X.columns
    assert "record_id" not in X.columns
    assert "date" not in X.columns
    assert len(y) == 2

    preprocessor = build_preprocessor()
    X_trans = preprocessor.fit_transform(X)
    assert X_trans.shape[0] == 2
    assert not np.isnan(X_trans).any(), "Imputer failed to resolve NaNs in features."


def test_champion_model_prediction():
    model = joblib.load("models/model.joblib")
    sample = pd.DataFrame([{
        "distance_km": 650.0,
        "weight_kg": 8.5,
        "hub_backlog": 120,
        "pin_code_tier": 2,
        "declared_value_rs": 1500.0,
        "transfers": 1,
        "service": "standard",
        "destination": "Tier-2",
    }])

    prediction = model.predict(sample)
    assert len(prediction) == 1
    assert 0.0 < float(prediction[0]) < 30.0, "Prediction value outside expected delivery days range."
