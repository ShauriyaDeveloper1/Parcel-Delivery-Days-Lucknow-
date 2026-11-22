"""
Unit and integration tests for Stage 4 drift monitoring, retraining, and holdout predictions.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import yaml
import joblib

from src.parcel_ml.monitor import calculate_psi


def test_psi_zero_drift():
    # Comparing identical distributions should yield PSI ~ 0
    s1 = pd.Series(np.random.normal(10, 2, 1000))
    s2 = pd.Series(np.random.normal(10, 2, 1000))
    psi = calculate_psi(s1, s2, num_bins=10)
    assert psi < 0.1, f"Expected near-zero PSI for identical distributions, got {psi}"


def test_psi_significant_drift():
    # Comparing distinctly shifted distributions should yield high PSI (> 0.25)
    s1 = pd.Series(np.random.normal(10, 2, 1000))
    s2 = pd.Series(np.random.normal(25, 2, 1000))
    psi = calculate_psi(s1, s2, num_bins=10)
    assert psi > 0.25, f"Expected significant PSI (>0.25) for shifted distributions, got {psi}"


def test_findings_yaml_monitoring_section():
    findings_path = Path("findings.yaml")
    assert findings_path.exists(), "findings.yaml missing"
    with open(findings_path, "r") as f:
        data = yaml.safe_load(f)

    monitoring = data.get("monitoring", {})
    assert monitoring is not None, "monitoring section is missing"

    # All PSI features must be populated numeric values
    psi_dict = monitoring.get("psi", {})
    required_psi_cols = [
        "distance_km",
        "weight_kg",
        "hub_backlog",
        "pin_code_tier",
        "declared_value_rs",
        "transfers",
    ]
    for col in required_psi_cols:
        val = psi_dict.get(col)
        assert val is not None, f"PSI for {col} is null"
        assert isinstance(val, (int, float)), f"PSI for {col} must be numeric"

    assert monitoring.get("most_drifted_feature") == "transfers"
    assert monitoring.get("worst_served_group") == "Metro"
    assert monitoring.get("decision") in ["promote", "keep"]

    # Mae by group must have Metro, Tier-2, Rural
    mae_grp = monitoring.get("mae_by_group", {})
    for grp in ["Metro", "Tier-2", "Rural"]:
        assert mae_grp.get(grp) is not None, f"MAE for group {grp} is null"

    assert monitoring.get("mae_v1_on_check") is not None
    assert monitoring.get("mae_v2_on_check") is not None


def test_holdout_predictions_file():
    pred_path = Path("predictions/holdout_predictions.csv")
    assert pred_path.exists(), "predictions/holdout_predictions.csv does not exist"

    df = pd.read_csv(pred_path)
    assert df.shape == (250, 2), f"Expected shape (250, 2), got {df.shape}"
    assert list(df.columns) == ["record_id", "prediction"]
    assert df["record_id"].nunique() == 250, "Duplicate record_ids in predictions"
    assert df["prediction"].isnull().sum() == 0, "Null predictions found"
    assert (df["prediction"] > 0).all(), "Predictions must be positive numbers"


def test_champion_v2_model_prediction():
    model_path = Path("models/model.joblib")
    assert model_path.exists(), "models/model.joblib missing"
    model = joblib.load(model_path)

    sample = pd.DataFrame([{
        "distance_km": 1960.0,
        "weight_kg": 15.0,
        "hub_backlog": 500,
        "pin_code_tier": 1,
        "declared_value_rs": 5000.0,
        "transfers": 8,
        "service": "standard",
        "destination": "Metro",
    }])

    pred = model.predict(sample)
    assert len(pred) == 1
    assert 2.0 < float(pred[0]) < 25.0
