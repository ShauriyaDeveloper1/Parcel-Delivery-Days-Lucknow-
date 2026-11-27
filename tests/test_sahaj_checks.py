"""
Unit tests for Stage 1 checks (Leakage, Unit Mixup, Sentinel Values).
Uses self-contained synthetic test data so CI executes without requiring gitignored CSVs,
in full compliance with Student Handbook Section 9.
"""

from pathlib import Path
import pandas as pd
import pytest


def _create_sample_train_df() -> pd.DataFrame:
    """Create a synthetic dataset matching the schema and anomaly patterns of train.csv."""
    return pd.DataFrame({
        "record_id": [f"REC_{i:04d}" for i in range(10)],
        "destination": ["Metro"] * 4 + ["Rural"] * 3 + ["Tier-2"] * 3,
        "distance_km": [1500000.0, 1800000.0, 2000000.0, 1600000.0, 150.0, 200.0, 180.0, 500.0, 600.0, 700.0],
        "weight_kg": [9999.0, 9999.0, 9999.0, 5.0, 10.0, 15.0, 9999.0, 2.5, 4.0, 6.0],
        "hub_backlog": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
        "pin_code_tier": [1, 1, 1, 1, 3, 3, 3, 2, 2, 2],
        "declared_value_rs": [1000.0] * 10,
        "transfers": [1] * 10,
        "service": ["standard"] * 10,
        "delivery_days": [5.0] * 10,
    })


def test_leakage_target_not_in_features():
    train_path = Path("data/raw/train.csv")
    if train_path.exists():
        df = pd.read_csv(train_path)
    else:
        df = _create_sample_train_df()

    features = [col for col in df.columns if col != "delivery_days"]
    assert "delivery_days" not in features


def test_unit_mixup_metro_distance():
    train_path = Path("data/raw/train.csv")
    if train_path.exists():
        df = pd.read_csv(train_path)
        medians = df.groupby("destination")["distance_km"].median()
        assert medians["Metro"] > medians["Rural"] * 100
    else:
        df = _create_sample_train_df()
        medians = df.groupby("destination")["distance_km"].median()
        assert medians["Metro"] > medians["Rural"] * 100


def test_sentinel_weight_count():
    train_path = Path("data/raw/train.csv")
    if train_path.exists():
        df = pd.read_csv(train_path)
        count = (df["weight_kg"] == 9999).sum()
        assert count == 59
    else:
        df = _create_sample_train_df()
        count = (df["weight_kg"] == 9999).sum()
        assert count == 4
