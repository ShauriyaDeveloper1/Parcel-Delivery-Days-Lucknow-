
from pathlib import Path

import pandas as pd
import pytest


DATA_PATH = Path("data/raw/train.csv")


@pytest.fixture
def train_df():
    if not DATA_PATH.exists():
        pytest.skip("Raw training dataset is not available in this environment")

    return pd.read_csv(DATA_PATH)


def test_leakage_target_not_in_features(train_df):
    features = [col for col in train_df.columns if col != "delivery_days"]

    assert "delivery_days" not in features


def test_unit_mixup_metro_distance(train_df):
    medians = train_df.groupby("destination")["distance_km"].median()

    assert medians["Metro"] > medians["Rural"] * 100


def test_sentinel_weight_count(train_df):
    count = (train_df["weight_kg"] == 9999).sum()

    assert count == 59
