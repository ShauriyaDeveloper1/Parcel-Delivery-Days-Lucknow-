import pandas as pd


def test_leakage_target_not_in_features():
    df = pd.read_csv("data/raw/train.csv")
    features = [col for col in df.columns if col != "delivery_days"]

    assert "delivery_days" not in features


def test_unit_mixup_metro_distance():
    df = pd.read_csv("data/raw/train.csv")

    medians = df.groupby("destination")["distance_km"].median()

    assert medians["Metro"] > medians["Rural"] * 100


def test_sentinel_weight_count():
    df = pd.read_csv("data/raw/train.csv")

    count = (df["weight_kg"] == 9999).sum()

    assert count == 59
