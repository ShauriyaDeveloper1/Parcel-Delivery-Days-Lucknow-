import pandas as pd
from src.parcel_ml.duplicate_check import check_duplicate_rows
from src.parcel_ml.stuck_sensor_check import check_stuck_sensor


def _sample_df() -> pd.DataFrame:
    dates = [f"2026-01-{d:02d}" for d in range(1, 21)]
    return pd.DataFrame({
        "record_id": [f"R{i:03d}" for i in range(20)],
        "date": dates,
        "distance_km": [200 + i * 5 for i in range(20)],
        "weight_kg": [1.5 + (i % 5) * 0.4 for i in range(20)],
        "hub_backlog": [100 + i * 3 for i in range(20)],
        "declared_value_rs": [500 + i * 20 for i in range(20)],
    })


def test_no_duplicates():
    df = _sample_df()
    assert check_duplicate_rows(df) == {"present": False, "count": None}


def test_with_duplicates():
    df = _sample_df()
    df = pd.concat([df, df.iloc[:2]], ignore_index=True)
    assert check_duplicate_rows(df) == {"present": True, "count": 2}


def test_stuck_sensor():
    df = _sample_df()
    df.loc[2:14, "hub_backlog"] = 2011
    assert check_stuck_sensor(df) == {
        "present": True,
        "column": "hub_backlog",
        "start_date": "2026-01-03",
        "end_date": "2026-01-15",
    }
