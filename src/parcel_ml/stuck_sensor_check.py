from pathlib import Path
import pandas as pd


def _get_settings() -> tuple[str, str, int]:
    try:
        from src.parcel_ml import load_params
        params = load_params()
        train_path = params.get("data", {}).get("train_path", "data/raw/train.csv")
        date_col = params.get("data", {}).get("date_col", "date")
        min_run = int(params.get("checks", {}).get("stuck_min_run", 10))
        return train_path, date_col, min_run
    except Exception:
        return "data/raw/train.csv", "date", 10


def check_stuck_sensor(
    train_df: pd.DataFrame,
    date_col: str = "date",
    min_run: int = 10,
) -> dict:
    df_sorted = train_df.sort_values(by=date_col, kind="mergesort").reset_index(drop=True)
    candidate_cols = [
        c for c in ["distance_km", "weight_kg", "hub_backlog", "declared_value_rs"]
        if c in df_sorted.columns
    ]

    best_run = None

    for col in candidate_cols:
        series = df_sorted[col]
        run_ids = series.ne(series.shift()).cumsum()
        grouped = df_sorted.groupby(run_ids).agg(
            value=(col, "first"),
            run_length=(col, "size"),
            start_date=(date_col, "first"),
            end_date=(date_col, "last"),
            distinct_days=(date_col, "nunique"),
        )
        longest = grouped.sort_values(by="run_length", ascending=False).iloc[0]
        print(
            f"[Stuck Sensor Check] '{col}': longest_run={longest['run_length']} rows "
            f"({longest['distinct_days']} days), value={longest['value']}, "
            f"range={longest['start_date']} to {longest['end_date']}"
        )

        if int(longest["run_length"]) >= min_run and int(longest["distinct_days"]) >= 5:
            if best_run is None or int(longest["run_length"]) > best_run["run_length"]:
                best_run = {
                    "run_length": int(longest["run_length"]),
                    "present": True,
                    "column": col,
                    "start_date": str(longest["start_date"]),
                    "end_date": str(longest["end_date"]),
                }

    if best_run is None:
        return {"present": False, "column": None, "start_date": None, "end_date": None}

    return {
        "present": True,
        "column": best_run["column"],
        "start_date": best_run["start_date"],
        "end_date": best_run["end_date"],
    }


if __name__ == "__main__":
    train_path, date_col, min_run = _get_settings()
    train = pd.read_csv(Path(train_path))
    result = check_stuck_sensor(train, date_col=date_col, min_run=min_run)
    print("\nResult:", result)
