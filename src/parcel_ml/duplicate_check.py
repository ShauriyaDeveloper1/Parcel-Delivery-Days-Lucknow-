from pathlib import Path
import pandas as pd


def _get_train_path() -> str:
    try:
        from src.parcel_ml import load_params
        params = load_params()
        return params.get("data", {}).get("train_path", "data/raw/train.csv")
    except Exception:
        return "data/raw/train.csv"


def check_duplicate_rows(train_df: pd.DataFrame) -> dict:
    exact_extra_copies = int(train_df.duplicated().sum())
    feature_cols = [c for c in train_df.columns if c != "record_id"]
    content_extra_copies = int(train_df.duplicated(subset=feature_cols).sum())

    print(f"[Duplicate Check] Total rows: {len(train_df)}")
    print(f"[Duplicate Check] Exact duplicate extra rows (df.duplicated().sum()): {exact_extra_copies}")
    print(f"[Duplicate Check] Duplicate rows ignoring record_id: {content_extra_copies}")

    if exact_extra_copies > 0:
        print("Sample duplicate rows:")
        print(train_df[train_df.duplicated(keep=False)].head(6).to_string())
        return {"present": True, "count": exact_extra_copies}

    return {"present": False, "count": None}


if __name__ == "__main__":
    train_path = Path(_get_train_path())
    train = pd.read_csv(train_path)
    result = check_duplicate_rows(train)
    print("\nResult:", result)
