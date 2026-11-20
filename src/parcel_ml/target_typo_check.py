from pathlib import Path
import pandas as pd


def check_target_typos(train_df: pd.DataFrame) -> dict:
    target_col = "delivery_days"

    if target_col not in train_df.columns:
        raise ValueError(f"Missing target column: {target_col}")

    suspicious = train_df[train_df[target_col] > 20]
    count = len(suspicious)

    print("[Target Typo Check] Column:", target_col)
    print("[Target Typo Check] Suspicious values (>20 days):", count)

    if count > 0:
        print("[Target Typo Check] Possible 10x values:")
        print(suspicious[target_col].sort_values().to_list())

    return {
        "present": count > 0,
        "count": count if count > 0 else None
    }


if __name__ == "__main__":
    train_path = Path("data/raw/train.csv")
    train = pd.read_csv(train_path)
    result = check_target_typos(train)
    print("\nResult:", result)