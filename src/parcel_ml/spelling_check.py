
from pathlib import Path
import re
import pandas as pd


def clean_service(value):
    value = str(value).strip().lower()
    value = re.sub(r"[^a-z0-9]", "", value)

    aliases = {
        "exp": "express",
        "eco": "economy",
        "sta": "standard"
    }

    return aliases.get(value, value)


def check_spelling_variants(train_df: pd.DataFrame) -> dict:
    column = "service"

    if column not in train_df.columns:
        raise ValueError(f"Missing column: {column}")

    raw_values = train_df[column].dropna().astype(str)

    original_levels = raw_values.nunique()
    cleaned_values = raw_values.map(clean_service)
    true_levels = cleaned_values.nunique()

    present = original_levels > true_levels

    print("[Spelling Check] Column:", column)
    print("[Spelling Check] Original unique values:", original_levels)
    print("[Spelling Check] True categories after cleaning:", true_levels)

    if present:
        print("[Spelling Check] Detected variants:")
        print(raw_values.value_counts().to_string())

    return {
        "present": bool(present),
        "column": column if present else None,
        "true_levels": int(true_levels) if present else None
    }


if __name__ == "__main__":
    train_path = Path("data/raw/train.csv")
    train = pd.read_csv(train_path)

    result = check_spelling_variants(train)

    print("\nResult:", result)
