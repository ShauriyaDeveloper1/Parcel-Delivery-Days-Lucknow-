"""
Data Cleaning Module for Lucknow Parcel Delivery Hub (MP012).
Applies verified cleaning transformations discovered during Stage 1.
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd


def clean_service_value(val: str) -> str:
    """Standardizes spelling variants for service column into 3 true categories."""
    if pd.isna(val):
        return "standard"
    s = str(val).strip().lower()
    s = re.sub(r"[^a-z0-9]", "", s)
    aliases = {
        "exp": "express",
        "eco": "economy",
        "sta": "standard",
    }
    return aliases.get(s, s)


def clean_dataset(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """
    Cleans raw dataframe based on verified Stage 1 findings:
    1. Unit mixup: Metro distance is in meters, divide by 1000 to convert to km.
    2. Sentinel values: weight_kg == 9999 converted to NaN.
    3. Stuck sensor: hub_backlog == 2011 between 2026-04-29 and 2026-05-14 converted to NaN.
    4. Target typos: delivery_days > 20 divided by 10.0 (in training data).
    5. Spelling variants: service strings normalized to express, economy, standard.
    6. Exact duplicates: removed if any.
    """
    cleaned = df.copy()

    # 1. Unit mixup: Metro distance recorded in meters instead of km
    if "distance_km" in cleaned.columns and "destination" in cleaned.columns:
        metro_mask = cleaned["destination"] == "Metro"
        mixup_mask = metro_mask & (cleaned["distance_km"] > 2500)
        n_unit = mixup_mask.sum()
        if n_unit > 0:
            cleaned.loc[mixup_mask, "distance_km"] = cleaned.loc[mixup_mask, "distance_km"] / 1000.0
            print(f"[Cleaning] Fixed {n_unit} Metro records with distance in meters (/ 1000).")

    # 2. Sentinel value: weight_kg == 9999 represents missing reading
    if "weight_kg" in cleaned.columns:
        sentinel_mask = cleaned["weight_kg"] == 9999
        n_sentinel = sentinel_mask.sum()
        if n_sentinel > 0:
            cleaned.loc[sentinel_mask, "weight_kg"] = np.nan
            print(f"[Cleaning] Converted {n_sentinel} sentinel values (9999) in weight_kg to NaN.")

    # 3. Stuck sensor: hub_backlog frozen at 2011 between 2026-04-29 and 2026-05-14
    if "hub_backlog" in cleaned.columns and "date" in cleaned.columns:
        stuck_mask = (
            (cleaned["date"] >= "2026-04-29")
            & (cleaned["date"] <= "2026-05-14")
            & (cleaned["hub_backlog"] == 2011)
        )
        n_stuck = stuck_mask.sum()
        if n_stuck > 0:
            cleaned.loc[stuck_mask, "hub_backlog"] = np.nan
            print(f"[Cleaning] Converted {n_stuck} stuck sensor values (2011) in hub_backlog to NaN.")

    # 4. Target typos: delivery_days typed 10x too large (> 20)
    if is_train and "delivery_days" in cleaned.columns:
        typo_mask = cleaned["delivery_days"] > 20
        n_typos = typo_mask.sum()
        if n_typos > 0:
            cleaned.loc[typo_mask, "delivery_days"] = cleaned.loc[typo_mask, "delivery_days"] / 10.0
            print(f"[Cleaning] Corrected {n_typos} target typos in delivery_days (/ 10.0).")

    # 5. Spelling variants in service
    if "service" in cleaned.columns:
        before_levels = cleaned["service"].nunique()
        cleaned["service"] = cleaned["service"].apply(clean_service_value)
        after_levels = cleaned["service"].nunique()
        print(f"[Cleaning] Standardized service categories from {before_levels} variants to {after_levels} levels.")

    # 6. Duplicates
    initial_len = len(cleaned)
    cleaned = cleaned.drop_duplicates()
    n_dupes = initial_len - len(cleaned)
    if n_dupes > 0:
        print(f"[Cleaning] Dropped {n_dupes} duplicate rows.")

    return cleaned


def main():
    raw_path = Path("data/raw/train.csv")
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cleaned_train.csv"

    print(f"Reading raw training data from {raw_path}...")
    df = pd.read_csv(raw_path)
    print(f"Initial shape: {df.shape}")

    cleaned_df = clean_dataset(df, is_train=True)
    print(f"Cleaned shape: {cleaned_df.shape}")

    cleaned_df.to_csv(out_path, index=False)
    print(f"Successfully saved cleaned training data to {out_path}")


if __name__ == "__main__":
    main()
