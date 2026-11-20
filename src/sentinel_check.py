import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

df = pd.read_csv(ROOT / "data/raw/train.csv")

print("SENTINEL VALUE CHECK")

for col in ["distance_km", "weight_kg", "hub_backlog"]:
    print(f"\nChecking: {col}")
    print(df[col].value_counts().head(10))

sentinel_count = (df["weight_kg"] == 9999).sum()

print("\nSentinel value investigation:")
print("Column: weight_kg")
print("Sentinel value: 9999")
print("Count:", sentinel_count)
