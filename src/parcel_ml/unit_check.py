
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

df = pd.read_csv(ROOT / "data/raw/train.csv")

print("UNIT MIXUP CHECK")

for col in ["distance_km", "weight_kg", "hub_backlog"]:
    print(f"\nChecking: {col}")

    stats = df.groupby("destination")[col].agg(
        ["min", "median", "mean", "max"]
    )

    print(stats)

print("\nExpected finding to investigate:")
print("Check whether Metro distance values are around 1000 times larger.")
