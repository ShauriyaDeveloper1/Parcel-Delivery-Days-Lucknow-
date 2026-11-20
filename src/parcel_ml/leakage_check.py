
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

train = pd.read_csv(ROOT / "data/raw/train.csv")
holdout = pd.read_csv(ROOT / "data/raw/holdout_features.csv")

target = "delivery_days"

# Target is not a feature.
features = [c for c in train.columns if c != target]

# Check which training features are unavailable in holdout.
missing = [c for c in features if c not in holdout.columns]

print("LEAKAGE CHECK")
print("Training features:", features)
print("Missing from holdout:", missing)

if missing:
    print("Potential leakage columns:", missing)
else:
    print("No missing training feature found.")

# Check suspiciously high numeric correlations.
numeric = train.select_dtypes(include="number")
correlations = numeric.corr()[target].drop(target).abs()
suspicious = correlations[correlations >= 0.98]

print("\nHighly correlated features:")
print(suspicious)

if suspicious.empty:
    print("No numeric feature with correlation >= 0.98.")
