"""
Preprocessing Module for Lucknow Parcel Delivery Hub (MP012).
Constructs scikit-learn pipelines with zero data leakage.
"""

from typing import Tuple, Optional
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


NUMERIC_FEATURES = [
    "distance_km",
    "weight_kg",
    "hub_backlog",
    "pin_code_tier",
    "declared_value_rs",
    "transfers",
]

CATEGORICAL_FEATURES = [
    "service",
    "destination",
]


def build_preprocessor() -> ColumnTransformer:
    """
    Constructs a ColumnTransformer that imputes and scales numeric features,
    and imputes and one-hot encodes categorical features.
    Fit strictly on training data to avoid leakage.
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    return preprocessor


def prepare_features(
    df: pd.DataFrame, target_col: Optional[str] = "delivery_days"
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Extracts features and target, ensuring ID/date columns are excluded.
    """
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required feature columns: {missing_cols}")

    X = df[feature_cols].copy()
    y = df[target_col].copy() if target_col and target_col in df.columns else None

    return X, y

