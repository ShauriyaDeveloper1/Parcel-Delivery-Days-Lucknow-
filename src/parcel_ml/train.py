"""
Model Training & MLflow Experiment Tracking for Lucknow Parcel Delivery Hub (MP012).
Trains and compares 3 regression algorithms, logs runs to SQLite, and registers champion.
"""

from pathlib import Path
import hashlib
import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from src.parcel_ml.preprocessing import build_preprocessor, prepare_features


def get_raw_data_sha256(raw_path: str = "data/raw/train.csv") -> str:
    """Computes SHA-256 fingerprint of the original raw dataset."""
    with open(raw_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def update_findings_yaml(
    experiment_name: str,
    best_run_id: str,
    registered_model_name: str,
    champion_version: str,
    findings_path: str = "findings.yaml",
):
    """
    Updates tracking fields in findings.yaml while preserving keys and structure.
    """
    with open(findings_path, "r") as f:
        content = f.read()

    import re
    content = re.sub(
        r"tracking:\n\s+experiment_name:.*\n\s+best_run_id:.*\n\s+registered_model_name:.*\n\s+champion_version:.*",
        f"tracking:\n  experiment_name: \"{experiment_name}\"\n  best_run_id: \"{best_run_id}\"\n  registered_model_name: \"{registered_model_name}\"\n  champion_version: {champion_version}",
        content,
    )

    with open(findings_path, "w") as f:
        f.write(content)

    print(f"[Findings] Successfully updated tracking fields in {findings_path}")


def train_and_evaluate():
    # 1. Load parameters
    params_path = Path("params.yaml")
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)

    data_params = params.get("data", {})
    cleaned_path = data_params.get("cleaned_train_path", "data/processed/cleaned_train.csv")
    target_col = data_params.get("target_column", "delivery_days")
    random_state = params.get("base", {}).get("random_state", 42)

    # 2. Load cleaned data
    print(f"Loading cleaned data from {cleaned_path}...")
    df = pd.read_csv(cleaned_path)
    X, y = prepare_features(df, target_col=target_col)

    # 3. Split into Train (80%) and Validation (20%)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    print(f"Train split: {X_train.shape}, Validation split: {X_val.shape}")

    # 4. Hash original raw train.csv (Student Handbook requirement)
    data_sha256 = get_raw_data_sha256("data/raw/train.csv")
    print(f"Raw data SHA-256 fingerprint: {data_sha256}")

    # 5. MLflow Tracking Setup
    mlflow_uri = "sqlite:///mlflow.db"
    mlflow.set_tracking_uri(mlflow_uri)
    experiment_name = "MP012_Parcel_Delivery_Days"
    mlflow.set_experiment(experiment_name)
    print(f"MLflow tracking URI: {mlflow_uri}, Experiment: {experiment_name}")

    # 6. Candidate Models (3 regression algorithms)
    candidate_models = {
        "Linear_Ridge": (
            Ridge(alpha=1.0, random_state=random_state),
            {"model_type": "Ridge", "alpha": 1.0},
        ),
        "Random_Forest": (
            RandomForestRegressor(n_estimators=100, max_depth=12, random_state=random_state),
            {"model_type": "RandomForestRegressor", "n_estimators": 100, "max_depth": 12},
        ),
        "Gradient_Boosting": (
            GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=random_state),
            {"model_type": "GradientBoostingRegressor", "n_estimators": 100, "learning_rate": 0.1, "max_depth": 4},
        ),
    }

    results = []

    for model_name, (model_obj, model_params) in candidate_models.items():
        print(f"\n--- Training {model_name} ---")
        preprocessor = build_preprocessor()
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", model_obj)])

        with mlflow.start_run(run_name=model_name) as run:
            run_id = run.info.run_id

            # Fit strictly on training partition
            pipeline.fit(X_train, y_train)

            # Predictions on validation partition
            y_pred = pipeline.predict(X_val)

            # Metrics
            mae = float(mean_absolute_error(y_val, y_pred))
            rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))
            r2 = float(r2_score(y_val, y_pred))

            print(f"[{model_name}] Val MAE: {mae:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f}")

            # Log to MLflow
            mlflow.log_param("data_sha256", data_sha256)
            mlflow.log_params(model_params)
            mlflow.log_param("val_split_size", 0.2)
            mlflow.log_param("random_state", random_state)

            mlflow.log_metric("mae_val", mae)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("rmse_val", rmse)
            mlflow.log_metric("r2_val", r2)

            # Log Model Artifact
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="model",
                serialization_format="cloudpickle",
            )

            results.append(
                {
                    "name": model_name,
                    "run_id": run_id,
                    "mae": mae,
                    "rmse": rmse,
                    "r2": r2,
                    "pipeline": pipeline,
                }
            )

    # 7. Select Best Model (Lowest MAE)
    results_df = pd.DataFrame(results).sort_values(by="mae").reset_index(drop=True)
    best_row = results_df.iloc[0]
    best_name = best_row["name"]
    best_run_id = best_row["run_id"]
    best_mae = best_row["mae"]
    best_pipeline = best_row["pipeline"]

    print("\n================ MODEL COMPARISON ================")
    print(results_df[["name", "mae", "rmse", "r2", "run_id"]].to_string(index=False))
    print(f"\nWinner: {best_name} with MAE: {best_mae:.4f} (Run ID: {best_run_id})")

    # 8. Save Champion Model for inference service (models/model.joblib)
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib_path = models_dir / "model.joblib"
    joblib.dump(best_pipeline, joblib_path)
    print(f"Saved winning model pipeline to {joblib_path}")

    # 9. Register in MLflow Model Registry
    reg_model_name = "parcel_delivery_days"
    model_uri = f"runs:/{best_run_id}/model"
    print(f"Registering model from {model_uri} under '{reg_model_name}'...")

    reg_version_obj = mlflow.register_model(model_uri=model_uri, name=reg_model_name)
    version = str(reg_version_obj.version)
    print(f"Successfully registered model version: {version}")

    # 10. Assign @champion alias
    client = MlflowClient()
    client.set_registered_model_alias(name=reg_model_name, alias="champion", version=version)
    print(f"Successfully assigned alias '@champion' to version {version} of '{reg_model_name}'")

    # 11. Update findings.yaml tracking section
    update_findings_yaml(
        experiment_name=experiment_name,
        best_run_id=best_run_id,
        registered_model_name=reg_model_name,
        champion_version=version,
    )

    return results_df, best_row


if __name__ == "__main__":
    train_and_evaluate()
