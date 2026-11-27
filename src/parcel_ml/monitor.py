"""
Stage 4 Monitoring, Drift Detection, Retraining and Evaluation Module
for Lucknow Parcel Delivery Hub (MP012).
"""

from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import yaml
import joblib
import re
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from src.parcel_ml.preprocessing import build_preprocessor, prepare_features
from src.parcel_ml.cleaning import clean_dataset
from src.parcel_ml.logger import get_logger
logger = get_logger("mp012.monitor")


def calculate_psi(expected: pd.Series, actual: pd.Series, num_bins: int = 10, epsilon: float = 1e-4) -> float:
    """
    Computes the Population Stability Index (PSI) between expected (train) and actual (later)
    using 10 quantile bins based on the reference distribution (as in Lab P13).
    """
    exp = expected.dropna().values
    act = actual.dropna().values

    # Determine 10 quantile bin edges from expected distribution
    percentiles = np.linspace(0, 100, num_bins + 1)
    bins = np.percentile(exp, percentiles)
    bins = np.unique(bins)
    bins[0] = -np.inf
    bins[-1] = np.inf

    # Bin frequencies
    exp_counts, _ = np.histogram(exp, bins=bins)
    act_counts, _ = np.histogram(act, bins=bins)

    # Fractions
    exp_pct = exp_counts / len(exp)
    act_pct = act_counts / len(act)

    # Replace zero bins with small epsilon to prevent division by zero / log(0)
    exp_pct = np.where(exp_pct == 0, epsilon, exp_pct)
    act_pct = np.where(act_pct == 0, epsilon, act_pct)

    psi_vector = (act_pct - exp_pct) * np.log(act_pct / exp_pct)
    return float(np.sum(psi_vector))


def run_stage4():
    # Paths
    train_path = Path("data/processed/cleaned_train.csv")
    later_path = Path("data/raw/later.csv")
    holdout_path = Path("data/raw/holdout_features.csv")
    findings_path = Path("findings.yaml")
    v1_model_path = Path("models/model.joblib")
    predictions_dir = Path("predictions")
    predictions_dir.mkdir(parents=True, exist_ok=True)
    holdout_pred_path = predictions_dir / "holdout_predictions.csv"

    # 1. Load Datasets
    logger.info("Stage 4 monitoring started")
    print(f"Loading reference data from {train_path}...")
    train = pd.read_csv(train_path)
    print(f"Loading monitoring data from {later_path}...")
    later = pd.read_csv(later_path)
    later = clean_dataset(later, is_train=False)

    # 2. Step 2: PSI Drift Detection
    psi_features = [
        "distance_km",
        "weight_kg",
        "hub_backlog",
        "pin_code_tier",
        "declared_value_rs",
        "transfers",
    ]
    psi_results = {}
    print("\n--- Calculating Population Stability Index (PSI) ---")
    for col in psi_features:
        psi_val = calculate_psi(train[col], later[col], num_bins=10)
        psi_results[col] = round(psi_val, 4)
        print(f"  PSI({col:18}): {psi_results[col]:.4f}")

    most_drifted = max(psi_results, key=psi_results.get)
    print(f"Most Drifted Feature: {most_drifted} (PSI = {psi_results[most_drifted]})")
    logger.info(
    f"Drift analysis completed. Feature: {most_drifted}, PSI: {psi_results[most_drifted]:.4f}"
)

    # 3. Step 4: Group-wise Evaluation of v1 on later.csv
    print(f"\n--- Group-wise Evaluation of v1 on later.csv ---")
    v1 = joblib.load(v1_model_path)
    y_later_true = later["delivery_days"]
    y_later_pred_v1 = v1.predict(later)

    groups = ["Metro", "Tier-2", "Rural"]
    mae_by_group = {}
    for grp in groups:
        mask = later["destination"] == grp
        grp_mae = mean_absolute_error(y_later_true[mask], y_later_pred_v1[mask])
        mae_by_group[grp] = round(float(grp_mae), 4)
        print(f"  MAE for {grp:10}: {mae_by_group[grp]:.4f} (count: {mask.sum()})")

    worst_served = max(mae_by_group, key=mae_by_group.get)
    print(f"Worst Served Group: {worst_served} (MAE = {mae_by_group[worst_served]})")

    # 4. Step 3: Retraining Decision & Promotion Rule
    print("\n--- Retraining & Promotion Evaluation ---")
    # Sort later.csv by date
    later_sorted = later.sort_values(by="date", kind="mergesort").reset_index(drop=True)
    n_split = int(len(later_sorted) * 0.7)  # 70% train split
    later_70 = later_sorted.iloc[:n_split].copy()
    later_30 = later_sorted.iloc[n_split:].copy()

    print(f"later.csv temporal split: {len(later_70)} rows (70%) train, {len(later_30)} rows (30%) check")

    # Evaluate v1 on the last 30% check partition
    X_check, y_check = prepare_features(later_30, target_col="delivery_days")
    pred_v1_check = v1.predict(X_check)
    mae_v1_check = round(float(mean_absolute_error(y_check, pred_v1_check)), 4)
    print(f"MAE of v1 on 30% check partition: {mae_v1_check:.4f}")

    # Retrain v2 on cleaned train.csv + first 70% of later.csv
    train_v2 = pd.concat([train, later_70], ignore_index=True)
    X_train_v2, y_train_v2 = prepare_features(train_v2, target_col="delivery_days")

    # Retrain pipeline (Ridge model matching v1 architecture)
    v2_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("regressor", Ridge(alpha=1.0, random_state=42)),
        ]
    )

    # Setup MLflow
    mlflow_uri = "sqlite:///mlflow_stage4.db"
    mlflow.set_tracking_uri(mlflow_uri)
    experiment_name = "MP012_Parcel_Delivery_Days"
    mlflow.set_experiment(experiment_name)

    # Raw train hash
    with open("data/raw/train.csv", "rb") as f:
        data_sha256 = hashlib.sha256(f.read()).hexdigest()

    with mlflow.start_run(run_name="v2_retrained") as run:
        v2_run_id = run.info.run_id
        v2_pipeline.fit(X_train_v2, y_train_v2)

        pred_v2_check = v2_pipeline.predict(X_check)
        mae_v2_check = round(float(mean_absolute_error(y_check, pred_v2_check)), 4)
        rmse_v2_check = float(np.sqrt(mean_squared_error(y_check, pred_v2_check)))
        r2_v2_check = float(r2_score(y_check, pred_v2_check))

        print(f"MAE of v2 on 30% check partition: {mae_v2_check:.4f}")

        # Log to MLflow
        mlflow.log_param("data_sha256", data_sha256)
        mlflow.log_param("model_type", "Ridge_v2_retrained")
        mlflow.log_param("retrain_split_ratio", 0.7)
        mlflow.log_metric("mae_v2_on_check", mae_v2_check)
        mlflow.log_metric("mae", mae_v2_check)
        mlflow.log_metric("rmse_v2_on_check", rmse_v2_check)
        mlflow.log_metric("r2_v2_on_check", r2_v2_check)

        mlflow.sklearn.log_model(
            sk_model=v2_pipeline,
            artifact_path="model",
            serialization_format="cloudpickle",
        )

    # Promotion Rule: promote v2 only if its MAE on last 30% is at least 8% lower than v1's
    improvement = (mae_v1_check - mae_v2_check) / mae_v1_check
    print(f"Performance Improvement: {improvement * 100:.2f}% (Threshold: 8.00%)")

    reg_model_name = "parcel_delivery_days"
    client = MlflowClient()

    if improvement >= 0.08:
        decision = "promote"
        print(f"Decision: {decision} (v2 achieves >= 8% improvement over v1).")

        # Register v2 in MLflow registry
        model_uri = f"runs:/{v2_run_id}/model"
        v2_version_obj = mlflow.register_model(model_uri=model_uri, name=reg_model_name)
        v2_version = str(v2_version_obj.version)
        print(f"Registered v2 version: {v2_version}")

        # Promote to @champion alias
        client.set_registered_model_alias(name=reg_model_name, alias="champion", version=v2_version)
        print(f"Promoted v2 (version {v2_version}) to '@champion'")

        # Save v2 as model.joblib
        joblib.dump(v2_pipeline, v1_model_path)
        print(f"Updated {v1_model_path} with champion v2 model.")
        champion_version_final = int(v2_version)
        champion_pipeline = v2_pipeline
    else:
        decision = "keep"
        print(f"Decision: {decision} (v2 did not meet 8% improvement threshold).")
        champion_version_final = 1
        champion_pipeline = v1
        logger.info(
    f"Model evaluation completed. Improvement: {improvement * 100:.2f}%, "
    f"Decision: {decision}"
)

    # 5. Step 5: Holdout Predictions using @champion Model
    print("\n--- Generating Holdout Predictions ---")
    holdout_df = pd.read_csv(holdout_path)
    X_holdout, _ = prepare_features(holdout_df, target_col=None)

    holdout_preds = champion_pipeline.predict(X_holdout)

    pred_df = pd.DataFrame({
        "record_id": holdout_df["record_id"],
        "prediction": holdout_preds,
    })

    pred_df.to_csv(holdout_pred_path, index=False)
    print(f"Successfully saved {len(pred_df)} holdout predictions to {holdout_pred_path}")
    logger.info(
    f"Holdout predictions generated successfully. Count: {len(pred_df)}"
)
    print(pred_df.head(5).to_string(index=False))

    # 6. Step 6: Update findings.yaml
    print(f"\n--- Updating {findings_path} ---")
    with open(findings_path, "r") as f:
        content = f.read()

    # Update champion version if promoted
    if decision == "promote":
        content = re.sub(r"champion_version:\s*\d+", f"champion_version: {champion_version_final}", content)

    # Format monitoring section exactly
    monitoring_block = f"""monitoring:
  psi:                     # cleaned train.csv vs later.csv, 10 bins
    distance_km: {psi_results['distance_km']}
    weight_kg: {psi_results['weight_kg']}
    hub_backlog: {psi_results['hub_backlog']}
    pin_code_tier: {psi_results['pin_code_tier']}
    declared_value_rs: {psi_results['declared_value_rs']}
    transfers: {psi_results['transfers']}
  most_drifted_feature: "{most_drifted}"
  mae_by_group:            # v1 on later.csv, per value of destination
    'Metro': {mae_by_group['Metro']}
    'Tier-2': {mae_by_group['Tier-2']}
    'Rural': {mae_by_group['Rural']}
  worst_served_group: "{worst_served}"
  mae_v1_on_check: {mae_v1_check}    # v1 on the last 30% of later.csv (by date)
  mae_v2_on_check: {mae_v2_check}    # v2 on the same rows
  decision: "{decision}"           # promote or keep"""

    content = re.sub(r"monitoring:\n(?:\s+.*\n)*?\s+decision:.*", monitoring_block, content)

    with open(findings_path, "w") as f:
        f.write(content)

    print(f"findings.yaml successfully updated with actual Stage 4 results!")

    return {
        "psi": psi_results,
        "most_drifted": most_drifted,
        "mae_by_group": mae_by_group,
        "worst_served": worst_served,
        "mae_v1_check": mae_v1_check,
        "mae_v2_check": mae_v2_check,
        "improvement": improvement,
        "decision": decision,
        "v2_run_id": v2_run_id,
        "champion_version": champion_version_final,
    }


if __name__ == "__main__":
    run_stage4()
