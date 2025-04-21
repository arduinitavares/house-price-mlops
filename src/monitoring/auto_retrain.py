"""
Automated model retraining based on data drift detection.

This script:
1. Checks for significant data drift using the drift report JSON
2. Retrains the model with training data if necessary
3. Registers the new model in MLflow
4. Saves a local copy of the model in models/model.pkl
"""

import json
import os
import pickle
import random

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def load_data(file_path: str) -> pd.DataFrame:
    """Load the raw housing dataset."""
    return pd.read_csv(file_path)


def preprocess_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Preprocess the dataset minimally for modeling."""
    df = df.copy()

    # Drop non-informative ID columns
    df.drop(columns=["Order", "PID"], inplace=True, errors="ignore")

    # Drop rows with missing target
    df = df.dropna(subset=["SalePrice"])

    # Fill numeric NaNs with median, categorical with "Missing"
    for col in df.select_dtypes(include="number"):
        df[col] = df[col].fillna(df[col].median())

    for col in df.select_dtypes(include="object"):
        df[col] = df[col].fillna("Missing")

    # Convert integer columns to float64 to prevent MLflow warnings
    for col in df.select_dtypes(include="integer"):
        df[col] = df[col].astype("float64")

    # Split features and target
    X = df.drop(columns=["SalePrice"])
    y = pd.Series(df["SalePrice"].values, name="SalePrice")
    return X, y


def build_pipeline(X: pd.DataFrame) -> ColumnTransformer:
    """Build a basic preprocessing pipeline."""
    categorical_cols = X.select_dtypes(include="object").columns.tolist()
    numeric_cols = X.select_dtypes(exclude="object").columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_cols),
            ("cat", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]), categorical_cols),
        ]
    )

    return preprocessor


def evaluate_model(y_true, y_pred) -> dict:
    """Return basic regression metrics."""
    return {
        "rmse": mean_squared_error(y_true, y_pred) ** 0.5,
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def check_drift(drift_report_path: str, threshold: float = 0.2) -> bool:
    """
    Check if significant drift occurred based on drift report.

    Args:
        drift_report_path: Path to the drift report JSON file
        threshold: Threshold for considering drift significant

    Returns:
        bool: True if retraining is needed, False otherwise
    """
    # Load the drift report
    with open(drift_report_path, "r") as f:
        drift_report = json.load(f)

    try:
        for metric in drift_report.get("metrics", []):
            if metric.get("type") == "DataDriftPreset":
                # Extract the drift share values
                config_drift_share = metric.get(
                    "config", {}).get("drift_share", 0.5)

                # Get the actual detected drift from our calculations
                results = metric.get("results", {})
                actual_drift = results.get("drift_share_detected", 0.0)
                features_drifted = results.get("features_drifted", 0)
                features_analyzed = results.get("features_analyzed", 0)

                print(f"📊 Drift threshold configurado: {config_drift_share}")
                print(
                    f"📊 Drift detectado: {actual_drift:.4f} ({features_drifted}/{features_analyzed} features)")

                # Compare with our threshold for retraining
                print(f"📊 Threshold para retreino: {threshold}")
                return actual_drift > threshold
    except Exception as e:
        print(f"❌ Erro ao analisar relatório de drift: {str(e)}")

    # Default to False if we can't determine drift
    return False


def retrain_and_save_model(data_path: str, model_output_path: str) -> str:
    """
    Retrain model and save it locally.

    Args:
        data_path: Path to the training data
        model_output_path: Path to save the model

    Returns:
        str: MLflow run ID of the new model
    """
    print("🔄 Retraining model due to significant data drift...")

    # Load and preprocess data
    df = load_data(data_path)
    X, y = preprocess_data(df)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Set MLflow experiment
    mlflow.set_experiment("house-price-regression-auto-retrain")

    # Train Random Forest model and log to MLflow
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    with mlflow.start_run(run_name="AutoRetrain_RandomForest"):
        # Build pipeline and fit
        pipeline = Pipeline(steps=[
            ("preprocessor", build_pipeline(X_train)),
            ("regressor", model)
        ])
        pipeline.fit(X_train, y_train)

        # Evaluate
        y_pred = pipeline.predict(X_test)
        metrics = evaluate_model(y_test, y_pred)

        # Log to MLflow
        mlflow.log_params(model.get_params())
        mlflow.log_metrics(metrics)

        # Register model
        input_example = X_test.iloc[0:1]
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            input_example=input_example,
            registered_model_name="house_price_rf_model_auto"
        )

        print(f"✅ Model retrained with metrics: {metrics}")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_output_path), exist_ok=True)

        # Save model locally
        with open(model_output_path, "wb") as f:
            pickle.dump(pipeline, f)

        print(f"💾 Model saved to {model_output_path}")

        # Return run ID for reference
        return mlflow.active_run().info.run_id


def main() -> None:
    """Main function to check drift and retrain if necessary."""
    # File paths
    drift_report_path = "reports/data_drift_report.json"
    data_path = "data/raw/AmesHousing.csv"
    model_output_path = "models/model.pkl"

    # Drift threshold (customize as needed)
    drift_threshold = 0.2

    # Check if drift is significant
    needs_retraining = check_drift(drift_report_path, drift_threshold)

    if needs_retraining:
        run_id = retrain_and_save_model(data_path, model_output_path)
        print(f"✅ Retraining complete. New model run ID: {run_id}")
    else:
        print("✅ No significant drift detected. Retraining not needed.")


if __name__ == "__main__":
    main()
