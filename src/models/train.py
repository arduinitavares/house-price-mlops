"""
Train regression models on Ames Housing dataset and track with MLflow.
"""


import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
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


def train_and_log(X_train, X_test, y_train, y_test, model, model_name: str):
    """Train, evaluate and log model with MLflow."""
    with mlflow.start_run(run_name=model_name):
        pipeline = Pipeline(steps=[
            ("preprocessor", build_pipeline(X_train)),
            ("regressor", model)
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        metrics = evaluate_model(y_test, y_pred)

        # Logging
        mlflow.log_params(model.get_params())
        mlflow.log_metrics(metrics)
        input_example = X_test.iloc[0:1]
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            input_example=input_example,
            registered_model_name="house_price_rf_model" if model_name == "RandomForest" else None
        )

        print(f"{model_name} metrics:", metrics)


def main():
    """Main function to load data, preprocess, train and log models."""
    data_path = "data/raw/AmesHousing.csv"
    df = load_data(data_path)
    X, y = preprocess_data(df)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    mlflow.set_experiment("house-price-regression")

    # Train models
    train_and_log(X_train, X_test, y_train, y_test,
                  LinearRegression(), "LinearRegression")
    train_and_log(X_train, X_test, y_train, y_test,
                  RandomForestRegressor(n_estimators=100), "RandomForest")


if __name__ == "__main__":
    main()
