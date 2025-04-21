"""
FastAPI app to serve house price predictions using MLflow-registered model.
"""

import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Load model from MLflow Model Registry
MODEL_URI = "models:/house_price_rf_model/1"
model = mlflow.pyfunc.load_model(MODEL_URI)

app = FastAPI(title="House Price Predictor", version="1.0")


class HouseFeatures(BaseModel):
    """Model for house features input."""
    data: list[dict]  # Expect a list of dictionaries (records)


@app.post("/predict")
def predict(features: HouseFeatures):
    """Predict house prices based on input features."""
    # Convert to DataFrame
    input_df = pd.DataFrame(features.data)

    # Debug prints
    print("🔍 Input received:", input_df.columns.tolist())
    print("📊 Preview of data:\n", input_df.head())
    print("📏 Input DataFrame shape:", input_df.shape)

    # Predict
    predictions = model.predict(input_df)

    # Debug prints
    print("✅ Predictions made:", predictions)

    # Return result
    return {"predictions": predictions.tolist()}
