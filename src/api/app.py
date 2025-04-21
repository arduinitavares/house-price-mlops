"""FastAPI app for house price prediction"""

import os
from typing import Dict, List

import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Carrega o modelo a partir de uma variável de ambiente
MODEL_URI = os.getenv(
    "MODEL_URI",
    "mlruns/330848658664964622/673361f259d44733b52881dd0a0c30cc/artifacts/model",
)
model = mlflow.pyfunc.load_model(MODEL_URI)

app = FastAPI()


class InputData(BaseModel):
    """Input data model for prediction."""
    data: List[Dict]


@app.post("/predict")
def predict(request_data: InputData):
    """Endpoint for making predictions."""
    input_df = pd.DataFrame(request_data.data)
    print("📊 Preview of data:")
    print(input_df.head())
    predictions = model.predict(input_df)
    return {"predictions": predictions.tolist()}
