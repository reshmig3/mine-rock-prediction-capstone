
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Locate the saved model relative to this file
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "mine_rock_model.joblib"

model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Mine vs Rock Prediction API",
    description="Predicts whether sonar measurements indicate a mine or a rock.",
    version="1.0.0",
)


class SonarInput(BaseModel):
    features: list[float] = Field(
        ...,
        min_length=60,
        max_length=60,
        description="Exactly 60 numerical sonar measurements",
    )


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict")
def predict(data: SonarInput):
    try:
        values = np.array(data.features, dtype=float).reshape(1, -1)
        prediction = str(model.predict(values)[0])

        labels = {"M": "Mine", "R": "Rock"}
        return {
            "prediction": labels.get(prediction, prediction),
            "label": prediction,
        }
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}",
        ) from exc
