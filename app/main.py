from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "mine_rock_pipeline.joblib"
METRICS_PATH = BASE_DIR / "model" / "metrics.json"

pipeline = joblib.load(MODEL_PATH)

LABELS = {"M": "Mine", "R": "Rock"}

app = FastAPI(
    title="SonarSense Mine vs Rock Prediction API",
    description="Predicts whether a 60-measurement sonar sample indicates a mine or a rock.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class SonarInput(BaseModel):
    features: list[float] = Field(
        ...,
        min_length=60,
        max_length=60,
        description="Exactly 60 numerical sonar measurements",
    )

    @field_validator("features")
    @classmethod
    def no_missing_or_infinite(cls, values):
        if any(not np.isfinite(v) for v in values):
            raise ValueError("All 60 measurements must be finite numbers.")
        return values


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": pipeline is not None,
        "model_file": MODEL_PATH.name,
    }


@app.get("/metrics")
def metrics():
    import json

    if not METRICS_PATH.exists():
        raise HTTPException(status_code=404, detail="Metrics file not found.")
    with open(METRICS_PATH) as f:
        return json.load(f)


@app.post("/predict")
def predict(data: SonarInput):
    try:
        values = np.array(data.features, dtype=float).reshape(1, -1)
        label = str(pipeline.predict(values)[0])
        probabilities = pipeline.predict_proba(values)[0]
        classes = list(pipeline.classes_)
        probability_map = {
            LABELS.get(str(c), str(c)): round(float(p), 4)
            for c, p in zip(classes, probabilities)
        }
        return {
            "prediction": LABELS.get(label, label),
            "label": label,
            "model": "SVM (RBF) pipeline with StandardScaler",
            "probabilities": probability_map,
        }
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="The model could not process this input. Please check the measurements and try again.",
        )