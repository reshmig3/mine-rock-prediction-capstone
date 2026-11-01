# SonarSense: Mine vs Rock Prediction System

An end-to-end machine learning deployment project. It takes a trained scikit-learn model that classifies 60-measurement sonar samples as a **Mine** or a **Rock**, serves it through a FastAPI backend, and exposes it through a Streamlit web frontend, packaged with Docker and deployed on Render.

## Live Application

- **Frontend (web app):** https://sonarsense-frontend-yk6q.onrender.com
- **Backend API documentation:** https://mine-rock-prediction-capstone.onrender.com/docs
- **Health check:** https://mine-rock-prediction-capstone.onrender.com/health

> Free Render services sleep when idle. The first request can take about a minute while the service wakes up.

## Project Overview

The project began as a small earlier machine learning exercise and was rebuilt for this capstone. The model was retrained on the UCI Sonar dataset (`data/sonar.csv`, 208 samples, 60 measurements each) with a reproducible training script, then packaged as a deployable service.

The work covers the full path from model to production:

1. **Model training and evaluation:** `training/train.py` compares four classifiers with 5-fold cross-validation, selects the best one, evaluates it on a held-out test set, and saves the results.
2. **Model serialization:** the complete scikit-learn pipeline (feature scaler plus classifier) is saved with joblib, so the same preprocessing is applied at inference.
3. **Backend API:** FastAPI with `GET /health`, `GET /metrics` and `POST /predict`, with Pydantic validation of the input.
4. **Frontend:** a Streamlit interface that calls the API, shows the prediction and estimated probabilities, and displays the model's evaluation results.
5. **Containerization:** a Dockerfile for the backend.
6. **Cloud deployment:** the backend and frontend run as two services on Render.

## Model Performance

Selected model: **SVM with RBF kernel**, with a StandardScaler in the same pipeline. Chosen by 5-fold cross-validation on the training split.

| Measure | Result |
|---|---|
| 5-fold cross-validation accuracy (training split) | 83.1% ± 6.2% |
| Held-out test accuracy (42 samples) | 85.7% |
| Mine precision | 80.8% |
| Mine recall | 95.5% |
| Mine F1-score | 87.5% |

Test set confusion matrix (rows = actual, columns = predicted):

| | Predicted Mine | Predicted Rock |
|---|---|---|
| **Actual Mine** | 21 | 1 |
| **Actual Rock** | 5 | 15 |

Other models in cross-validation: Logistic Regression 73.5%, KNN 80.7%, Random Forest 80.7%.

**Caution:** the test set has only 42 samples, so each sample changes accuracy by about 2.4 points. The cross-validation estimate is the more stable figure. These results are for an educational project and are not a claim of production-grade accuracy.

## Project Structure

```
sentiment-analyzer-capstone/
├── app/
│   ├── main.py            # FastAPI backend (/health, /metrics, /predict)
│   └── frontend.py        # Streamlit frontend
├── data/
│   └── sonar.csv          # UCI Sonar dataset
├── model/
│   ├── mine_rock_pipeline.joblib   # Trained scaler + SVM pipeline
│   ├── metrics.json                # Saved evaluation results
│   └── confusion_matrix.png        # Test set confusion matrix
├── training/
│   └── train.py           # Training, evaluation and model saving
├── Dockerfile             # Container build for the backend
├── requirements.txt       # Pinned Python dependencies
└── README.md
```

## Run Locally

Requirements: Python 3.11 or 3.13 (tested on 3.13).

```bash
# 1. Create and activate a virtual environment (Windows PowerShell)
python -m venv .venv
.venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Retrain the model and regenerate metrics
pip install pandas matplotlib
python training/train.py

# 4. Start the API (Terminal 1)
python -m uvicorn app.main:app --reload
# API docs: http://127.0.0.1:8000/docs

# 5. Start the frontend (Terminal 2)
python -m streamlit run app/frontend.py
# Web app: http://localhost:8501
```

The frontend reads the backend address from the `API_BASE_URL` environment variable. If it isn't set, it uses `http://127.0.0.1:8000`.

## Run with Docker (backend)

```bash
docker build -t sonarsense-api .
docker run -p 8000:8000 sonarsense-api
```

Then open http://127.0.0.1:8000/docs.

## API Usage

### `GET /health`

Returns service status and whether the model is loaded.

```json
{ "status": "healthy", "model_loaded": true, "model_file": "mine_rock_pipeline.joblib" }
```

### `POST /predict`

Accepts exactly 60 finite numeric values in a JSON field named `features`.

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"features": [0.02, 0.03, ..., 0.39]}'
```

Response (200):

```json
{
  "prediction": "Mine",
  "label": "M",
  "model": "SVM (RBF) pipeline with StandardScaler",
  "probabilities": { "Mine": 0.6528, "Rock": 0.3472 }
}
```

Invalid input (for example, 3 values instead of 60) returns **422 Unprocessable Entity** with a description of the error.

The probabilities are the model's estimates and do not guarantee accuracy.

## Deployment (Render)

Two Render web services are connected to this GitHub repository:

- **Backend (FastAPI):** build from `requirements.txt`, start with `uvicorn app.main:app --host 0.0.0.0 --port $PORT`, with `PYTHON_VERSION=3.11.9`.
- **Frontend (Streamlit):** start with `streamlit run app/frontend.py --server.port $PORT --server.address 0.0.0.0`, with `PYTHON_VERSION=3.11.9` and the environment variable `API_BASE_URL=https://mine-rock-prediction-capstone.onrender.com`.

## Limitations

- A prediction is not proof that an object is a mine or a rock.
- Results come from a small dataset (208 samples) and a 42-sample test set, so the figures carry real uncertainty.
- This is an educational project and must not be used for operational mine detection or safety-critical decisions.

## Demo Video

[Add your video link here]