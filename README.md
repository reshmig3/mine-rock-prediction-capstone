# Mine vs Rock Prediction System

An end-to-end machine learning application that predicts whether sonar measurements indicate a mine or a rock.

## Features
- Machine learning model saved using Joblib
- FastAPI backend for real-time predictions
- Health-check endpoint to verify API status
- Streamlit frontend for user interaction
- Input validation for 60 sonar measurements

## Tech Stack
- Python
- Scikit-learn
- NumPy
- Joblib
- FastAPI
- Streamlit
- Requests

## Project Structure
```text
mine-rock-prediction-capstone/
├── app/
│   ├── main.py
│   └── frontend.py
├── model/
│   └── mine_rock_model.joblib
├── requirements.txt
└── README.md
```

## Run Locally

### 1. Start the API
```bash
python -m uvicorn app.main:app --reload
```

API documentation: http://127.0.0.1:8000/docs

### 2. Start the frontend

Open a second terminal and run:
```bash
python -m streamlit run app/frontend.py
```

## API Endpoints
- `GET /health` — checks API and model status
- `POST /predict` — predicts Mine or Rock from 60 numerical sonar measurements

## Project Goal
To demonstrate the deployment of a trained machine learning model through a backend API and an interactive frontend.
