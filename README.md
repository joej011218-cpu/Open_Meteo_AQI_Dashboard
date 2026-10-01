# Open_Meteo deployment package

This package contains a Flask backend and a simple frontend for the trained
Vadodara Open-Meteo XGBoost delta models.

## 1. Copy your 24 trained model folders

Place them in:

backend/models/horizon_01h/
...
backend/models/horizon_24h/

Each folder must contain:

- xgboost_delta_model.joblib
- feature_columns.joblib
- model_config.joblib

The uploaded saved feature list was verified to contain 198 features.

## 2. Create Python environment

From the backend folder:

Windows:
    py -m venv .venv
    .venv\Scripts\activate

macOS/Linux:
    python3 -m venv .venv
    source .venv/bin/activate

Install:
    pip install -r requirements.txt

Important: if joblib/XGBoost shows a compatibility warning, use the same
XGBoost version that was used to train the models.

## 3. Start backend

    python app.py

It should run at:
    http://127.0.0.1:5000

Check:
    http://127.0.0.1:5000/api/health

Expected:
    models_loaded = 24

## 4. First refresh

Use a second terminal:

    python refresh_once.py

or call:

    POST http://127.0.0.1:5000/api/refresh

This downloads recent hourly Open-Meteo CAMS values, builds CPCB-style AQI,
creates the exact 198 training features, runs all 24 horizon models and stores
the newest forecast in SQLite.

## 5. Start frontend

From the frontend folder:

    python -m http.server 5500

Open:
    http://127.0.0.1:5500

## API endpoints

GET  /api/health
POST /api/refresh
GET  /api/latest
GET  /api/history?hours=168
GET  /api/forecast
GET  /api/model

## Important wording for the dashboard/thesis

The source is Open-Meteo CAMS Global modeled air-quality data.
It is not direct CPCB station sensor data.
The AQI shown here is a CPCB-style AQI calculated from the available
Open-Meteo pollutant concentrations.
