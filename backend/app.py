from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd

from config import (
    CORS_ORIGINS,
    BOOTSTRAP_PAST_DAYS,
)
from database import (
    init_db,
    fetch_air_quality,
    fetch_latest,
    fetch_predictions,
    save_predictions,
    count_air_quality,
)
from predictor import XGBoostForecastService
from service import refresh_openmeteo_and_aqi


app = Flask(__name__)

CORS(
    app,
    resources={
        r"/api/*": {
            "origins": CORS_ORIGINS
        }
    },
)

init_db()

# Load models once at backend startup.
forecast_service = XGBoostForecastService()


def build_dataframe_for_prediction():
    rows = fetch_air_quality(limit=1000)

    if len(rows) < 200:
        raise RuntimeError(
            "Not enough hourly history for prediction. "
            f"Database contains only {len(rows)} rows."
        )

    df = pd.DataFrame(rows)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    for col in [
        "pm25",
        "pm10",
        "no2",
        "so2",
        "co",
        "o3",
        "aqi",
    ]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    return (
        df.sort_values("timestamp")
        .reset_index(drop=True)
    )


def create_forecast():
    df = build_dataframe_for_prediction()

    predictions = (
        forecast_service
        .predict_24h(df)
    )

    save_predictions(predictions)

    return predictions


@app.get("/api/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "database_rows": count_air_quality(),
            "models_loaded": len(
                forecast_service.models
            ),
            "expected_models": 24,
        }
    )


@app.post("/api/refresh")
def refresh():
    """
    Manually fetch recent Open-Meteo data and immediately run
    the 24 XGBoost models.
    """
    refresh_info = (
        refresh_openmeteo_and_aqi(
            past_days=BOOTSTRAP_PAST_DAYS
        )
    )

    predictions = create_forecast()

    return jsonify(
        {
            "status": "ok",
            "refresh": refresh_info,
            "forecast_count": len(
                predictions
            ),
        }
    )


@app.get("/api/latest")
def latest():
    row = fetch_latest()

    if row is None:
        return jsonify(
            {
                "error": "No air-quality data yet. Call POST /api/refresh."
            }
        ), 404

    return jsonify(row)


@app.get("/api/history")
def history():
    try:
        hours = int(
            request.args.get(
                "hours",
                168,
            )
        )
    except ValueError:
        return jsonify(
            {
                "error": "hours must be an integer"
            }
        ), 400

    hours = max(
        1,
        min(hours, 1000),
    )

    rows = fetch_air_quality(
        limit=hours
    )

    return jsonify(
        {
            "hours_requested": hours,
            "count": len(rows),
            "data": rows,
        }
    )


@app.get("/api/forecast")
def forecast():
    predictions = fetch_predictions()

    if not predictions:
        return jsonify(
            {
                "error": (
                    "No forecast yet. Call POST /api/refresh first."
                )
            }
        ), 404

    return jsonify(
        {
            "generated_at":
                predictions[0][
                    "generated_at"
                ],

            "current_aqi":
                predictions[0][
                    "current_aqi"
                ],

            "model":
                predictions[0][
                    "model"
                ],

            "predictions":
                predictions,
        }
    )


@app.get("/api/model")
def model_info():
    return jsonify(
        forecast_service.model_info()
    )


@app.errorhandler(Exception)
def handle_exception(error):
    app.logger.exception(error)

    return jsonify(
        {
            "error": str(error),
            "type": error.__class__.__name__,
        }
    ), 500


if __name__ == "__main__":
    print(
        "Open-Meteo XGBoost backend starting..."
    )
    print(
        "First run: open another terminal and execute:"
    )
    print(
        "curl -X POST http://127.0.0.1:5000/api/refresh"
    )

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True,
        use_reloader=False,
    )
