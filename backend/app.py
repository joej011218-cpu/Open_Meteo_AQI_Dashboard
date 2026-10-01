from datetime import datetime, timezone, timedelta
from threading import Lock

from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd

from config import BOOTSTRAP_PAST_DAYS
from database import (
    init_db,
    fetch_air_quality,
    fetch_latest,
    fetch_predictions,
    save_predictions,
    count_air_quality,
    fetch_alert_rows,
    get_state,
    set_state,
)
from predictor import XGBoostForecastService
from service import refresh_openmeteo_and_aqi


app = Flask(__name__)
CORS(app)

init_db()

forecast_service = XGBoostForecastService()
refresh_lock = Lock()

# Check for newer Open-Meteo data every 30 minutes.
AUTO_REFRESH_MINUTES = 30


def aqi_category(aqi):
    if aqi is None:
        return None

    value = float(aqi)

    if value <= 50:
        return "Good"
    if value <= 100:
        return "Satisfactory"
    if value <= 200:
        return "Moderate"
    if value <= 300:
        return "Poor"
    if value <= 400:
        return "Very Poor"
    return "Severe"


def build_dataframe_for_prediction():
    rows = fetch_air_quality(limit=1000)

    if len(rows) < 200:
        raise RuntimeError(
            "Not enough hourly history for prediction. "
            f"Database contains only {len(rows)} rows."
        )

    df = pd.DataFrame(rows)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
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


def parse_utc_datetime(value):
    if not value:
        return None

    try:
        parsed = datetime.fromisoformat(value)

        if parsed.tzinfo is None:
            parsed = parsed.replace(
                tzinfo=timezone.utc
            )

        return parsed.astimezone(
            timezone.utc
        )

    except Exception:
        return None


def refresh_is_due():
    if count_air_quality() < 200:
        return True

    if not fetch_predictions():
        return True

    last_refresh = parse_utc_datetime(
        get_state("last_refresh_at")
    )

    if last_refresh is None:
        return True

    age = (
        datetime.now(timezone.utc)
        - last_refresh
    )

    return age >= timedelta(
        minutes=AUTO_REFRESH_MINUTES
    )


def perform_full_refresh():
    refresh_info = (
        refresh_openmeteo_and_aqi(
            past_days=BOOTSTRAP_PAST_DAYS
        )
    )

    predictions = create_forecast()

    return {
        "refresh": refresh_info,
        "forecast_count": len(predictions),
    }


def ensure_data_ready(force=False):
    if (
        not force
        and
        not refresh_is_due()
    ):
        return {
            "refreshed": False,
            "reason": "cached data is still fresh",
        }

    with refresh_lock:
        if (
            not force
            and
            not refresh_is_due()
        ):
            return {
                "refreshed": False,
                "reason": "another request already refreshed the data",
            }

        try:
            result = perform_full_refresh()

            return {
                "refreshed": True,
                **result,
            }

        except Exception as error:
            set_state(
                "last_refresh_error",
                str(error),
            )

            # Keep serving stored data if Open-Meteo is temporarily unavailable.
            if (
                fetch_latest() is not None
                and
                fetch_predictions()
            ):
                app.logger.exception(
                    "Automatic refresh failed; serving cached data."
                )

                return {
                    "refreshed": False,
                    "warning": str(error),
                    "reason": "automatic refresh failed; cached data served",
                }

            raise


@app.get("/")
def home():
    return jsonify(
        {
            "service": "Vadodara Open-Meteo AQI Forecast API",
            "status": "ok",
            "source": "Open-Meteo CAMS Global",
            "aqi_method": "CPCB-style rolling AQI",
            "endpoints": {
                "health": "/api/health",
                "latest": "/api/latest",
                "history": "/api/history?hours=168",
                "alerts": "/api/alerts?limit=200",
                "forecast": "/api/forecast",
                "model": "/api/model",
                "refresh": "POST /api/refresh",
            },
        }
    )


@app.get("/api/health")
def health():
    latest = fetch_latest()
    predictions = fetch_predictions()

    return jsonify(
        {
            "status": "ok",
            "database_rows": count_air_quality(),
            "models_loaded": len(
                forecast_service.models
            ),
            "expected_models": 24,
            "forecast_rows": len(predictions),
            "latest_data_timestamp": (
                latest["timestamp"]
                if latest
                else None
            ),
            "last_refresh_at": get_state(
                "last_refresh_at"
            ),
            "last_refresh_error": get_state(
                "last_refresh_error",
                "",
            ),
            "auto_refresh_minutes": AUTO_REFRESH_MINUTES,
        }
    )


@app.post("/api/refresh")
def refresh():
    result = ensure_data_ready(force=True)

    return jsonify(
        {
            "status": "ok",
            **result,
        }
    )


@app.get("/api/latest")
def latest():
    ensure_data_ready()

    row = fetch_latest()

    if row is None:
        return jsonify(
            {
                "error": "Air-quality data is unavailable."
            }
        ), 503

    row["category"] = aqi_category(
        row.get("aqi")
    )

    return jsonify(row)


@app.get("/api/history")
def history():
    ensure_data_ready()

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

    for row in rows:
        row["category"] = aqi_category(
            row.get("aqi")
        )

    return jsonify(
        {
            "hours_requested": hours,
            "count": len(rows),
            "data": rows,
        }
    )


@app.get("/api/alerts")
def alerts():
    ensure_data_ready()

    try:
        limit = int(
            request.args.get(
                "limit",
                200,
            )
        )
    except ValueError:
        return jsonify(
            {
                "error": "limit must be an integer"
            }
        ), 400

    limit = max(
        1,
        min(limit, 1000),
    )

    try:
        threshold = float(
            request.args.get(
                "threshold",
                201,
            )
        )
    except ValueError:
        return jsonify(
            {
                "error": "threshold must be numeric"
            }
        ), 400

    rows = fetch_alert_rows(
        threshold=threshold,
        limit=limit,
    )

    for row in rows:
        row["category"] = aqi_category(
            row.get("aqi")
        )

    return jsonify(
        {
            "threshold": threshold,
            "count": len(rows),
            "data": rows,
        }
    )


@app.get("/api/forecast")
def forecast():
    ensure_data_ready()

    predictions = fetch_predictions()

    if not predictions:
        return jsonify(
            {
                "error": "Forecast is currently unavailable."
            }
        ), 503

    return jsonify(
        {
            "generated_at": predictions[0][
                "generated_at"
            ],
            "current_aqi": predictions[0][
                "current_aqi"
            ],
            "model": predictions[0][
                "model"
            ],
            "predictions": predictions,
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

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True,
        use_reloader=False,
    )
