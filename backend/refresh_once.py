"""
Run this file manually whenever you want to refresh Open-Meteo
data and create a new 24-hour forecast without using the browser.

Usage:
    python refresh_once.py
"""

import pandas as pd

from config import BOOTSTRAP_PAST_DAYS
from database import (
    init_db,
    fetch_air_quality,
    save_predictions,
)
from predictor import XGBoostForecastService
from service import refresh_openmeteo_and_aqi


def main():
    init_db()

    info = refresh_openmeteo_and_aqi(
        past_days=BOOTSTRAP_PAST_DAYS
    )

    print("Refresh:")
    print(info)

    rows = fetch_air_quality(
        limit=1000
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

    predictor = XGBoostForecastService()

    predictions = predictor.predict_24h(
        df
    )

    save_predictions(predictions)

    print(
        f"Saved {len(predictions)} forecasts."
    )

    for p in predictions:
        print(
            f"+{p['horizon']:02d}h "
            f"{p['forecast_time']} "
            f"AQI={p['predicted_aqi']:.2f}"
        )


if __name__ == "__main__":
    main()
