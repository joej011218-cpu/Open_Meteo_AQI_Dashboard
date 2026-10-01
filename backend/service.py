import pandas as pd

from aqi import calculate_aqi_dataframe
from database import (
    fetch_air_quality,
    replace_aqi_values,
    upsert_air_quality_rows,
)
from openmeteo import fetch_recent_hourly


def refresh_openmeteo_and_aqi(past_days=30):
    """
    Refresh recent Open-Meteo data, store it, recalculate AQI,
    and update the database.
    """
    rows = fetch_recent_hourly(
        past_days=past_days
    )

    upsert_air_quality_rows(rows)

    db_rows = fetch_air_quality(
        limit=max(
            past_days * 24 + 48,
            240,
        )
    )

    if not db_rows:
        raise RuntimeError(
            "Database is empty after Open-Meteo refresh."
        )

    df = pd.DataFrame(db_rows)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    numeric_cols = [
        "pm25",
        "pm10",
        "no2",
        "so2",
        "co",
        "o3",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    # Runtime source is expected to be complete hourly data.
    # Forward fill is causal and matches the deployment philosophy.
    df[numeric_cols] = (
        df[numeric_cols]
        .ffill()
    )

    aqi_df = calculate_aqi_dataframe(
        df
    )

    pairs = []

    for _, row in aqi_df.iterrows():
        if pd.isna(row["aqi"]):
            aqi_value = None
        else:
            aqi_value = float(row["aqi"])

        pairs.append(
            (
                row["timestamp"].isoformat(),
                aqi_value,
            )
        )

    replace_aqi_values(pairs)

    return {
        "fetched_rows": len(rows),
        "stored_rows_used_for_aqi": len(aqi_df),
        "latest_timestamp": (
            aqi_df["timestamp"]
            .iloc[-1]
            .isoformat()
        ),
        "latest_aqi": (
            None
            if pd.isna(
                aqi_df["aqi"].iloc[-1]
            )
            else float(
                aqi_df["aqi"].iloc[-1]
            )
        ),
    }
