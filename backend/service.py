from datetime import datetime, timezone

import pandas as pd

from aqi import calculate_aqi_dataframe
from database import (
    fetch_air_quality,
    fetch_latest,
    replace_aqi_values,
    set_state,
    upsert_air_quality_rows,
)
from openmeteo import fetch_recent_hourly


def refresh_openmeteo_and_aqi(
    past_days=30,
):
    """
    Fetch recent Open-Meteo CAMS data, store the pollutant values,
    calculate CPCB-style AQI, and store every calculated hourly AQI
    back into the same air_quality table.
    """
    rows = fetch_recent_hourly(
        past_days=past_days
    )

    if not rows:
        raise RuntimeError(
            "Open-Meteo returned no hourly rows."
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
        df["timestamp"],
        errors="coerce",
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

    # Causal fill only.
    df[numeric_cols] = (
        df[numeric_cols]
        .ffill()
    )

    aqi_df = calculate_aqi_dataframe(df)

    pairs = []

    for _, row in aqi_df.iterrows():
        if pd.isna(row["aqi"]):
            aqi_value = None
        else:
            aqi_value = float(
                row["aqi"]
            )

        pairs.append(
            (
                row["timestamp"].isoformat(),
                aqi_value,
            )
        )

    # This is the step that persists the calculated AQI history.
    replace_aqi_values(pairs)

    latest_row = fetch_latest()

    refresh_time = (
        datetime.now(timezone.utc)
        .isoformat()
    )

    set_state(
        "last_refresh_at",
        refresh_time,
    )

    set_state(
        "last_refresh_error",
        "",
    )

    return {
        "fetched_rows": len(rows),
        "stored_rows_used_for_aqi": len(aqi_df),
        "latest_timestamp": (
            latest_row["timestamp"]
            if latest_row
            else None
        ),
        "latest_aqi": (
            latest_row["aqi"]
            if latest_row
            else None
        ),
        "refreshed_at_utc": refresh_time,
    }
