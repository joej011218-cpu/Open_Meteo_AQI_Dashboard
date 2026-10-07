from datetime import datetime, timezone

import pandas as pd

from aqi import calculate_aqi_dataframe

from config import DATA_RETENTION_DAYS

from database import (
    cleanup_old_air_quality,
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
    Refresh the live air-quality database.

    Processing sequence:

    1. Fetch recent hourly Open-Meteo CAMS data.
    2. Store/update pollutant readings in PostgreSQL.
    3. Retrieve chronological history.
    4. Calculate CPCB-style AQI.
    5. Store calculated AQI values.
    6. Remove data older than the configured
       retention period.
    7. Update application refresh state.

    The database keeps up to DATA_RETENTION_DAYS
    of historical hourly air-quality data.
    """

    # -----------------------------------------------------
    # FETCH OPEN-METEO
    # -----------------------------------------------------

    rows = fetch_recent_hourly(
        past_days=past_days
    )

    if not rows:
        raise RuntimeError(
            "Open-Meteo returned no hourly rows."
        )

    # -----------------------------------------------------
    # STORE POLLUTANT DATA
    # -----------------------------------------------------

    inserted_or_updated = (
        upsert_air_quality_rows(rows)
    )

    # -----------------------------------------------------
    # LOAD HISTORY REQUIRED FOR AQI
    # -----------------------------------------------------

    history_limit = max(
        past_days * 24 + 48,
        240,
    )

    db_rows = fetch_air_quality(
        limit=history_limit
    )

    if not db_rows:
        raise RuntimeError(
            "Database is empty after "
            "Open-Meteo refresh."
        )

    df = pd.DataFrame(
        db_rows
    )

    # -----------------------------------------------------
    # TIMESTAMP CONVERSION
    # -----------------------------------------------------

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    df = df.dropna(
        subset=["timestamp"]
    )

    df = (
        df.sort_values(
            "timestamp"
        )
        .drop_duplicates(
            subset=["timestamp"],
            keep="last",
        )
        .reset_index(
            drop=True
        )
    )

    # -----------------------------------------------------
    # NUMERIC CONVERSION
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # CAUSAL FILL
    # -----------------------------------------------------

    # Forward fill only.
    #
    # Do not use backward fill because that would
    # introduce future information into older rows.
    df[numeric_cols] = (
        df[numeric_cols]
        .ffill()
    )

    # -----------------------------------------------------
    # CPCB-STYLE AQI
    # -----------------------------------------------------

    aqi_df = (
        calculate_aqi_dataframe(
            df
        )
    )

    pairs = []

    for _, row in aqi_df.iterrows():

        if pd.isna(
            row["aqi"]
        ):
            aqi_value = None

        else:
            aqi_value = float(
                row["aqi"]
            )

        timestamp = row[
            "timestamp"
        ]

        # PostgreSQL accepts the ISO timestamp.
        pairs.append(
            (
                timestamp.isoformat(),
                aqi_value,
            )
        )

    # Persist calculated historical AQI values.
    replace_aqi_values(
        pairs
    )

    # -----------------------------------------------------
    # RETENTION
    # -----------------------------------------------------

    # Automatically remove hourly data older than
    # the configured retention period.
    deleted_old_rows = (
        cleanup_old_air_quality(
            retention_days=DATA_RETENTION_DAYS
        )
    )

    # -----------------------------------------------------
    # LATEST VALUE
    # -----------------------------------------------------

    latest_row = fetch_latest()

    # -----------------------------------------------------
    # REFRESH STATE
    # -----------------------------------------------------

    refresh_time = (
        datetime.now(
            timezone.utc
        )
        .isoformat()
    )

    set_state(
        "last_refresh_at",
        refresh_time,
    )

    # Clear any previous Open-Meteo error.
    set_state(
        "last_refresh_error",
        "",
    )

    # -----------------------------------------------------
    # RESULT
    # -----------------------------------------------------

    return {
        "fetched_rows": len(rows),

        "inserted_or_updated_rows": (
            inserted_or_updated
        ),

        "stored_rows_used_for_aqi": (
            len(aqi_df)
        ),

        "retention_days": (
            DATA_RETENTION_DAYS
        ),

        "deleted_old_rows": (
            deleted_old_rows
        ),

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

        "refreshed_at_utc": (
            refresh_time
        ),
    }