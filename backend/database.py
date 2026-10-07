from contextlib import contextmanager

import psycopg
from psycopg.rows import dict_row

from config import (
    DATABASE_URL,
    DATA_RETENTION_DAYS,
)


if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL environment variable "
        "is not configured."
    )


# ---------------------------------------------------------
# CONNECTION
# ---------------------------------------------------------

@contextmanager
def get_connection():

    conn = psycopg.connect(
        DATABASE_URL,
        row_factory=dict_row,
        connect_timeout=30,
    )

    try:
        yield conn
        conn.commit()

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()


# ---------------------------------------------------------
# INITIALIZATION
# ---------------------------------------------------------

def init_db():

    with get_connection() as conn:

        with conn.cursor() as cur:

            # -------------------------------------------------
            # AIR QUALITY
            # -------------------------------------------------

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS air_quality (
                    timestamp TIMESTAMP PRIMARY KEY,

                    pm25 DOUBLE PRECISION NOT NULL,
                    pm10 DOUBLE PRECISION NOT NULL,
                    no2 DOUBLE PRECISION NOT NULL,
                    so2 DOUBLE PRECISION NOT NULL,
                    co DOUBLE PRECISION NOT NULL,
                    o3 DOUBLE PRECISION NOT NULL,

                    aqi DOUBLE PRECISION,

                    source TEXT NOT NULL
                    DEFAULT 'Open-Meteo CAMS Global'
                )
                """
            )

            # -------------------------------------------------
            # PREDICTIONS
            # -------------------------------------------------

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    generated_at TIMESTAMPTZ NOT NULL,

                    forecast_time TIMESTAMP NOT NULL,

                    horizon INTEGER NOT NULL,

                    current_aqi DOUBLE PRECISION NOT NULL,

                    predicted_delta DOUBLE PRECISION NOT NULL,

                    predicted_aqi DOUBLE PRECISION NOT NULL,

                    model TEXT NOT NULL
                    DEFAULT 'XGBoost_DeltaOnly',

                    PRIMARY KEY (
                        generated_at,
                        horizon
                    )
                )
                """
            )

            # -------------------------------------------------
            # APPLICATION STATE
            # -------------------------------------------------

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS app_state (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
                """
            )

            # -------------------------------------------------
            # INDEXES
            # -------------------------------------------------

            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS
                idx_air_quality_timestamp
                ON air_quality (
                    timestamp DESC
                )
                """
            )

            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS
                idx_air_quality_aqi
                ON air_quality (
                    aqi
                )
                """
            )


# ---------------------------------------------------------
# RETENTION
# ---------------------------------------------------------

def cleanup_old_air_quality(
    retention_days=DATA_RETENTION_DAYS,
):
    """
    Delete air-quality records older than
    retention_days.

    Default = 365 days.
    """

    retention_days = int(
        retention_days
    )

    if retention_days < 1:
        raise ValueError(
            "retention_days must be "
            "at least 1."
        )

    with get_connection() as conn:

        with conn.cursor() as cur:

            cur.execute(
                """
                DELETE FROM air_quality
                WHERE timestamp <
                    CURRENT_TIMESTAMP
                    - (%s * INTERVAL '1 day')
                """,
                (
                    retention_days,
                ),
            )

            deleted = cur.rowcount

    return deleted


# ---------------------------------------------------------
# UPSERT AIR QUALITY
# ---------------------------------------------------------

def upsert_air_quality_rows(rows):

    if not rows:
        return 0

    values = [
        (
            row["timestamp"],
            row["pm25"],
            row["pm10"],
            row["no2"],
            row["so2"],
            row["co"],
            row["o3"],
            row.get("aqi"),
            row.get(
                "source",
                "Open-Meteo CAMS Global",
            ),
        )
        for row in rows
    ]

    with get_connection() as conn:

        with conn.cursor() as cur:

            cur.executemany(
                """
                INSERT INTO air_quality (
                    timestamp,
                    pm25,
                    pm10,
                    no2,
                    so2,
                    co,
                    o3,
                    aqi,
                    source
                )

                VALUES (
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s
                )

                ON CONFLICT (timestamp)
                DO UPDATE SET

                    pm25 = EXCLUDED.pm25,
                    pm10 = EXCLUDED.pm10,
                    no2 = EXCLUDED.no2,
                    so2 = EXCLUDED.so2,
                    co = EXCLUDED.co,
                    o3 = EXCLUDED.o3,

                    aqi = EXCLUDED.aqi,

                    source = EXCLUDED.source
                """,
                values,
            )

    return len(values)


# ---------------------------------------------------------
# AQI UPDATE
# ---------------------------------------------------------

def replace_aqi_values(
    timestamp_aqi_pairs
):

    if not timestamp_aqi_pairs:
        return 0

    values = [
        (
            aqi,
            timestamp,
        )
        for timestamp, aqi
        in timestamp_aqi_pairs
    ]

    with get_connection() as conn:

        with conn.cursor() as cur:

            cur.executemany(
                """
                UPDATE air_quality

                SET aqi = %s

                WHERE timestamp = %s
                """,
                values,
            )

    return len(values)


# ---------------------------------------------------------
# FETCH HISTORY
# ---------------------------------------------------------

def fetch_air_quality(
    limit=720
):

    limit = max(
        1,
        int(limit),
    )

    with get_connection() as conn:

        with conn.cursor() as cur:

            cur.execute(
                """
                SELECT
                    timestamp,
                    pm25,
                    pm10,
                    no2,
                    so2,
                    co,
                    o3,
                    aqi,
                    source

                FROM air_quality

                ORDER BY timestamp DESC

                LIMIT %s
                """,
                (
                    limit,
                ),
            )

            rows = cur.fetchall()

    # Original SQLite implementation returned
    # chronological order: oldest -> newest.
    rows = list(
        reversed(rows)
    )

    return [
        serialize_air_quality_row(
            row
        )
        for row in rows
    ]


# ---------------------------------------------------------
# FETCH LATEST
# ---------------------------------------------------------

def fetch_latest():

    with get_connection() as conn:

        with conn.cursor() as cur:

            cur.execute(
                """
                SELECT
                    timestamp,
                    pm25,
                    pm10,
                    no2,
                    so2,
                    co,
                    o3,
                    aqi,
                    source

                FROM air_quality

                ORDER BY timestamp DESC

                LIMIT 1
                """
            )

            row = cur.fetchone()

    if row is None:
        return None

    return serialize_air_quality_row(
        row
    )


# ---------------------------------------------------------
# COUNT
# ---------------------------------------------------------

def count_air_quality():

    with get_connection() as conn:

        with conn.cursor() as cur:

            cur.execute(
                """
                SELECT COUNT(*) AS n
                FROM air_quality
                """
            )

            row = cur.fetchone()

    return int(
        row["n"]
    )


# ---------------------------------------------------------
# ALERTS
# ---------------------------------------------------------

def fetch_alert_rows(
    threshold=201,
    limit=200,
):

    limit = max(
        1,
        int(limit),
    )

    with get_connection() as conn:

        with conn.cursor() as cur:

            cur.execute(
                """
                SELECT
                    timestamp,
                    aqi,
                    pm25,
                    pm10,
                    no2,
                    so2,
                    co,
                    o3,
                    source

                FROM air_quality

                WHERE aqi IS NOT NULL
                AND aqi >= %s

                ORDER BY timestamp DESC

                LIMIT %s
                """,
                (
                    float(threshold),
                    limit,
                ),
            )

            rows = cur.fetchall()

    return [
        serialize_air_quality_row(
            row
        )
        for row in rows
    ]


# ---------------------------------------------------------
# SAVE PREDICTIONS
# ---------------------------------------------------------

def save_predictions(
    predictions
):

    if not predictions:
        return 0

    values = [
        (
            p["generated_at"],
            p["forecast_time"],
            p["horizon"],
            p["current_aqi"],
            p["predicted_delta"],
            p["predicted_aqi"],
            p.get(
                "model",
                "XGBoost_DeltaOnly",
            ),
        )
        for p in predictions
    ]

    with get_connection() as conn:

        with conn.cursor() as cur:

            # Only the newest 24-hour forecast
            # needs to be served publicly.
            cur.execute(
                """
                DELETE FROM predictions
                """
            )

            cur.executemany(
                """
                INSERT INTO predictions (
                    generated_at,
                    forecast_time,
                    horizon,
                    current_aqi,
                    predicted_delta,
                    predicted_aqi,
                    model
                )

                VALUES (
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s
                )
                """,
                values,
            )

    return len(values)


# ---------------------------------------------------------
# FETCH PREDICTIONS
# ---------------------------------------------------------

def fetch_predictions():

    with get_connection() as conn:

        with conn.cursor() as cur:

            cur.execute(
                """
                SELECT
                    generated_at,
                    forecast_time,
                    horizon,
                    current_aqi,
                    predicted_delta,
                    predicted_aqi,
                    model

                FROM predictions

                ORDER BY horizon ASC
                """
            )

            rows = cur.fetchall()

    return [
        serialize_prediction_row(
            row
        )
        for row in rows
    ]


# ---------------------------------------------------------
# APPLICATION STATE
# ---------------------------------------------------------

def set_state(
    key,
    value
):

    with get_connection() as conn:

        with conn.cursor() as cur:

            cur.execute(
                """
                INSERT INTO app_state (
                    key,
                    value
                )

                VALUES (
                    %s,
                    %s
                )

                ON CONFLICT (key)
                DO UPDATE SET
                    value = EXCLUDED.value
                """,
                (
                    str(key),

                    None
                    if value is None
                    else str(value),
                ),
            )


def get_state(
    key,
    default=None,
):

    with get_connection() as conn:

        with conn.cursor() as cur:

            cur.execute(
                """
                SELECT value

                FROM app_state

                WHERE key = %s
                """,
                (
                    str(key),
                ),
            )

            row = cur.fetchone()

    if row is None:
        return default

    return row["value"]


# ---------------------------------------------------------
# SERIALIZATION
# ---------------------------------------------------------

def serialize_air_quality_row(
    row
):

    result = dict(row)

    timestamp = result.get(
        "timestamp"
    )

    if timestamp is not None:
        result["timestamp"] = (
            timestamp.isoformat()
        )

    return result


def serialize_prediction_row(
    row
):

    result = dict(row)

    generated_at = result.get(
        "generated_at"
    )

    forecast_time = result.get(
        "forecast_time"
    )

    if generated_at is not None:

        result["generated_at"] = (
            generated_at.isoformat()
        )

    if forecast_time is not None:

        result["forecast_time"] = (
            forecast_time.isoformat()
        )

    return result