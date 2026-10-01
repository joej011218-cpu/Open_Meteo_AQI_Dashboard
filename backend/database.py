import sqlite3
from contextlib import contextmanager

from config import DATA_DIR, DB_PATH


@contextmanager
def get_connection():
    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    conn = sqlite3.connect(
        DB_PATH,
        timeout=30,
    )

    conn.row_factory = sqlite3.Row

    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS air_quality (
                timestamp TEXT PRIMARY KEY,
                pm25 REAL NOT NULL,
                pm10 REAL NOT NULL,
                no2 REAL NOT NULL,
                so2 REAL NOT NULL,
                co REAL NOT NULL,
                o3 REAL NOT NULL,
                aqi REAL,
                source TEXT NOT NULL DEFAULT 'Open-Meteo CAMS Global'
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                generated_at TEXT NOT NULL,
                forecast_time TEXT NOT NULL,
                horizon INTEGER NOT NULL,
                current_aqi REAL NOT NULL,
                predicted_delta REAL NOT NULL,
                predicted_aqi REAL NOT NULL,
                model TEXT NOT NULL DEFAULT 'XGBoost_DeltaOnly',
                PRIMARY KEY (generated_at, horizon)
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS app_state (
                key TEXT PRIMARY KEY,
                value TEXT
            )
            """
        )


def upsert_air_quality_rows(rows):
    if not rows:
        return 0

    with get_connection() as conn:
        conn.executemany(
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
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)

            ON CONFLICT(timestamp) DO UPDATE SET
                pm25=excluded.pm25,
                pm10=excluded.pm10,
                no2=excluded.no2,
                so2=excluded.so2,
                co=excluded.co,
                o3=excluded.o3,
                aqi=excluded.aqi,
                source=excluded.source
            """,
            [
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
            ],
        )

    return len(rows)


def replace_aqi_values(
    timestamp_aqi_pairs
):
    if not timestamp_aqi_pairs:
        return

    with get_connection() as conn:
        conn.executemany(
            """
            UPDATE air_quality
            SET aqi=?
            WHERE timestamp=?
            """,
            [
                (
                    aqi,
                    timestamp,
                )
                for timestamp, aqi
                in timestamp_aqi_pairs
            ],
        )


def fetch_air_quality(limit=720):
    with get_connection() as conn:
        rows = conn.execute(
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
            LIMIT ?
            """,
            (
                int(limit),
            ),
        ).fetchall()

    return [
        dict(row)
        for row in reversed(rows)
    ]


def fetch_latest():
    with get_connection() as conn:
        row = conn.execute(
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
        ).fetchone()

    return (
        dict(row)
        if row
        else None
    )


def count_air_quality():
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT COUNT(*) AS n
            FROM air_quality
            """
        ).fetchone()

    return int(row["n"])


def fetch_alert_rows(
    threshold=201,
    limit=200,
):
    """
    Alerts are derived from the AQI values already stored in air_quality.
    No duplicate alerts table is required.
    """
    with get_connection() as conn:
        rows = conn.execute(
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
              AND aqi >= ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (
                float(threshold),
                int(limit),
            ),
        ).fetchall()

    return [
        dict(row)
        for row in rows
    ]


def save_predictions(predictions):
    if not predictions:
        return

    with get_connection() as conn:
        conn.execute(
            "DELETE FROM predictions"
        )

        conn.executemany(
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
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
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
            ],
        )


def fetch_predictions():
    with get_connection() as conn:
        rows = conn.execute(
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
        ).fetchall()

    return [
        dict(row)
        for row in rows
    ]


def set_state(key, value):
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO app_state (
                key,
                value
            )
            VALUES (?, ?)

            ON CONFLICT(key) DO UPDATE SET
                value=excluded.value
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
        row = conn.execute(
            """
            SELECT value
            FROM app_state
            WHERE key=?
            """,
            (
                str(key),
            ),
        ).fetchone()

    if row is None:
        return default

    return row["value"]
