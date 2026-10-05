import time

import requests
import pandas as pd

from config import (
    LATITUDE,
    LONGITUDE,
    TIMEZONE,
    OPEN_METEO_DOMAIN,
    OPEN_METEO_URL,
    BOOTSTRAP_PAST_DAYS,
)


HOURLY_VARIABLES = [
    "pm2_5",
    "pm10",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
]


def fetch_recent_hourly(
    past_days=BOOTSTRAP_PAST_DAYS
):
    """
    Fetch recent hourly Open-Meteo CAMS data.

    This function performs ONE HTTP request containing
    all required pollutants.

    The returned data is used for:
        1. SQLite storage
        2. CPCB-style AQI calculation
        3. XGBoost prediction

    Open-Meteo CO is returned in ug/m3 and is converted
    to mg/m3 to match the training data.
    """

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": ",".join(
            HOURLY_VARIABLES
        ),
        "past_days": int(past_days),
        "forecast_days": 1,
        "timezone": TIMEZONE,
        "domains": OPEN_METEO_DOMAIN,
    }

    max_attempts = 3

    for attempt in range(
        1,
        max_attempts + 1,
    ):

        try:
            response = requests.get(
                OPEN_METEO_URL,
                params=params,
                timeout=60,
            )

            if response.status_code == 429:

                if attempt < max_attempts:
                    wait_seconds = 20 * attempt

                    print(
                        "Open-Meteo returned HTTP 429. "
                        f"Waiting {wait_seconds}s before retry "
                        f"({attempt}/{max_attempts})."
                    )

                    time.sleep(
                        wait_seconds
                    )

                    continue

                raise RuntimeError(
                    "Open-Meteo returned HTTP 429 "
                    "Too Many Requests after "
                    f"{max_attempts} attempts. "
                    "The existing cached database data "
                    "will be preserved."
                )

            if response.status_code >= 500:

                if attempt < max_attempts:
                    wait_seconds = 10 * attempt

                    print(
                        "Open-Meteo returned "
                        f"HTTP {response.status_code}. "
                        f"Retrying in {wait_seconds}s "
                        f"({attempt}/{max_attempts})."
                    )

                    time.sleep(
                        wait_seconds
                    )

                    continue

            response.raise_for_status()

            payload = response.json()

            break

        except requests.RequestException as error:

            if attempt < max_attempts:
                wait_seconds = 10 * attempt

                print(
                    "Open-Meteo request failed: "
                    f"{error}. "
                    f"Retrying in {wait_seconds}s "
                    f"({attempt}/{max_attempts})."
                )

                time.sleep(
                    wait_seconds
                )

                continue

            raise RuntimeError(
                "Unable to retrieve data from "
                f"Open-Meteo after {max_attempts} attempts: "
                f"{error}"
            ) from error

    if "hourly" not in payload:
        raise RuntimeError(
            "Open-Meteo response does not contain "
            f"'hourly': {payload}"
        )

    df = pd.DataFrame(
        payload["hourly"]
    )

    df = df.rename(
        columns={
            "time": "timestamp",
            "pm2_5": "pm25",
            "pm10": "pm10",
            "carbon_monoxide": "co",
            "nitrogen_dioxide": "no2",
            "sulphur_dioxide": "so2",
            "ozone": "o3",
        }
    )

    required = [
        "timestamp",
        "pm25",
        "pm10",
        "no2",
        "so2",
        "co",
        "o3",
    ]

    missing = [
        c
        for c in required
        if c not in df.columns
    ]

    if missing:
        raise RuntimeError(
            "Open-Meteo is missing columns: "
            f"{missing}"
        )

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
    ]:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

        df.loc[
            df[col] < 0,
            col
        ] = pd.NA

    # Open-Meteo CO:
    # micrograms/m3 -> milligrams/m3
    #
    # This matches the training data unit.
    df["co"] = (
        df["co"] / 1000.0
    )

    df = (
        df.dropna(
            subset=[
                "timestamp",
                "pm25",
                "pm10",
                "no2",
                "so2",
                "co",
                "o3",
            ]
        )
        .sort_values("timestamp")
        .drop_duplicates(
            subset=["timestamp"],
            keep="last",
        )
        .reset_index(
            drop=True
        )
    )

    # Do not store future forecast rows
    # in the historical live database.
    now_local = (
        pd.Timestamp.now(
            tz=TIMEZONE
        )
        .tz_localize(None)
    )

    current_hour = (
        now_local.floor("h")
    )

    df = df[
        df["timestamp"]
        <= current_hour
    ].copy()

    rows = []

    for _, r in df.iterrows():

        rows.append(
            {
                "timestamp": (
                    r["timestamp"]
                    .isoformat()
                ),
                "pm25": float(
                    r["pm25"]
                ),
                "pm10": float(
                    r["pm10"]
                ),
                "no2": float(
                    r["no2"]
                ),
                "so2": float(
                    r["so2"]
                ),
                "co": float(
                    r["co"]
                ),
                "o3": float(
                    r["o3"]
                ),
                "aqi": None,
                "source": (
                    "Open-Meteo "
                    "CAMS Global"
                ),
            }
        )

    return rows