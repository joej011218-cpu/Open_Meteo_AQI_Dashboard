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


def fetch_recent_hourly(past_days=BOOTSTRAP_PAST_DAYS):
    """
    Fetch recent hourly Open-Meteo CAMS data.

    We use the hourly series for model input because the XGBoost model
    was trained on hourly rows. Open-Meteo CO is returned in ug/m3;
    it is converted here to mg/m3 to match training.
    """
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": ",".join(HOURLY_VARIABLES),
        "past_days": int(past_days),
        "forecast_days": 1,
        "timezone": TIMEZONE,
        "domains": OPEN_METEO_DOMAIN,
    }

    response = requests.get(
        OPEN_METEO_URL,
        params=params,
        timeout=60,
    )
    response.raise_for_status()

    payload = response.json()

    if "hourly" not in payload:
        raise RuntimeError(
            f"Open-Meteo response does not contain 'hourly': {payload}"
        )

    df = pd.DataFrame(payload["hourly"])

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

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"Open-Meteo is missing columns: {missing}"
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    for col in ["pm25", "pm10", "no2", "so2", "co", "o3"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[df[col] < 0, col] = pd.NA

    # Exact same unit conversion used during training.
    # Open-Meteo CO: ug/m3 -> training/CPCB CO: mg/m3
    df["co"] = df["co"] / 1000.0

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
        .drop_duplicates(subset=["timestamp"], keep="last")
        .reset_index(drop=True)
    )

    # Keep only timestamps up to the latest available modeled hour.
    # forecast_days=1 is requested so the response remains available
    # near day boundaries; future rows are not written to the live DB.
    now_local = pd.Timestamp.now(tz=TIMEZONE).tz_localize(None)
    current_hour = now_local.floor("h")
    df = df[df["timestamp"] <= current_hour].copy()

    rows = []
    for _, r in df.iterrows():
        rows.append(
            {
                "timestamp": r["timestamp"].isoformat(),
                "pm25": float(r["pm25"]),
                "pm10": float(r["pm10"]),
                "no2": float(r["no2"]),
                "so2": float(r["so2"]),
                "co": float(r["co"]),
                "o3": float(r["o3"]),
                "aqi": None,
                "source": "Open-Meteo CAMS Global",
            }
        )

    return rows
