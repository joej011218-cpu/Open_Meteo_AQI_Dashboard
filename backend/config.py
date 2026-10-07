import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

# ---------------------------------------------------------
# DATABASE
# ---------------------------------------------------------

# PostgreSQL connection URL.
#
# On Render this must be configured as:
# DATABASE_URL=<Render Internal Database URL>
#
# Never hard-code the database password or URL here.
DATABASE_URL = os.getenv("DATABASE_URL")

# Keep one year of hourly air-quality history.
DATA_RETENTION_DAYS = 365


# ---------------------------------------------------------
# LOCATION
# ---------------------------------------------------------

# Vadodara, Gujarat, India
LATITUDE = 22.3072
LONGITUDE = 73.1812

TIMEZONE = "Asia/Kolkata"


# ---------------------------------------------------------
# OPEN-METEO
# ---------------------------------------------------------

# Keep the runtime source consistent with the
# dataset used for model training.
OPEN_METEO_DOMAIN = "cams_global"

OPEN_METEO_URL = (
    "https://air-quality-api.open-meteo.com/"
    "v1/air-quality"
)

# Initial/bootstrap history.
#
# 30 days provides much more history than the
# longest model lag (168 hours) and the 24-hour
# AQI rolling requirement.
BOOTSTRAP_PAST_DAYS = 30


# ---------------------------------------------------------
# POLLUTANTS
# ---------------------------------------------------------

POLLUTANTS = [
    "pm25",
    "pm10",
    "no2",
    "so2",
    "co",
    "o3",
]


# ---------------------------------------------------------
# MODEL FEATURES
# ---------------------------------------------------------

AQI_LAGS = [
    1,
    2,
    3,
    6,
    12,
    24,
    48,
    72,
    168,
]

POLLUTANT_LAGS = [
    1,
    2,
    3,
    6,
    12,
    24,
    48,
    72,
    168,
]

ROLLING_WINDOWS = [
    3,
    6,
    12,
    24,
]

TREND_WINDOWS = [
    3,
    6,
    12,
]

EXPECTED_FEATURE_COUNT = 198

FORECAST_HORIZONS = range(
    1,
    25,
)


# ---------------------------------------------------------
# CORS
# ---------------------------------------------------------

# These remain useful during local frontend development.
CORS_ORIGINS = [
    "http://127.0.0.1:5500",
    "http://localhost:5500",
    "http://127.0.0.1:8000",
    "http://localhost:8000",
]