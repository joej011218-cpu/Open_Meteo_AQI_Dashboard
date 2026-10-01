from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "air_quality.db"

# Vadodara
LATITUDE = 22.3072
LONGITUDE = 73.1812

# Keep the runtime source consistent with training.
TIMEZONE = "Asia/Kolkata"
OPEN_METEO_DOMAIN = "cams_global"

OPEN_METEO_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

# 30 days is comfortably larger than the longest model lag (168 h)
# plus the 24-hour AQI rolling requirement.
BOOTSTRAP_PAST_DAYS = 30

POLLUTANTS = ["pm25", "pm10", "no2", "so2", "co", "o3"]

AQI_LAGS = [1, 2, 3, 6, 12, 24, 48, 72, 168]
POLLUTANT_LAGS = [1, 2, 3, 6, 12, 24, 48, 72, 168]
ROLLING_WINDOWS = [3, 6, 12, 24]
TREND_WINDOWS = [3, 6, 12]

EXPECTED_FEATURE_COUNT = 198
FORECAST_HORIZONS = range(1, 25)

# CORS origins for local frontend development.
CORS_ORIGINS = [
    "http://127.0.0.1:5500",
    "http://localhost:5500",
    "http://127.0.0.1:8000",
    "http://localhost:8000",
]
