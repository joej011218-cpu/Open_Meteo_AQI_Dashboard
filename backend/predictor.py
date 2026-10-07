from datetime import datetime, timezone, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from config import (
    MODELS_DIR,
    FORECAST_HORIZONS,
    EXPECTED_FEATURE_COUNT,
)

from features import latest_feature_row


class XGBoostForecastService:
    def __init__(self):
        self.models = {}
        self.feature_columns = None
        self.configs = {}

        self.load_all_models()

    # ---------------------------------------------------------
    # LOAD ALL TRAINED XGBOOST MODELS
    # ---------------------------------------------------------

    def load_all_models(self):
        reference_features = None

        for horizon in FORECAST_HORIZONS:

            folder = (
                MODELS_DIR
                / f"horizon_{horizon:02d}h"
            )

            model_path = (
                folder
                / "xgboost_delta_model.joblib"
            )

            features_path = (
                folder
                / "feature_columns.joblib"
            )

            config_path = (
                folder
                / "model_config.joblib"
            )

            required = [
                model_path,
                features_path,
                config_path,
            ]

            missing = [
                str(path)
                for path in required
                if not path.exists()
            ]

            if missing:
                raise FileNotFoundError(
                    "Missing runtime model file(s) for "
                    f"horizon {horizon}: {missing}"
                )

            # Load trained XGBoost model
            model = joblib.load(
                model_path
            )

            # Load model feature list
            features = list(
                joblib.load(
                    features_path
                )
            )

            # Load saved model configuration
            config = joblib.load(
                config_path
            )

            # Verify expected number of features
            if len(features) != EXPECTED_FEATURE_COUNT:
                raise RuntimeError(
                    f"Horizon {horizon} has "
                    f"{len(features)} features; "
                    f"expected "
                    f"{EXPECTED_FEATURE_COUNT}."
                )

            # Verify model horizon
            saved_horizon = int(
                config.get(
                    "forecast_horizon",
                    horizon,
                )
            )

            if saved_horizon != horizon:
                raise RuntimeError(
                    f"Folder horizon_"
                    f"{horizon:02d}h contains "
                    f"a model_config for horizon "
                    f"{saved_horizon}."
                )

            # All 24 models must use
            # exactly the same feature columns.
            if reference_features is None:
                reference_features = features

            elif features != reference_features:
                raise RuntimeError(
                    f"Feature list for horizon "
                    f"{horizon} does not match "
                    f"the other models."
                )

            self.models[horizon] = model
            self.configs[horizon] = config

        self.feature_columns = (
            reference_features
        )

    # ---------------------------------------------------------
    # GENERATE 24-HOUR AQI FORECAST
    # ---------------------------------------------------------

    def predict_24h(self, df):

        if self.feature_columns is None:
            raise RuntimeError(
                "Models are not loaded."
            )

        # -----------------------------------------------------
        # PREPARE LATEST FEATURE VECTOR
        # -----------------------------------------------------

        X, latest = latest_feature_row(
            df,
            self.feature_columns,
        )

        current_aqi = float(
            latest["aqi"]
        )

        # -----------------------------------------------------
        # FORECAST ORIGIN
        # -----------------------------------------------------
        #
        # The forecast origin is the timestamp of the latest
        # Open-Meteo hourly observation available to the model.
        #
        # Example:
        #
        # latest Open-Meteo observation = 23:00
        #
        # horizon 1  -> 00:00
        # horizon 2  -> 01:00
        # ...
        # horizon 24 -> 23:00
        #
        # This timestamp is intentionally kept separate from
        # generated_at.
        # -----------------------------------------------------

        origin = pd.Timestamp(
            latest["timestamp"]
        )

        # Remove timezone information from origin if one was
        # unexpectedly supplied. The air_quality.timestamp
        # PostgreSQL column stores Open-Meteo local timestamps
        # as TIMESTAMP WITHOUT TIME ZONE.
        if origin.tzinfo is not None:
            origin = origin.tz_localize(None)

        # -----------------------------------------------------
        # ACTUAL FORECAST GENERATION TIME
        # -----------------------------------------------------
        #
        # generated_at must represent the actual instant when
        # this forecast batch was calculated.
        #
        # Store it as timezone-aware UTC because PostgreSQL
        # predictions.generated_at is TIMESTAMPTZ.
        # -----------------------------------------------------

        generated_at = datetime.now(
            timezone.utc
        )

        predictions = []

        # -----------------------------------------------------
        # RUN EACH OF THE 24 TRAINED XGBOOST MODELS
        # -----------------------------------------------------

        for horizon in FORECAST_HORIZONS:

            model = self.models[horizon]

            # Each model predicts:
            #
            # Future AQI - Current AQI
            #
            predicted_delta = float(
                model.predict(X)[0]
            )

            # Final AQI prediction:
            #
            # Current AQI + predicted delta
            #
            # AQI is constrained to 0-500.
            predicted_aqi = float(
                np.clip(
                    current_aqi
                    + predicted_delta,
                    0,
                    500,
                )
            )

            # Forecast target time
            forecast_time = (
                origin
                + timedelta(
                    hours=horizon
                )
            )

            predictions.append(
                {
                    # Actual time when this prediction
                    # batch was generated.
                    "generated_at":
                        generated_at.isoformat(),

                    # Hour being forecast.
                    "forecast_time":
                        forecast_time.isoformat(),

                    # +1 through +24
                    "horizon":
                        int(horizon),

                    # AQI at forecast origin
                    "current_aqi":
                        current_aqi,

                    # XGBoost model output
                    "predicted_delta":
                        predicted_delta,

                    # Final AQI prediction
                    "predicted_aqi":
                        predicted_aqi,

                    # Model identifier
                    "model":
                        "XGBoost_DeltaOnly",
                }
            )

        return predictions

    # ---------------------------------------------------------
    # MODEL INFORMATION
    # ---------------------------------------------------------

    def model_info(self):

        if not self.configs:
            return {
                "model": "XGBoost_DeltaOnly",
                "source": "Open-Meteo CAMS Global",
                "features": 0,
                "horizons": 0,
                "pollutants": [],
                "nh3_included": False,
                "target":
                    "Future AQI - Current AQI",
                "final_prediction":
                    "Current AQI + Predicted Delta",
            }

        # Horizon 1 configuration is used because all
        # 24 trained models share the same feature design.
        cfg = self.configs[1]

        return {
            "model": cfg.get(
                "model",
                "XGBoost_DeltaOnly",
            ),

            "source": cfg.get(
                "source",
                "Open-Meteo CAMS Global",
            ),

            "features": len(
                self.feature_columns
            ),

            "horizons": len(
                self.models
            ),

            "pollutants": cfg.get(
                "pollutants",
                [],
            ),

            "nh3_included": cfg.get(
                "nh3_included",
                False,
            ),

            "target": cfg.get(
                "target",
                "Future AQI - Current AQI",
            ),

            "final_prediction": cfg.get(
                "final_prediction",
                "Current AQI + Predicted Delta",
            ),
        }