from datetime import timedelta
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

            model = joblib.load(model_path)
            features = list(
                joblib.load(features_path)
            )
            config = joblib.load(config_path)

            if len(features) != EXPECTED_FEATURE_COUNT:
                raise RuntimeError(
                    f"Horizon {horizon} has {len(features)} features; "
                    f"expected {EXPECTED_FEATURE_COUNT}."
                )

            saved_horizon = int(
                config.get(
                    "forecast_horizon",
                    horizon,
                )
            )

            if saved_horizon != horizon:
                raise RuntimeError(
                    f"Folder horizon_{horizon:02d}h contains a "
                    f"model_config for horizon {saved_horizon}."
                )

            if reference_features is None:
                reference_features = features
            elif features != reference_features:
                raise RuntimeError(
                    f"Feature list for horizon {horizon} does not "
                    "match the other models."
                )

            self.models[horizon] = model
            self.configs[horizon] = config

        self.feature_columns = reference_features

    def predict_24h(self, df):
        if self.feature_columns is None:
            raise RuntimeError(
                "Models are not loaded."
            )

        X, latest = latest_feature_row(
            df,
            self.feature_columns,
        )

        current_aqi = float(
            latest["aqi"]
        )

        origin = pd.Timestamp(
            latest["timestamp"]
        )

        predictions = []

        for horizon in FORECAST_HORIZONS:
            model = self.models[horizon]

            predicted_delta = float(
                model.predict(X)[0]
            )

            predicted_aqi = float(
                np.clip(
                    current_aqi
                    + predicted_delta,
                    0,
                    500,
                )
            )

            forecast_time = (
                origin
                + timedelta(hours=horizon)
            )

            predictions.append(
                {
                    "generated_at":
                        origin.isoformat(),

                    "forecast_time":
                        forecast_time.isoformat(),

                    "horizon":
                        horizon,

                    "current_aqi":
                        current_aqi,

                    "predicted_delta":
                        predicted_delta,

                    "predicted_aqi":
                        predicted_aqi,

                    "model":
                        "XGBoost_DeltaOnly",
                }
            )

        return predictions

    def model_info(self):
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
            "horizons": 24,
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
