import numpy as np
import pandas as pd

from config import (
    AQI_LAGS,
    POLLUTANTS,
    POLLUTANT_LAGS,
    ROLLING_WINDOWS,
    TREND_WINDOWS,
    EXPECTED_FEATURE_COUNT,
)


CHANGE_LAGS = [1, 3, 6, 12, 24]


def create_features(df):
    """
    Reproduce the exact 198-feature engineering order used during training.
    """
    data = (
        df.copy()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    feature_columns = []

    # AQI lags
    for lag in AQI_LAGS:
        name = f"aqi_lag_{lag}"
        data[name] = data["aqi"].shift(lag)
        feature_columns.append(name)

    # Pollutant lags
    for pollutant in POLLUTANTS:
        for lag in POLLUTANT_LAGS:
            name = f"{pollutant}_lag_{lag}"
            data[name] = data[pollutant].shift(lag)
            feature_columns.append(name)

    # AQI rolling stats
    for window in ROLLING_WINDOWS:
        prefix = f"aqi_roll_{window}h"

        data[f"{prefix}_mean"] = (
            data["aqi"]
            .rolling(window=window, min_periods=window)
            .mean()
        )

        data[f"{prefix}_max"] = (
            data["aqi"]
            .rolling(window=window, min_periods=window)
            .max()
        )

        data[f"{prefix}_min"] = (
            data["aqi"]
            .rolling(window=window, min_periods=window)
            .min()
        )

        data[f"{prefix}_std"] = (
            data["aqi"]
            .rolling(window=window, min_periods=window)
            .std()
        )

        feature_columns.extend(
            [
                f"{prefix}_mean",
                f"{prefix}_max",
                f"{prefix}_min",
                f"{prefix}_std",
            ]
        )

    # Pollutant rolling stats
    for pollutant in POLLUTANTS:
        for window in ROLLING_WINDOWS:
            prefix = f"{pollutant}_roll_{window}h"

            data[f"{prefix}_mean"] = (
                data[pollutant]
                .rolling(window=window, min_periods=window)
                .mean()
            )

            data[f"{prefix}_std"] = (
                data[pollutant]
                .rolling(window=window, min_periods=window)
                .std()
            )

            feature_columns.extend(
                [
                    f"{prefix}_mean",
                    f"{prefix}_std",
                ]
            )

    # Pollutant changes
    for pollutant in POLLUTANTS:
        for lag in CHANGE_LAGS:
            name = f"{pollutant}_change_{lag}h"
            data[name] = (
                data[pollutant]
                - data[pollutant].shift(lag)
            )
            feature_columns.append(name)

    # AQI changes
    for lag in CHANGE_LAGS:
        name = f"aqi_change_{lag}h"
        data[name] = (
            data["aqi"]
            - data["aqi"].shift(lag)
        )
        feature_columns.append(name)

    # Pollutant trends
    for pollutant in POLLUTANTS:
        for window in TREND_WINDOWS:
            name = f"{pollutant}_trend_{window}h"
            data[name] = (
                data[pollutant].diff(window)
                / window
            )
            feature_columns.append(name)

    # AQI trends
    for window in TREND_WINDOWS:
        name = f"aqi_trend_{window}h"
        data[name] = (
            data["aqi"].diff(window)
            / window
        )
        feature_columns.append(name)

    # Pollutant acceleration
    for pollutant in POLLUTANTS:
        change_1h = data[pollutant].diff(1)
        change_3h = data[pollutant].diff(3)

        name = f"{pollutant}_acceleration"
        data[name] = (
            change_1h
            - change_3h / 3
        )
        feature_columns.append(name)

    # AQI acceleration
    data["aqi_acceleration"] = (
        data["aqi"].diff(1)
        - data["aqi"].diff(3) / 3
    )
    feature_columns.append("aqi_acceleration")

    # PM ratio
    data["pm25_pm10_ratio"] = (
        data["pm25"]
        / data["pm10"].replace(0, np.nan)
    )
    feature_columns.append("pm25_pm10_ratio")

    # PM2.5 dominance
    data["pm25_dominance"] = (
        data["pm25"]
        / (
            data["pm25"]
            + data["pm10"]
            + 1e-6
        )
    )
    feature_columns.append("pm25_dominance")

    # Cyclical time features
    hour = data["timestamp"].dt.hour
    day_of_week = data["timestamp"].dt.dayofweek
    day_of_year = data["timestamp"].dt.dayofyear

    data["sin_hour"] = np.sin(2 * np.pi * hour / 24)
    data["cos_hour"] = np.cos(2 * np.pi * hour / 24)

    data["sin_day"] = np.sin(
        2 * np.pi * day_of_week / 7
    )
    data["cos_day"] = np.cos(
        2 * np.pi * day_of_week / 7
    )

    data["sin_year"] = np.sin(
        2 * np.pi * day_of_year / 365.25
    )
    data["cos_year"] = np.cos(
        2 * np.pi * day_of_year / 365.25
    )

    feature_columns.extend(
        [
            "sin_hour",
            "cos_hour",
            "sin_day",
            "cos_day",
            "sin_year",
            "cos_year",
        ]
    )

    for col in feature_columns:
        data[col] = pd.to_numeric(
            data[col],
            errors="coerce",
        )

    data[feature_columns] = (
        data[feature_columns]
        .replace([np.inf, -np.inf], np.nan)
    )

    if len(feature_columns) != EXPECTED_FEATURE_COUNT:
        raise RuntimeError(
            f"Feature generator created {len(feature_columns)} features; "
            f"expected {EXPECTED_FEATURE_COUNT}."
        )

    return data, feature_columns


def latest_feature_row(df, expected_feature_columns):
    data, generated_columns = create_features(df)

    if generated_columns != list(expected_feature_columns):
        # Show the first mismatch to make debugging easy.
        for i, (generated, expected) in enumerate(
            zip(generated_columns, expected_feature_columns)
        ):
            if generated != expected:
                raise RuntimeError(
                    f"Feature order mismatch at index {i}: "
                    f"generated={generated!r}, expected={expected!r}"
                )

        raise RuntimeError(
            "Feature list differs from the saved training feature list."
        )

    latest = data.iloc[-1]

    missing = [
        col
        for col in expected_feature_columns
        if pd.isna(latest[col])
    ]

    if missing:
        raise RuntimeError(
            "Latest row cannot be predicted because these engineered "
            f"features are missing: {missing[:15]}"
            + (" ..." if len(missing) > 15 else "")
        )

    X = (
        latest[list(expected_feature_columns)]
        .astype(np.float32)
        .to_numpy()
        .reshape(1, -1)
    )

    return X, latest
