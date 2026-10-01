import numpy as np
import pandas as pd


PM25_BREAKPOINTS = [
    (0, 30, 0, 50),
    (31, 60, 51, 100),
    (61, 90, 101, 200),
    (91, 120, 201, 300),
    (121, 250, 301, 400),
    (251, np.inf, 401, 500),
]

PM10_BREAKPOINTS = [
    (0, 50, 0, 50),
    (51, 100, 51, 100),
    (101, 250, 101, 200),
    (251, 350, 201, 300),
    (351, 430, 301, 400),
    (431, np.inf, 401, 500),
]

NO2_BREAKPOINTS = [
    (0, 40, 0, 50),
    (41, 80, 51, 100),
    (81, 180, 101, 200),
    (181, 280, 201, 300),
    (281, 400, 301, 400),
    (401, np.inf, 401, 500),
]

SO2_BREAKPOINTS = [
    (0, 40, 0, 50),
    (41, 80, 51, 100),
    (81, 380, 101, 200),
    (381, 800, 201, 300),
    (801, 1600, 301, 400),
    (1601, np.inf, 401, 500),
]

# CO is mg/m3.
CO_BREAKPOINTS = [
    (0, 1.0, 0, 50),
    (1.1, 2.0, 51, 100),
    (2.1, 10.0, 101, 200),
    (10.1, 17.0, 201, 300),
    (17.1, 34.0, 301, 400),
    (34.1, np.inf, 401, 500),
]

O3_BREAKPOINTS = [
    (0, 50, 0, 50),
    (51, 100, 51, 100),
    (101, 168, 101, 200),
    (169, 208, 201, 300),
    (209, 748, 301, 400),
    (749, np.inf, 401, 500),
]


def calculate_subindex(concentration, breakpoints):
    """
    Kept consistent with the training script.
    """
    if pd.isna(concentration):
        return np.nan

    concentration = float(concentration)

    if concentration < 0:
        return np.nan

    for c_low, c_high, i_low, i_high in breakpoints:
        if concentration <= c_high:
            if np.isinf(c_high):
                return float(i_low)

            return (
                ((i_high - i_low) / (c_high - c_low))
                * (concentration - c_low)
                + i_low
            )

    return 500.0


def add_cpcb_rolling_values(df):
    data = (
        df.copy()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    for col in ["pm25", "pm10", "no2", "so2"]:
        data[f"{col}_24h"] = (
            data[col]
            .rolling(window=24, min_periods=24)
            .mean()
        )

    for col in ["co", "o3"]:
        data[f"{col}_8h"] = (
            data[col]
            .rolling(window=8, min_periods=8)
            .mean()
        )

    return data


def calculate_row_aqi(row):
    values = [
        calculate_subindex(row["pm25_24h"], PM25_BREAKPOINTS),
        calculate_subindex(row["pm10_24h"], PM10_BREAKPOINTS),
        calculate_subindex(row["no2_24h"], NO2_BREAKPOINTS),
        calculate_subindex(row["so2_24h"], SO2_BREAKPOINTS),
        calculate_subindex(row["co_8h"], CO_BREAKPOINTS),
        calculate_subindex(row["o3_8h"], O3_BREAKPOINTS),
    ]

    valid = [v for v in values if not pd.isna(v)]
    return max(valid) if valid else np.nan


def calculate_aqi_dataframe(df):
    data = add_cpcb_rolling_values(df)

    data["aqi"] = data.apply(
        calculate_row_aqi,
        axis=1,
    )

    data["aqi"] = pd.to_numeric(
        data["aqi"],
        errors="coerce",
    )

    return data
