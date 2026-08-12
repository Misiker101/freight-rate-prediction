from __future__ import annotations

import numpy as np
import pandas as pd

CATEGORICAL_COLUMNS = ["pickup", "delivery", "equipment"]

NUMERIC_COLUMNS = [
    "distance",
    "weight",
    "market_index",
    "quote_signal",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
]

DATE_FEATURE_COLUMNS = [
    "month",
    "day_of_week",
    "day_of_year",
    "week_of_year",
    "is_weekend",
    "month_sin",
    "month_cos",
    "doy_sin",
    "doy_cos",
]

FEATURE_COLUMNS = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS + DATE_FEATURE_COLUMNS


def build_city_coords(df: pd.DataFrame) -> pd.DataFrame:
    
    pickup_side = df[["pickup", "pickup_lat", "pickup_lon"]].rename(
        columns={"pickup": "city", "pickup_lat": "lat", "pickup_lon": "lon"}
    )
    delivery_side = df[["delivery", "delivery_lat", "delivery_lon"]].rename(
        columns={"delivery": "city", "delivery_lat": "lat", "delivery_lon": "lon"}
    )
    combined = pd.concat([pickup_side, delivery_side], ignore_index=True)
    return combined.groupby("city", as_index=False).first()


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dates = pd.to_datetime(df["date"])
    df["month"] = dates.dt.month
    df["day_of_week"] = dates.dt.dayofweek
    df["day_of_year"] = dates.dt.dayofyear
    df["week_of_year"] = dates.dt.isocalendar().week.astype(int)
    df["is_weekend"] = (dates.dt.dayofweek >= 5).astype(int)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)
    return df


def fill_missing_coords(df: pd.DataFrame, city_coords: pd.DataFrame) -> pd.DataFrame:
    
    df = df.copy()
    lookup = city_coords.set_index("city")[["lat", "lon"]]

    for side in ["pickup", "delivery"]:
        lat_col, lon_col = f"{side}_lat", f"{side}_lon"
        if lat_col not in df.columns:
            df[lat_col] = np.nan
        if lon_col not in df.columns:
            df[lon_col] = np.nan
        matched_lat = df[side].map(lookup["lat"])
        df[lat_col] = df[lat_col].fillna(matched_lat)
        matched_lon = df[side].map(lookup["lon"])
        df[lon_col] = df[lon_col].fillna(matched_lon)
    return df


# Hold market_index / quote_signal at their trained-on median if absent.
def fill_missing_market_features(
    df: pd.DataFrame, fallback_values: dict[str, float]
) -> pd.DataFrame:
    
    df = df.copy()
    for col, value in fallback_values.items():
        if col not in df.columns:
            df[col] = value
        else:
            df[col] = df[col].fillna(value)
    return df


def set_categorical_dtypes(df: pd.DataFrame, category_levels: dict[str, list]) -> pd.DataFrame:
    df = df.copy()
    for col, levels in category_levels.items():
        df[col] = pd.Categorical(df[col], categories=levels)
    return df


def build_feature_matrix(
    df: pd.DataFrame,
    city_coords: pd.DataFrame,
    category_levels: dict[str, list],
    market_fallback: dict[str, float],
) -> pd.DataFrame:
    df = df.copy()
    df = fill_missing_coords(df, city_coords)
    df = fill_missing_market_features(df, market_fallback)
    df = add_date_features(df)
    df = set_categorical_dtypes(df, category_levels)
    return df[FEATURE_COLUMNS]
