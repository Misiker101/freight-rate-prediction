from __future__ import annotations

import pandas as pd

IMPUTE_COLUMNS = ["weight", "market_index"]

# Negative weights are a sign-entry error; recover the magnitude.
def fix_weight_sign(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["weight"] = df["weight"].abs()
    return df


def impute_with_reference(
    df: pd.DataFrame, reference_medians: dict[str, float]
) -> pd.DataFrame:
    
    df = df.copy()
    for col, median_value in reference_medians.items():
        if col in df.columns:
            df[col] = df[col].fillna(median_value)
    return df


def compute_reference_medians(df: pd.DataFrame, columns: list[str]) -> dict[str, float]:
    return {col: float(df[col].median()) for col in columns if col in df.columns}

