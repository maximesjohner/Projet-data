"""
Feature engineering for the Hospital Decision Support System.
"""
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import MODEL_CONFIG


def get_feature_columns(df: Optional[pd.DataFrame] = None) -> List[str]:
    """
    Get the list of feature columns to use for modeling.

    Only returns columns that exist in the provided DataFrame.

    Args:
        df: Optional DataFrame to check column existence.

    Returns:
        List of feature column names.
    """
    features = MODEL_CONFIG.feature_columns.copy()

    if df is not None:
        features = [col for col in features if col in df.columns]

    return features


def build_features(
    df: pd.DataFrame,
    include_lags: bool = False,
    include_rolling: bool = False
) -> pd.DataFrame:
    """
    Build features for the prediction model.

    Args:
        df: Preprocessed DataFrame.
        include_lags: If True, include lagged features.
        include_rolling: If True, include rolling statistics.

    Returns:
        pd.DataFrame with engineered features.
    """
    df = df.copy()

    # Ensure temporal features exist
    if "date" in df.columns:
        if "dow" not in df.columns:
            df["dow"] = df["date"].dt.dayofweek
        if "month" not in df.columns:
            df["month"] = df["date"].dt.month
        if "is_weekend" not in df.columns:
            df["is_weekend"] = (df["dow"] >= 5).astype(int)

    # Create interaction features
    df = create_interaction_features(df)

    # Add lag features if requested
    if include_lags:
        df = create_lag_features(df)

    # Add rolling features if requested
    if include_rolling:
        df = create_rolling_features(df)

    return df


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create interaction features between variables.

    Args:
        df: DataFrame with base features.

    Returns:
        DataFrame with added interaction features.
    """
    df = df.copy()

    # Staff availability adjusted for absence
    if "available_staff" in df.columns and "staff_absence_rate" in df.columns:
        df["effective_staff"] = df["available_staff"] * (1 - df["staff_absence_rate"])

    # Stress indicator (high epidemic + low staff)
    if "epidemic_level" in df.columns and "staff_absence_rate" in df.columns:
        df["stress_indicator"] = (
            df["epidemic_level"] * df["staff_absence_rate"]
        )

    # Capacity utilization estimate
    if "available_beds" in df.columns:
        df["bed_utilization_proxy"] = 1500 / (df["available_beds"] + 1)

    return df


def create_lag_features(
    df: pd.DataFrame,
    target_col: str = "total_admissions",
    lags: List[int] = [1, 7, 14]
) -> pd.DataFrame:
    """
    Create lagged features for time series prediction.

    Note: This introduces NaN values at the beginning of the series.

    Args:
        df: DataFrame with the target column.
        target_col: Name of the target column.
        lags: List of lag periods to create.

    Returns:
        DataFrame with lag features.
    """
    df = df.copy()

    if target_col not in df.columns:
        return df

    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)

    return df


def create_rolling_features(
    df: pd.DataFrame,
    target_col: str = "total_admissions",
    windows: List[int] = [7, 14, 30]
) -> pd.DataFrame:
    """
    Create rolling statistics features.

    Note: This introduces NaN values at the beginning of the series.

    Args:
        df: DataFrame with the target column.
        target_col: Name of the target column.
        windows: List of window sizes for rolling statistics.

    Returns:
        DataFrame with rolling features.
    """
    df = df.copy()

    if target_col not in df.columns:
        return df

    for window in windows:
        df[f"{target_col}_rolling_mean_{window}"] = (
            df[target_col].rolling(window=window).mean()
        )
        df[f"{target_col}_rolling_std_{window}"] = (
            df[target_col].rolling(window=window).std()
        )

    return df


def prepare_model_data(
    df: pd.DataFrame,
    target_col: str = "total_admissions"
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare data for model training/prediction.

    Args:
        df: Preprocessed DataFrame with features.
        target_col: Name of the target column.

    Returns:
        Tuple of (X features DataFrame, y target Series).
    """
    feature_cols = get_feature_columns(df)

    X = df[feature_cols].copy()
    y = df[target_col].copy() if target_col in df.columns else None

    return X, y
