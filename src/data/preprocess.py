"""
Data preprocessing utilities for the Hospital Decision Support System.
"""
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw hospital data.

    - Removes duplicates
    - Handles missing values
    - Converts data types appropriately

    Args:
        df: Raw DataFrame to clean.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    df = df.copy()

    # Remove duplicates
    df = df.drop_duplicates()

    # Define column types
    categorical_cols = ["day_of_week", "season"]
    boolean_like_cols = [
        "heatwave_event", "accident_event",
        "supply_delivery_day", "it_system_outage"
    ]

    # Convert categorical columns
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # Convert boolean-like columns
    for col in boolean_like_cols:
        if col in df.columns:
            df[col] = df[col].astype("Int64")

    return df


def preprocess_data(
    df: pd.DataFrame,
    add_time_features: bool = True,
    fill_missing: bool = True
) -> pd.DataFrame:
    """
    Preprocess the hospital data for modeling.

    Args:
        df: DataFrame to preprocess.
        add_time_features: If True, add time-based features.
        fill_missing: If True, fill missing values.

    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    df = clean_data(df)

    if add_time_features and "date" in df.columns:
        df = add_temporal_features(df)

    if fill_missing:
        df = fill_missing_values(df)

    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add temporal features derived from the date column.

    Args:
        df: DataFrame with a 'date' column.

    Returns:
        pd.DataFrame: DataFrame with additional temporal features.
    """
    df = df.copy()

    if "date" not in df.columns:
        return df

    # Ensure date is datetime
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Extract temporal features
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["dow"] = df["date"].dt.dayofweek  # Monday=0
    df["week"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["dow"] >= 5).astype(int)

    # Day of year for seasonality
    df["day_of_year"] = df["date"].dt.dayofyear

    return df


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values using appropriate strategies.

    Args:
        df: DataFrame with potential missing values.

    Returns:
        pd.DataFrame: DataFrame with filled missing values.
    """
    df = df.copy()

    # Numeric columns: fill with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    # Categorical columns: fill with mode
    categorical_cols = df.select_dtypes(include=["category", "object"]).columns
    for col in categorical_cols:
        if df[col].isna().any():
            mode_val = df[col].mode()
            if len(mode_val) > 0:
                df[col] = df[col].fillna(mode_val[0])

    return df


def get_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    date_column: str = "date"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data temporally (no shuffle) to avoid data leakage.

    Args:
        df: DataFrame to split.
        test_size: Proportion of data for testing.
        date_column: Name of the date column for sorting.

    Returns:
        Tuple of (train_df, test_df).
    """
    df = df.sort_values(date_column).reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_size))

    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    return train_df, test_df
