import pandas as pd
import numpy as np
from typing import Tuple
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import logger


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().drop_duplicates()
    for col in ["day_of_week", "season"]:
        if col in df.columns:
            df[col] = df[col].astype("category")
    for col in ["heatwave_event", "accident_event", "supply_delivery_day", "it_system_outage"]:
        if col in df.columns:
            df[col] = df[col].astype("Int64")
    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "date" not in df.columns:
        return df
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["dow"] = df["date"].dt.dayofweek
    df["week"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["day_of_year"] = df["date"].dt.dayofyear
    return df


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include=["category", "object"]).columns:
        if df[col].isna().any():
            mode_val = df[col].mode()
            if len(mode_val) > 0:
                df[col] = df[col].fillna(mode_val[0])
    return df


def preprocess_data(df: pd.DataFrame, add_time_features: bool = True, fill_missing: bool = True) -> pd.DataFrame:
    logger.info("Prétraitement des données...")
    df = clean_data(df)
    if add_time_features and "date" in df.columns:
        df = add_temporal_features(df)
    if fill_missing:
        df = fill_missing_values(df)
    logger.info("Prétraitement terminé")
    return df


def get_train_test_split(df: pd.DataFrame, test_size: float = 0.2, date_column: str = "date") -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values(date_column).reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    logger.info(f"Split train/test: {len(train_df)} / {len(test_df)} lignes")
    return train_df, test_df
