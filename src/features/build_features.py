import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import MODEL_CONFIG, logger


def get_feature_columns(df: Optional[pd.DataFrame] = None) -> List[str]:
    features = MODEL_CONFIG.feature_columns.copy()
    if df is not None:
        features = [col for col in features if col in df.columns]
    return features


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Construction des features...")
    df = df.copy()

    if "date" in df.columns:
        if "dow" not in df.columns:
            df["dow"] = df["date"].dt.dayofweek
        if "month" not in df.columns:
            df["month"] = df["date"].dt.month
        if "is_weekend" not in df.columns:
            df["is_weekend"] = (df["dow"] >= 5).astype(int)

    if "available_staff" in df.columns and "staff_absence_rate" in df.columns:
        df["effective_staff"] = df["available_staff"] * (1 - df["staff_absence_rate"])

    if "epidemic_level" in df.columns and "staff_absence_rate" in df.columns:
        df["stress_indicator"] = df["epidemic_level"] * df["staff_absence_rate"]

    if "available_beds" in df.columns:
        df["bed_utilization_proxy"] = 1500 / (df["available_beds"] + 1)

    logger.info(f"Features construites: {len(get_feature_columns(df))} colonnes")
    return df


def prepare_model_data(df: pd.DataFrame, target_col: str = "total_admissions") -> Tuple[pd.DataFrame, pd.Series]:
    feature_cols = get_feature_columns(df)
    X = df[feature_cols].copy()
    y = df[target_col].copy() if target_col in df.columns else None
    return X, y
