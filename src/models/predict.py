"""
Prediction utilities for the Hospital Decision Support System.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Optional, Dict, Any
import joblib
import json
import sys

from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import MODEL_CONFIG, PROJECT_ROOT
from src.features.build_features import get_feature_columns


def load_model(name: str = "random_forest") -> Pipeline:
    """
    Load a trained model from disk.

    Args:
        name: Name of the saved model.

    Returns:
        Loaded sklearn Pipeline.

    Raises:
        FileNotFoundError: If model file doesn't exist.
    """
    models_dir = Path(MODEL_CONFIG.model_save_path)
    model_path = models_dir / f"{name}.joblib"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    return joblib.load(model_path)


def load_baseline(name: str = "baseline") -> Dict[str, Any]:
    """
    Load baseline model parameters from disk.

    Args:
        name: Name of the saved baseline model.

    Returns:
        Baseline model parameters.

    Raises:
        FileNotFoundError: If model file doesn't exist.
    """
    models_dir = Path(MODEL_CONFIG.model_save_path)
    model_path = models_dir / f"{name}.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Baseline model not found: {model_path}")

    with open(model_path, "r") as f:
        return json.load(f)


def predict(
    model: Union[Pipeline, str],
    X: pd.DataFrame
) -> np.ndarray:
    """
    Make predictions using a trained model.

    Args:
        model: Trained Pipeline or model name string.
        X: Feature DataFrame for prediction.

    Returns:
        Array of predictions.
    """
    if isinstance(model, str):
        model = load_model(model)

    # Ensure we have the right columns
    feature_cols = get_feature_columns(X)
    X_pred = X[feature_cols].copy()

    predictions = model.predict(X_pred)

    return predictions


def predict_baseline(
    baseline_params: Union[Dict[str, Any], str],
    df: pd.DataFrame
) -> np.ndarray:
    """
    Make predictions using the baseline model.

    Uses day-of-week and monthly seasonality.

    Args:
        baseline_params: Baseline parameters or model name string.
        df: DataFrame with date/temporal info.

    Returns:
        Array of baseline predictions.
    """
    if isinstance(baseline_params, str):
        baseline_params = load_baseline(baseline_params)

    predictions = []
    dow_means = baseline_params.get("dow_means", {})
    month_means = baseline_params.get("month_means", {})
    overall_mean = baseline_params.get("overall_mean", 400)

    for _, row in df.iterrows():
        dow = str(int(row.get("dow", 0)))
        month = str(int(row.get("month", 1)))

        # Combine day-of-week and month effects
        dow_effect = dow_means.get(dow, overall_mean)
        month_effect = month_means.get(month, overall_mean)

        # Simple combination: average of effects
        pred = (dow_effect + month_effect) / 2

        predictions.append(pred)

    return np.array(predictions)


def generate_future_dates(
    start_date: pd.Timestamp,
    n_days: int = 30
) -> pd.DataFrame:
    """
    Generate a DataFrame with future dates for forecasting.

    Args:
        start_date: Starting date for forecast.
        n_days: Number of days to forecast.

    Returns:
        DataFrame with date and temporal features.
    """
    dates = pd.date_range(start=start_date, periods=n_days, freq="D")

    df = pd.DataFrame({"date": dates})
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["dow"] = df["date"].dt.dayofweek
    df["week"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["day_of_year"] = df["date"].dt.dayofyear

    # Add season based on month
    def get_season(month):
        if month in [12, 1, 2]:
            return "winter"
        elif month in [3, 4, 5]:
            return "spring"
        elif month in [6, 7, 8]:
            return "summer"
        else:
            return "autumn"

    df["season"] = df["month"].apply(get_season)

    # Add day_of_week name
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday",
                 "Friday", "Saturday", "Sunday"]
    df["day_of_week"] = df["dow"].apply(lambda x: day_names[x])

    return df


def forecast(
    model: Union[Pipeline, str],
    start_date: pd.Timestamp,
    n_days: int = 30,
    scenario_features: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Generate forecasts for future dates.

    Args:
        model: Trained model or model name.
        start_date: Starting date for forecast.
        n_days: Number of days to forecast.
        scenario_features: Optional dict of feature values for scenarios.

    Returns:
        DataFrame with dates and predictions.
    """
    # Generate future dates
    future_df = generate_future_dates(start_date, n_days)

    # Add default values for required features
    default_values = {
        "epidemic_level": 0,
        "temperature_c": 15.0,
        "heatwave_event": 0,
        "strike_level": 0,
        "staff_absence_rate": 0.06,
        "available_staff": 414,
        "available_beds": 1400,
        "scheduled_surgeries": 40,
        "medical_stock_level_pct": 75.0,
        "accident_event": 0,
        "external_alert_level": 0
    }

    for col, value in default_values.items():
        if col not in future_df.columns:
            future_df[col] = value

    # Apply scenario features if provided
    if scenario_features:
        for col, value in scenario_features.items():
            if col in future_df.columns or col in default_values:
                future_df[col] = value

    # Make predictions
    predictions = predict(model, future_df)
    future_df["predicted_admissions"] = predictions

    return future_df
