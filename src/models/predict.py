import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Optional, Dict, Any
import joblib
import json
import sys

from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import MODEL_CONFIG, logger
from src.features.build_features import get_feature_columns


def load_model(name: str = "random_forest") -> Pipeline:
    model_path = Path(MODEL_CONFIG.model_save_path) / f"{name}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    logger.info(f"Chargement du modèle: {model_path.name}")
    return joblib.load(model_path)


def load_baseline(name: str = "baseline") -> Dict[str, Any]:
    model_path = Path(MODEL_CONFIG.model_save_path) / f"{name}.json"
    if not model_path.exists():
        raise FileNotFoundError(f"Baseline not found: {model_path}")
    with open(model_path, "r") as f:
        return json.load(f)


def predict(model: Union[Pipeline, str], X: pd.DataFrame) -> np.ndarray:
    if isinstance(model, str):
        model = load_model(model)
    feature_cols = get_feature_columns(X)
    return model.predict(X[feature_cols].copy())


def predict_baseline(baseline_params: Union[Dict[str, Any], str], df: pd.DataFrame) -> np.ndarray:
    if isinstance(baseline_params, str):
        baseline_params = load_baseline(baseline_params)

    predictions = []
    dow_means = baseline_params.get("dow_means", {})
    month_means = baseline_params.get("month_means", {})
    overall_mean = baseline_params.get("overall_mean", 400)

    for _, row in df.iterrows():
        dow = str(int(row.get("dow", 0)))
        month = str(int(row.get("month", 1)))
        pred = (dow_means.get(dow, overall_mean) + month_means.get(month, overall_mean)) / 2
        predictions.append(pred)
    return np.array(predictions)


def generate_future_dates(start_date: pd.Timestamp, n_days: int = 30) -> pd.DataFrame:
    dates = pd.date_range(start=start_date, periods=n_days, freq="D")
    df = pd.DataFrame({"date": dates})
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["dow"] = df["date"].dt.dayofweek
    df["week"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["day_of_year"] = df["date"].dt.dayofyear

    seasons = {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
               6: "summer", 7: "summer", 8: "summer", 9: "autumn", 10: "autumn", 11: "autumn"}
    df["season"] = df["month"].map(seasons)

    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    df["day_of_week"] = df["dow"].map(lambda x: days[x])
    return df


def forecast(model: Union[Pipeline, str], start_date: pd.Timestamp, n_days: int = 30, scenario_features: Optional[Dict[str, Any]] = None, hospital_id: str = "PITIE") -> pd.DataFrame:
    logger.info(f"Génération des prévisions: {n_days} jours à partir du {start_date.strftime('%Y-%m-%d')} (hôpital: {hospital_id})")

    future_df = generate_future_dates(start_date, n_days)

    future_df["hospital_id"] = hospital_id

    month_temps = {1: 4, 2: 5, 3: 9, 4: 13, 5: 17, 6: 21, 7: 24, 8: 23, 9: 19, 10: 13, 11: 8, 12: 5}
    future_df["temperature_c"] = future_df["month"].map(month_temps).astype(float)

    month_surgeries = {1: 42, 2: 44, 3: 45, 4: 43, 5: 40, 6: 35, 7: 30, 8: 28, 9: 38, 10: 42, 11: 44, 12: 40}
    future_df["scheduled_surgeries"] = future_df["month"].map(month_surgeries)
    future_df.loc[future_df["is_weekend"] == 1, "scheduled_surgeries"] = 20

    defaults = {
        "epidemic_level": 0, "heatwave_event": 0, "strike_level": 0,
        "staff_absence_rate": 0.06, "available_staff": 420, "available_beds": 1420,
        "medical_stock_level_pct": 72.0, "accident_event": 0, "external_alert_level": 0
    }

    for col, val in defaults.items():
        if col not in future_df.columns:
            future_df[col] = val

    if scenario_features:
        for col, val in scenario_features.items():
            future_df[col] = val

    future_df["predicted_admissions"] = predict(model, future_df)
    logger.info(f"Prévisions: moyenne={future_df['predicted_admissions'].mean():.0f}/jour")
    return future_df
