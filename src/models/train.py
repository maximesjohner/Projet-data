import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import joblib
import json
import sys

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import MODEL_CONFIG, PROJECT_ROOT, logger
from src.features.build_features import get_feature_columns


def create_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=["number", "bool", "int64", "float64", "Int64"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]), num_cols),
            ("cat", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]), cat_cols),
        ],
        remainder="drop"
    )
    return preprocessor


def train_model(X_train: pd.DataFrame, y_train: pd.Series, n_estimators: int = None, random_state: int = None) -> Pipeline:
    n_estimators = n_estimators or MODEL_CONFIG.n_estimators
    random_state = random_state or MODEL_CONFIG.random_state

    logger.info(f"=== ENTRAINEMENT DU MODELE ===")
    logger.info(f"Données d'entraînement: {len(X_train)} lignes, {len(X_train.columns)} features")
    logger.info(f"Paramètres: n_estimators={n_estimators}, random_state={random_state}")

    preprocessor = create_preprocessor(X_train)

    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)

    pipeline = Pipeline(steps=[("prep", preprocessor), ("rf", model)])

    logger.info("Entraînement du Random Forest en cours...")
    pipeline.fit(X_train, y_train)
    logger.info("Entraînement terminé avec succès!")

    return pipeline


def train_baseline(df: pd.DataFrame, target_col: str = "total_admissions") -> Dict[str, Any]:
    logger.info("Entraînement du modèle de base (saisonnier)...")

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found")

    overall_mean = df[target_col].mean()
    overall_std = df[target_col].std()
    dow_means = df.groupby("dow")[target_col].mean().to_dict()
    month_means = df.groupby("month")[target_col].mean().to_dict()

    baseline_params = {
        "type": "seasonal_naive",
        "overall_mean": overall_mean,
        "overall_std": overall_std,
        "dow_means": dow_means,
        "month_means": month_means,
        "window_size": 7
    }

    logger.info("Modèle de base entraîné")
    return baseline_params


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    logger.info(f"Évaluation du modèle sur {len(X_test)} lignes de test...")

    predictions = model.predict(X_test)

    metrics = {
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": np.sqrt(mean_squared_error(y_test, predictions)),
        "R2": r2_score(y_test, predictions)
    }

    logger.info(f"Résultats: MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}, R²={metrics['R2']:.2%}")
    return metrics


def save_model(model: Pipeline, name: str = "random_forest", metrics: Optional[Dict[str, float]] = None) -> str:
    models_dir = Path(MODEL_CONFIG.model_save_path)
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / f"{name}.joblib"
    joblib.dump(model, model_path)
    logger.info(f"Modèle sauvegardé: {model_path}")

    if metrics is not None:
        metrics_path = PROJECT_ROOT / "reports" / f"{name}_metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Métriques sauvegardées: {metrics_path}")

    return str(model_path)


def save_baseline(baseline_params: Dict[str, Any], name: str = "baseline") -> str:
    models_dir = Path(MODEL_CONFIG.model_save_path)
    models_dir.mkdir(parents=True, exist_ok=True)

    params_serializable = {}
    for key, value in baseline_params.items():
        if isinstance(value, dict):
            params_serializable[key] = {str(k): float(v) for k, v in value.items()}
        elif isinstance(value, (np.floating, np.integer)):
            params_serializable[key] = float(value)
        else:
            params_serializable[key] = value

    model_path = models_dir / f"{name}.json"
    with open(model_path, "w") as f:
        json.dump(params_serializable, f, indent=2)

    logger.info(f"Modèle de base sauvegardé: {model_path}")
    return str(model_path)
