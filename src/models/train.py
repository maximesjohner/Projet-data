"""
Model training utilities for the Hospital Decision Support System.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import joblib
import json
import sys

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import MODEL_CONFIG, PROJECT_ROOT
from src.features.build_features import get_feature_columns


def create_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Create a preprocessing pipeline for the model.

    Args:
        X: Feature DataFrame to determine column types.

    Returns:
        ColumnTransformer for preprocessing.
    """
    # Identify column types
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(
        include=["number", "bool", "int64", "float64", "Int64"]
    ).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median"))
            ]), num_cols),
            ("cat", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]), cat_cols),
        ],
        remainder="drop"
    )

    return preprocessor


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = None,
    random_state: int = None
) -> Pipeline:
    """
    Train a Random Forest model with preprocessing.

    Args:
        X_train: Training features.
        y_train: Training target.
        n_estimators: Number of trees. Defaults to config value.
        random_state: Random seed. Defaults to config value.

    Returns:
        Trained sklearn Pipeline.
    """
    n_estimators = n_estimators or MODEL_CONFIG.n_estimators
    random_state = random_state or MODEL_CONFIG.random_state

    preprocessor = create_preprocessor(X_train)

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1
    )

    pipeline = Pipeline(steps=[
        ("prep", preprocessor),
        ("rf", model)
    ])

    pipeline.fit(X_train, y_train)

    return pipeline


def train_baseline(
    df: pd.DataFrame,
    target_col: str = "total_admissions"
) -> Dict[str, Any]:
    """
    Train a simple baseline model (7-day moving average).

    Args:
        df: DataFrame with the target column.
        target_col: Name of the target column.

    Returns:
        Dictionary with baseline parameters.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")

    # Calculate statistics for baseline
    overall_mean = df[target_col].mean()
    overall_std = df[target_col].std()

    # Day-of-week means
    dow_means = df.groupby("dow")[target_col].mean().to_dict()

    # Month means
    month_means = df.groupby("month")[target_col].mean().to_dict()

    baseline_params = {
        "type": "seasonal_naive",
        "overall_mean": overall_mean,
        "overall_std": overall_std,
        "dow_means": dow_means,
        "month_means": month_means,
        "window_size": 7
    }

    return baseline_params


def evaluate_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, float]:
    """
    Evaluate a trained model on test data.

    Args:
        model: Trained sklearn Pipeline.
        X_test: Test features.
        y_test: Test target.

    Returns:
        Dictionary with evaluation metrics.
    """
    predictions = model.predict(X_test)

    metrics = {
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": np.sqrt(mean_squared_error(y_test, predictions)),
        "R2": r2_score(y_test, predictions)
    }

    return metrics


def save_model(
    model: Pipeline,
    name: str = "random_forest",
    metrics: Optional[Dict[str, float]] = None
) -> str:
    """
    Save a trained model to disk.

    Args:
        model: Trained sklearn Pipeline.
        name: Name for the saved model.
        metrics: Optional metrics to save alongside.

    Returns:
        Path to saved model.
    """
    models_dir = Path(MODEL_CONFIG.model_save_path)
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / f"{name}.joblib"
    joblib.dump(model, model_path)

    # Save metrics if provided
    if metrics is not None:
        metrics_path = PROJECT_ROOT / "reports" / f"{name}_metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

    return str(model_path)


def save_baseline(
    baseline_params: Dict[str, Any],
    name: str = "baseline"
) -> str:
    """
    Save baseline model parameters to disk.

    Args:
        baseline_params: Baseline model parameters.
        name: Name for the saved model.

    Returns:
        Path to saved model.
    """
    models_dir = Path(MODEL_CONFIG.model_save_path)
    models_dir.mkdir(parents=True, exist_ok=True)

    # Convert numpy types to Python types for JSON serialization
    params_serializable = {}
    for key, value in baseline_params.items():
        if isinstance(value, dict):
            params_serializable[key] = {
                str(k): float(v) for k, v in value.items()
            }
        elif isinstance(value, (np.floating, np.integer)):
            params_serializable[key] = float(value)
        else:
            params_serializable[key] = value

    model_path = models_dir / f"{name}.json"
    with open(model_path, "w") as f:
        json.dump(params_serializable, f, indent=2)

    return str(model_path)


def train_and_save_all(
    df: pd.DataFrame,
    target_col: str = "total_admissions"
) -> Dict[str, Any]:
    """
    Train both baseline and Random Forest models and save them.

    Args:
        df: Preprocessed DataFrame with features.
        target_col: Name of the target column.

    Returns:
        Dictionary with training results and paths.
    """
    from src.data.preprocess import get_train_test_split
    from src.features.build_features import prepare_model_data

    # Split data temporally
    train_df, test_df = get_train_test_split(df)

    # Prepare features
    X_train, y_train = prepare_model_data(train_df, target_col)
    X_test, y_test = prepare_model_data(test_df, target_col)

    # Train Random Forest
    rf_model = train_model(X_train, y_train)
    rf_metrics = evaluate_model(rf_model, X_test, y_test)
    rf_path = save_model(rf_model, "random_forest", rf_metrics)

    # Train baseline
    baseline_params = train_baseline(df, target_col)
    baseline_path = save_baseline(baseline_params, "baseline")

    results = {
        "rf_model_path": rf_path,
        "rf_metrics": rf_metrics,
        "baseline_path": baseline_path,
        "train_size": len(train_df),
        "test_size": len(test_df)
    }

    return results
