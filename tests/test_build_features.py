import pandas as pd
import numpy as np
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.build_features import get_feature_columns, prepare_model_data, build_features
from src.config import MODEL_CONFIG


@pytest.fixture
def sample_df():
    data = {
        "date": pd.to_datetime(pd.date_range(start="2023-01-01", periods=20)),
        "total_admissions": np.arange(100, 120),
        "available_staff": np.linspace(300, 280, 20),
        "staff_absence_rate": np.linspace(0.05, 0.1, 20),
        "epidemic_level": np.tile([1, 2, 3, 4, 3], 4),
        "available_beds": np.linspace(1400, 1350, 20),
        "month": pd.date_range(start="2023-01-01", periods=20).month,
        "dow": pd.date_range(start="2023-01-01", periods=20).dayofweek,
        "is_weekend": (pd.date_range(start="2023-01-01", periods=20).dayofweek >= 5).astype(int),
    }
    for col in MODEL_CONFIG.feature_columns:
        if col not in data:
            data[col] = np.random.rand(20) * 10
    return pd.DataFrame(data)


def test_get_feature_columns_from_config():
    cols = get_feature_columns()
    assert isinstance(cols, list)
    assert len(cols) > 0
    assert "epidemic_level" in cols


def test_get_feature_columns_filters_missing(sample_df):
    df_subset = sample_df[["date", "total_admissions", "epidemic_level"]]
    cols = get_feature_columns(df_subset)
    assert "epidemic_level" in cols
    assert "temperature_c" not in cols


def test_build_features_adds_columns(sample_df):
    df = build_features(sample_df)
    assert "dow" in df.columns
    assert "month" in df.columns
    assert "is_weekend" in df.columns


def test_build_features_creates_effective_staff(sample_df):
    df = build_features(sample_df)
    assert "effective_staff" in df.columns
    expected = sample_df.loc[0, "available_staff"] * (1 - sample_df.loc[0, "staff_absence_rate"])
    assert np.isclose(df.loc[0, "effective_staff"], expected)


def test_build_features_creates_stress_indicator(sample_df):
    df = build_features(sample_df)
    assert "stress_indicator" in df.columns
    expected = sample_df.loc[0, "epidemic_level"] * sample_df.loc[0, "staff_absence_rate"]
    assert np.isclose(df.loc[0, "stress_indicator"], expected)


def test_prepare_model_data(sample_df):
    df = build_features(sample_df)
    X, y = prepare_model_data(df, target_col="total_admissions")

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert "total_admissions" not in X.columns
    assert len(X) == len(y)


def test_prepare_model_data_returns_none_if_no_target(sample_df):
    df = sample_df.drop(columns=["total_admissions"])
    df = build_features(df)
    X, y = prepare_model_data(df, target_col="total_admissions")

    assert isinstance(X, pd.DataFrame)
    assert y is None
