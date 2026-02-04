
import pandas as pd
import numpy as np
import pytest
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.features.build_features import (
    create_interaction_features,
    create_lag_features,
    create_rolling_features,
    get_feature_columns,
    prepare_model_data,
    build_features,
)
from src.config import MODEL_CONFIG

@pytest.fixture
def sample_preprocessed_df():
    """
    Provides a sample preprocessed DataFrame for feature engineering tests.
    It simulates a time series.
    """
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
    # Add other feature columns required by config with default values
    for col in MODEL_CONFIG.feature_columns:
        if col not in data:
            data[col] = np.random.rand(20) * 10

    return pd.DataFrame(data)


def test_create_interaction_features(sample_preprocessed_df):
    """
    Tests the creation of interaction features.
    """
    df = create_interaction_features(sample_preprocessed_df)
    
    assert "effective_staff" in df.columns
    assert "stress_indicator" in df.columns
    
    # Check a value
    expected_staff = sample_preprocessed_df.loc[0, "available_staff"] * (1 - sample_preprocessed_df.loc[0, "staff_absence_rate"])
    assert np.isclose(df.loc[0, "effective_staff"], expected_staff)

def test_create_lag_features(sample_preprocessed_df):
    """
    Tests the creation of lag features.
    """
    lags = [1, 7]
    df = create_lag_features(sample_preprocessed_df, lags=lags)
    
    # Check that columns are created
    for lag in lags:
        assert f"total_admissions_lag_{lag}" in df.columns
    
    # Check value: lag 1 of row 1 should be value of row 0
    assert df.loc[1, "total_admissions_lag_1"] == sample_preprocessed_df.loc[0, "total_admissions"]
    
    # Check for NaNs at the beginning
    assert pd.isna(df.loc[0, "total_admissions_lag_1"])
    assert pd.isna(df.loc[6, "total_admissions_lag_7"])
    assert not pd.isna(df.loc[7, "total_admissions_lag_7"])

def test_create_rolling_features(sample_preprocessed_df):
    """
    Tests the creation of rolling statistical features.
    """
    windows = [7, 14]
    df = create_rolling_features(sample_preprocessed_df, windows=windows)
    
    # Check that columns are created
    for window in windows:
        assert f"total_admissions_rolling_mean_{window}" in df.columns
        assert f"total_admissions_rolling_std_{window}" in df.columns
        
    # Check for NaNs at the beginning
    assert pd.isna(df.loc[5, "total_admissions_rolling_mean_7"])
    assert not pd.isna(df.loc[6, "total_admissions_rolling_mean_7"])
    
    # Check calculation
    expected_mean = sample_preprocessed_df["total_admissions"].iloc[:7].mean()
    assert np.isclose(df.loc[6, "total_admissions_rolling_mean_7"], expected_mean)
    
def test_prepare_model_data(sample_preprocessed_df):
    """
    Tests the final preparation of data for the model.
    """
    df = build_features(sample_preprocessed_df)
    
    X, y = prepare_model_data(df, target_col="total_admissions")
    
    # 1. Check types
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    
    # 2. Check that target is not in features
    assert "total_admissions" not in X.columns
    
    # 3. Check that only valid feature columns are present
    expected_cols = get_feature_columns(df)
    assert X.columns.tolist().sort() == expected_cols.sort()
