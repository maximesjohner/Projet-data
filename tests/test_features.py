import pytest
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import load_data, preprocess_data
from src.features import build_features, get_feature_columns
from src.features.build_features import prepare_model_data


class TestFeatureEngineering:
    @pytest.fixture
    def sample_data(self):
        df = load_data()
        df = preprocess_data(df)
        return df

    def test_build_features_returns_dataframe(self, sample_data):
        result = build_features(sample_data)
        assert isinstance(result, pd.DataFrame)

    def test_build_features_adds_columns(self, sample_data):
        original_cols = len(sample_data.columns)
        result = build_features(sample_data)
        assert len(result.columns) >= original_cols

    def test_get_feature_columns_returns_list(self, sample_data):
        result = build_features(sample_data)
        cols = get_feature_columns(result)
        assert isinstance(cols, list)
        assert len(cols) > 0

    def test_prepare_model_data_returns_x_y(self, sample_data):
        result = build_features(sample_data)
        X, y = prepare_model_data(result)
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == len(y)

    def test_prepare_model_data_x_has_no_target(self, sample_data):
        result = build_features(sample_data)
        X, y = prepare_model_data(result)
        assert "total_admissions" not in X.columns
