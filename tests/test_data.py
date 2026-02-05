import pytest
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import load_data, preprocess_data
from src.data.preprocess import get_train_test_split


class TestDataLoading:
    def test_load_data_returns_dataframe(self):
        df = load_data()
        assert isinstance(df, pd.DataFrame)

    def test_load_data_has_required_columns(self):
        df = load_data()
        required = ["date", "total_admissions", "emergency_admissions", "available_beds"]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_load_data_not_empty(self):
        df = load_data()
        assert len(df) > 0

    def test_load_data_date_is_datetime(self):
        df = load_data()
        assert pd.api.types.is_datetime64_any_dtype(df["date"])


class TestPreprocessing:
    def test_preprocess_returns_dataframe(self):
        df = load_data()
        result = preprocess_data(df)
        assert isinstance(result, pd.DataFrame)

    def test_preprocess_adds_temporal_features(self):
        df = load_data()
        result = preprocess_data(df)
        temporal_cols = ["year", "month", "dow", "is_weekend"]
        for col in temporal_cols:
            assert col in result.columns, f"Missing temporal column: {col}"

    def test_preprocess_no_missing_target(self):
        df = load_data()
        result = preprocess_data(df)
        assert result["total_admissions"].isna().sum() == 0


class TestTrainTestSplit:
    def test_split_returns_two_dataframes(self):
        df = load_data()
        df = preprocess_data(df)
        train, test = get_train_test_split(df)
        assert isinstance(train, pd.DataFrame)
        assert isinstance(test, pd.DataFrame)

    def test_split_preserves_data(self):
        df = load_data()
        df = preprocess_data(df)
        train, test = get_train_test_split(df)
        assert len(train) + len(test) == len(df)

    def test_split_ratio(self):
        df = load_data()
        df = preprocess_data(df)
        train, test = get_train_test_split(df, test_size=0.2)
        ratio = len(test) / len(df)
        assert 0.18 < ratio < 0.22
