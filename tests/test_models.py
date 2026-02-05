import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import load_data, preprocess_data
from src.data.preprocess import get_train_test_split
from src.features import build_features
from src.features.build_features import prepare_model_data
from src.models.train import train_model, train_baseline, evaluate_model
from src.models.predict import predict, forecast, generate_future_dates


class TestModelTraining:
    @pytest.fixture
    def training_data(self):
        df = load_data()
        df = preprocess_data(df)
        df = build_features(df)
        train_df, test_df = get_train_test_split(df)
        X_train, y_train = prepare_model_data(train_df)
        X_test, y_test = prepare_model_data(test_df)
        return X_train, y_train, X_test, y_test

    def test_train_model_returns_pipeline(self, training_data):
        X_train, y_train, _, _ = training_data
        model = train_model(X_train.head(100), y_train.head(100), n_estimators=10)
        assert hasattr(model, "predict")

    def test_model_can_predict(self, training_data):
        X_train, y_train, X_test, _ = training_data
        model = train_model(X_train.head(100), y_train.head(100), n_estimators=10)
        predictions = model.predict(X_test.head(10))
        assert len(predictions) == 10

    def test_predictions_are_positive(self, training_data):
        X_train, y_train, X_test, _ = training_data
        model = train_model(X_train.head(100), y_train.head(100), n_estimators=10)
        predictions = model.predict(X_test.head(10))
        assert all(p > 0 for p in predictions)


class TestModelEvaluation:
    @pytest.fixture
    def trained_model(self):
        df = load_data()
        df = preprocess_data(df)
        df = build_features(df)
        train_df, test_df = get_train_test_split(df)
        X_train, y_train = prepare_model_data(train_df)
        X_test, y_test = prepare_model_data(test_df)
        model = train_model(X_train.head(200), y_train.head(200), n_estimators=10)
        return model, X_test.head(50), y_test.head(50)

    def test_evaluate_returns_metrics(self, trained_model):
        model, X_test, y_test = trained_model
        metrics = evaluate_model(model, X_test, y_test)
        assert "MAE" in metrics
        assert "RMSE" in metrics
        assert "R2" in metrics

    def test_metrics_are_reasonable(self, trained_model):
        model, X_test, y_test = trained_model
        metrics = evaluate_model(model, X_test, y_test)
        assert metrics["MAE"] >= 0
        assert metrics["RMSE"] >= 0
        assert -1 <= metrics["R2"] <= 1


class TestBaseline:
    def test_train_baseline_returns_dict(self):
        df = load_data()
        df = preprocess_data(df)
        df = build_features(df)
        params = train_baseline(df)
        assert isinstance(params, dict)
        assert "dow_means" in params
        assert "month_means" in params


class TestForecast:
    def test_generate_future_dates(self):
        start = pd.Timestamp("2024-01-01")
        result = generate_future_dates(start, n_days=7)
        assert len(result) == 7
        assert "date" in result.columns
        assert "dow" in result.columns

    def test_forecast_returns_predictions(self):
        df = load_data()
        df = preprocess_data(df)
        df = build_features(df)
        train_df, _ = get_train_test_split(df)
        X_train, y_train = prepare_model_data(train_df)
        model = train_model(X_train.head(200), y_train.head(200), n_estimators=10)

        start = pd.Timestamp("2024-01-01")
        result = forecast(model, start, n_days=7)
        assert "predicted_admissions" in result.columns
        assert len(result) == 7
