#!/usr/bin/env python
"""
Health check script for Hospital Decision Support System.
Verifies all components are working correctly.
"""
import sys
import time
import subprocess
import requests
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

CHECKS = []


def check(name):
    def decorator(func):
        CHECKS.append((name, func))
        return func
    return decorator


@check("Data Loading")
def check_data_loading():
    from src.data import load_data
    df = load_data()
    assert len(df) > 0, "No data loaded"
    assert "total_admissions" in df.columns, "Missing target column"
    return f"{len(df)} rows loaded"


@check("Preprocessing")
def check_preprocessing():
    from src.data import load_data, preprocess_data
    df = load_data()
    df = preprocess_data(df)
    assert "dow" in df.columns, "Temporal features missing"
    return "OK"


@check("Feature Engineering")
def check_features():
    from src.data import load_data, preprocess_data
    from src.features import build_features, get_feature_columns
    df = load_data()
    df = preprocess_data(df)
    df = build_features(df)
    cols = get_feature_columns(df)
    assert len(cols) >= 10, "Too few features"
    return f"{len(cols)} features"


@check("Model Training")
def check_model_training():
    from src.data import load_data, preprocess_data
    from src.data.preprocess import get_train_test_split
    from src.features import build_features
    from src.features.build_features import prepare_model_data
    from src.models.train import train_model, evaluate_model

    df = load_data()
    df = preprocess_data(df)
    df = build_features(df)
    train_df, test_df = get_train_test_split(df)
    X_train, y_train = prepare_model_data(train_df)
    X_test, y_test = prepare_model_data(test_df)

    model = train_model(X_train.head(200), y_train.head(200), n_estimators=10)
    metrics = evaluate_model(model, X_test.head(50), y_test.head(50))

    assert metrics["R2"] > 0, "Model not learning"
    return f"R²={metrics['R2']:.2%}"


@check("Forecasting")
def check_forecasting():
    import pandas as pd
    from src.models.predict import generate_future_dates

    start = pd.Timestamp("2024-01-01")
    future = generate_future_dates(start, n_days=7)
    assert len(future) == 7, "Wrong number of forecast days"
    return "7 days generated"


@check("Scenarios")
def check_scenarios():
    import pandas as pd
    from src.scenarios.simulate import create_preset_scenarios, apply_scenario

    scenarios = create_preset_scenarios()
    assert len(scenarios) >= 5, "Missing preset scenarios"

    forecast = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=3),
        "predicted_admissions": [400, 410, 420]
    })

    result = apply_scenario(forecast, list(scenarios.values())[0])
    assert "scenario_admissions" in result.columns
    return f"{len(scenarios)} scenarios"


@check("Recommendations")
def check_recommendations():
    import pandas as pd
    from src.reco.recommend import generate_recommendations

    scenario_df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=3),
        "scenario_admissions": [400, 500, 450],
        "predicted_admissions": [400, 400, 400],
        "capacity_gap": [-50, 50, 0],
        "occupancy_rate": [0.7, 0.9, 0.8],
        "effective_staff": [430, 350, 400],
        "effective_stock_pct": [75, 45, 60],
        "is_overcapacity": [False, True, False],
        "is_critical": [False, True, False]
    })

    recs = generate_recommendations(scenario_df)
    assert len(recs) > 0, "No recommendations generated"
    return f"{len(recs)} recommendations"


@check("Streamlit App Syntax")
def check_streamlit_syntax():
    import ast

    app_files = [
        "app/Home.py",
        "app/pages/1_Dashboard.py",
        "app/pages/2_Forecast.py",
        "app/pages/3_Scenarios.py",
        "app/pages/4_Recommendations.py"
    ]

    for file in app_files:
        path = Path(__file__).parent.parent / file
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                ast.parse(f.read())

    return f"{len(app_files)} files OK"


def run_health_checks():
    print("=" * 60)
    print("HOSPITAL DSS - HEALTH CHECK")
    print("=" * 60)
    print()

    passed = 0
    failed = 0

    for name, check_func in CHECKS:
        try:
            result = check_func()
            print(f"[PASS] {name}: {result}")
            passed += 1
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
            failed += 1

    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


def check_streamlit_running(port=8501, timeout=30):
    """Check if Streamlit app is running and healthy."""
    print(f"\nChecking Streamlit app on port {port}...")

    try:
        response = requests.get(f"http://localhost:{port}", timeout=5)
        if response.status_code == 200:
            print(f"[PASS] Streamlit app is running on port {port}")
            return True
    except requests.exceptions.ConnectionError:
        print(f"[INFO] Streamlit not running on port {port}")
    except Exception as e:
        print(f"[WARN] Error checking Streamlit: {e}")

    return False


if __name__ == "__main__":
    success = run_health_checks()

    if "--with-app" in sys.argv:
        check_streamlit_running()

    sys.exit(0 if success else 1)
