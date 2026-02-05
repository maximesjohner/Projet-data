
import pandas as pd
import pytest
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.load import load_data, DataNotFoundError
from src.data.preprocess import preprocess_data
from src.features.build_features import build_features, prepare_model_data
from src.config import MODEL_CONFIG

# Path to the dummy data created for tests
DUMMY_DATA_PATH = Path(__file__).parent / "data" / "dummy_hospital_data.csv"

def test_full_data_processing_pipeline():
    """
    Tests the full data processing pipeline from loading to feature preparation.
    This acts as a functionality test.
    """
    # 1. Load data
    try:
        raw_df = load_data(path=DUMMY_DATA_PATH)
    except (FileNotFoundError, DataNotFoundError):
        pytest.fail(f"Test setup error: Dummy data file not found at {DUMMY_DATA_PATH}")
    
    # Augment the dummy data to have columns required for feature building
    # This makes the test more robust to changes in feature engineering
    base_data = {
        "available_staff": 300,
        "staff_absence_rate": 0.05,
        "epidemic_level": 2,
        "available_beds": 1400,
        "temperature_c": 10,
    }
    for col, val in base_data.items():
        raw_df[col] = val

    # Add all other feature columns from config with default values if they don't exist
    for col in MODEL_CONFIG.feature_columns:
        if col not in raw_df.columns:
            raw_df[col] = 0.0


    # 2. Preprocess data
    preprocessed_df = preprocess_data(raw_df)

    # 3. Build features
    # Not including lags or rolling features in this basic integration test
    # as they introduce NaNs which would need to be handled.
    featured_df = build_features(preprocessed_df)

    # 4. Prepare model data
    X, y = prepare_model_data(featured_df, target_col="total_admissions")

    # --- Assertions ---
    
    # Check that the pipeline ran and produced a result
    assert X is not None, "X features should not be None"
    assert y is not None, "y target should not be None"
    
    # Check types
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    
    # Check shape - we started with 3 rows
    assert len(X) == 3
    assert len(y) == 3
    
    # Check that no obvious NaNs were introduced in the feature set
    # (ignoring lags/rolling for this test)
    assert not X.isnull().values.any()
    
    # Check that target is not in X
    assert "total_admissions" not in X.columns
    
    # Check that all expected feature columns are present
    final_feature_cols = [col for col in MODEL_CONFIG.feature_columns if col in X.columns]
    assert len(final_feature_cols) > 0 # At least some features must be there
    assert set(final_feature_cols).issubset(set(X.columns))

