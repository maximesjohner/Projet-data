
import pandas as pd
import numpy as np
import pytest
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.preprocess import (
    clean_data,
    add_temporal_features,
    fill_missing_values,
    get_train_test_split,
    preprocess_data
)

@pytest.fixture
def sample_raw_df():
    """
    Provides a sample DataFrame for testing preprocessing functions.
    The last row is an exact duplicate of the one before it.
    """
    data = {
        "date": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-03"]),
        "total_admissions": [100, 120, 110, 110],
        "temperature_c": [5.0, 6.0, 5.5, 5.5],
        "day_of_week": ["Sunday", "Monday", "Tuesday", "Tuesday"],
        "season": ["Winter", "Winter", "Winter", "Winter"],
        "heatwave_event": [0, 0, 1, 1],
    }
    # Add a row with a missing value to test filling
    df = pd.DataFrame(data)
    df.loc[1, "temperature_c"] = np.nan
    df.loc[0, "season"] = np.nan
    return df

def test_clean_data(sample_raw_df):
    """
    Tests the clean_data function.
    - Checks for duplicate removal.
    - Checks for correct type conversion.
    """
    cleaned_df = clean_data(sample_raw_df)
    
    # 1. Check if duplicates are dropped (3 unique rows from 4)
    assert len(cleaned_df) == 3
    
    # 2. Check for correct data types
    assert cleaned_df["day_of_week"].dtype == "category"
    assert cleaned_df["heatwave_event"].dtype == "Int64"

def test_add_temporal_features(sample_raw_df):
    """
    Tests the add_temporal_features function.
    """
    # Using the cleaned df for this test
    df = clean_data(sample_raw_df)
    df_with_features = add_temporal_features(df)
    
    expected_features = ["year", "month", "day", "dow", "week", "is_weekend", "day_of_year"]
    for feature in expected_features:
        assert feature in df_with_features.columns
        
    # Check a specific value: 2023-01-01 is a Sunday (dow=6), which is a weekend
    # Note: after dropping duplicates, the first row is at index 0
    first_row = df_with_features.iloc[0]
    assert first_row["dow"] == 6
    assert first_row["is_weekend"] == 1
    assert first_row["year"] == 2023

def test_fill_missing_values(sample_raw_df):
    """
    Tests the fill_missing_values function.
    - Numeric should be filled with median.
    - Categorical should be filled with mode.
    """
    df = clean_data(sample_raw_df)
    
    # Calculate expected fill values before filling
    median_temp = df["temperature_c"].median() # Median of [5.0, 5.5] is 5.25
    mode_season = df["season"].mode()[0]       # Mode of ["Winter", "Winter"] is "Winter"
    
    filled_df = fill_missing_values(df)
    
    # Check that there are no more NaNs in the tested columns
    assert not filled_df["temperature_c"].isna().any()
    assert not filled_df["season"].isna().any()
    
    # Check that NaN was filled with the correct value
    # The row at index 1 had the NaN temperature
    assert filled_df.loc[1, "temperature_c"] == median_temp
    # The row at index 0 had the NaN season
    assert filled_df.loc[0, "season"] == mode_season

def test_get_train_test_split(sample_raw_df):
    """
    Tests the temporal train-test split.
    """
    df = preprocess_data(sample_raw_df, fill_missing=True, add_time_features=True)
    
    # After preprocessing, we should have 3 unique rows
    assert len(df) == 3
    
    train_df, test_df = get_train_test_split(df, test_size=0.33)
    
    # With 3 unique rows, a 33% test set should be 1 row
    assert len(train_df) == 2
    assert len(test_df) == 1
    
    # Check that the test set contains the latest date
    assert test_df["date"].max() > train_df["date"].max()

def test_preprocess_data_pipeline(sample_raw_df):
    """
    An integration test for the main preprocess_data function.
    """
    # sample_raw_df has 4 rows (1 duplicate)
    processed_df = preprocess_data(sample_raw_df, fill_missing=True, add_time_features=True)
    
    # 1. Duplicates should be removed
    assert len(processed_df) == 3
    assert not processed_df.duplicated().any()
    
    # 2. No missing values in key columns
    assert not processed_df["temperature_c"].isna().any()
    assert not processed_df["season"].isna().any()
    
    # 3. Temporal features are added
    assert "year" in processed_df.columns
    assert "is_weekend" in processed_df.columns
