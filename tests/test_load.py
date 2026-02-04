
import pandas as pd
import pytest
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.load import load_data
from src.config import DATA_CONFIG

# Path to the dummy data created for tests
DUMMY_DATA_PATH = Path(__file__).parent / "data" / "dummy_hospital_data.csv"

def test_load_data_success():
    """
    Tests successful loading and parsing of a data file.
    """
    df = load_data(path=DUMMY_DATA_PATH)

    # 1. Check if it returns a pandas DataFrame
    assert isinstance(df, pd.DataFrame)

    # 2. Check for expected shape (3 rows, 2 columns)
    assert df.shape == (3, 2)

    # 3. Check if date column is parsed correctly
    assert df[DATA_CONFIG.date_column].dtype == "datetime64[ns]"
    
    # 4. Check if data is sorted by date
    assert df[DATA_CONFIG.date_column].is_monotonic_increasing

def test_load_data_no_file():
    """
    Tests that FileNotFoundError is raised for a non-existent file.
    """
    with pytest.raises(FileNotFoundError):
        load_data(path="non_existent_file.csv")

def test_load_data_no_date_parsing():
    """
    Tests loading data without parsing the date column.
    """
    df = load_data(path=DUMMY_DATA_PATH, parse_dates=False)
    # Check if date column remains as object (string)
    assert df[DATA_CONFIG.date_column].dtype == "object"
