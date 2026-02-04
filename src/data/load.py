"""
Data loading utilities for the Hospital Decision Support System.
"""
import pandas as pd
from pathlib import Path
from typing import Optional, Union
import os
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import DATA_CONFIG, get_available_data_path


def get_data_path() -> str:
    """
    Get the path to the hospital data file.

    Searches for the data file in multiple locations in order of preference.

    Returns:
        str: Path to the data file.

    Raises:
        FileNotFoundError: If no data file is found in any location.
    """
    path = get_available_data_path()
    if path is None:
        searched = "\n  - ".join(DATA_CONFIG.data_paths)
        raise FileNotFoundError(
            f"Data file not found. Searched locations:\n  - {searched}"
        )
    return path


def load_data(
    path: Optional[Union[str, Path]] = None,
    parse_dates: bool = True
) -> pd.DataFrame:
    """
    Load the hospital data from CSV file.

    Args:
        path: Optional path to the CSV file. If None, searches default locations.
        parse_dates: If True, parse the date column to datetime.

    Returns:
        pd.DataFrame: Loaded hospital data.

    Raises:
        FileNotFoundError: If the data file is not found.
        ValueError: If the data file cannot be parsed.
    """
    if path is None:
        path = get_data_path()

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    try:
        df = pd.read_csv(
            path,
            sep=DATA_CONFIG.csv_separator,
            encoding=DATA_CONFIG.encoding
        )
    except Exception as e:
        raise ValueError(f"Failed to parse CSV file: {e}")

    if parse_dates and DATA_CONFIG.date_column in df.columns:
        df[DATA_CONFIG.date_column] = pd.to_datetime(
            df[DATA_CONFIG.date_column],
            format=DATA_CONFIG.date_format,
            errors="coerce"
        )

    # Sort by date
    if DATA_CONFIG.date_column in df.columns:
        df = df.sort_values(DATA_CONFIG.date_column).reset_index(drop=True)

    return df


def load_sample_data(n_rows: int = 100) -> pd.DataFrame:
    """
    Load a sample of the hospital data for testing.

    Args:
        n_rows: Number of rows to load.

    Returns:
        pd.DataFrame: Sample of hospital data.
    """
    df = load_data()
    return df.head(n_rows)
