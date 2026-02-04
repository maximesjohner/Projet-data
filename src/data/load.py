import pandas as pd
from pathlib import Path
from typing import Optional, Union
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import DATA_CONFIG, get_available_data_path, logger


def get_data_path() -> str:
    path = get_available_data_path()
    if path is None:
        searched = "\n  - ".join(DATA_CONFIG.data_paths)
        raise FileNotFoundError(f"Data file not found. Searched:\n  - {searched}")
    return path


def load_data(path: Optional[Union[str, Path]] = None, parse_dates: bool = True) -> pd.DataFrame:
    if path is None:
        path = get_data_path()

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    logger.info(f"Chargement des données depuis {path.name}...")

    try:
        df = pd.read_csv(path, sep=DATA_CONFIG.csv_separator, encoding=DATA_CONFIG.encoding)
    except Exception as e:
        raise ValueError(f"Failed to parse CSV: {e}")

    if parse_dates and DATA_CONFIG.date_column in df.columns:
        df[DATA_CONFIG.date_column] = pd.to_datetime(
            df[DATA_CONFIG.date_column], format=DATA_CONFIG.date_format, errors="coerce"
        )

    if DATA_CONFIG.date_column in df.columns:
        df = df.sort_values(DATA_CONFIG.date_column).reset_index(drop=True)

    logger.info(f"Données chargées: {len(df)} lignes, {len(df.columns)} colonnes")
    return df
