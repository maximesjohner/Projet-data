import pandas as pd
from pathlib import Path
from typing import Optional, Union, List
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import DATA_CONFIG, get_available_data_path, logger, PROJECT_ROOT


class DataNotFoundError(Exception):
    pass


def get_data_path() -> str:
    path = get_available_data_path()
    if path is None:
        raise DataNotFoundError("No data file found. Run: python run.py generate")
    return path


def get_training_data_dir() -> Path:
    """Get the path to the training data directory."""
    return PROJECT_ROOT / "data" / "processed" / "training"


def load_data(path: Optional[Union[str, Path]] = None, parse_dates: bool = True) -> pd.DataFrame:
    """Load data from a single file (default: donnees_hopital.csv for frontend)."""
    if path is None:
        path = get_data_path()
    path = Path(path)

    if not path.exists():
        raise DataNotFoundError(f"Data file not found: {path}")

    logger.info(f"Chargement des données depuis {path.name}...")
    df = pd.read_csv(path, sep=DATA_CONFIG.csv_separator, encoding=DATA_CONFIG.encoding)

    if parse_dates and DATA_CONFIG.date_column in df.columns:
        df[DATA_CONFIG.date_column] = pd.to_datetime(
            df[DATA_CONFIG.date_column], format=DATA_CONFIG.date_format, errors="coerce"
        )

    if DATA_CONFIG.date_column in df.columns:
        df = df.sort_values(DATA_CONFIG.date_column).reset_index(drop=True)

    logger.info(f"Données chargées: {len(df)} lignes, {len(df.columns)} colonnes")
    return df


def load_training_data(parse_dates: bool = True) -> pd.DataFrame:
    """Load all hospital data from training directory for model training."""
    training_dir = get_training_data_dir()

    if not training_dir.exists():
        logger.warning(f"Training directory not found: {training_dir}")
        logger.info("Falling back to single file load...")
        return load_data(parse_dates=parse_dates)

    csv_files = list(training_dir.glob("donnees_*.csv"))

    if not csv_files:
        logger.warning(f"No training files found in {training_dir}")
        logger.info("Falling back to single file load...")
        return load_data(parse_dates=parse_dates)

    logger.info(f"Chargement des données d'entraînement depuis {len(csv_files)} fichiers...")

    all_dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file, sep=DATA_CONFIG.csv_separator, encoding=DATA_CONFIG.encoding)
        all_dfs.append(df)
        logger.info(f"  - {csv_file.name}: {len(df)} lignes")

    combined_df = pd.concat(all_dfs, ignore_index=True)

    if parse_dates and DATA_CONFIG.date_column in combined_df.columns:
        combined_df[DATA_CONFIG.date_column] = pd.to_datetime(
            combined_df[DATA_CONFIG.date_column], format=DATA_CONFIG.date_format, errors="coerce"
        )

    if DATA_CONFIG.date_column in combined_df.columns:
        combined_df = combined_df.sort_values(DATA_CONFIG.date_column).reset_index(drop=True)

    hospitals = combined_df["hospital_id"].nunique() if "hospital_id" in combined_df.columns else 1
    logger.info(f"Données d'entraînement chargées: {len(combined_df)} lignes, {hospitals} hôpitaux")

    return combined_df
