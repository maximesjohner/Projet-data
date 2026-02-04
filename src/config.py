"""
Configuration settings for the Hospital Decision Support System.
"""
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional
import os


# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.absolute()


@dataclass
class DataConfig:
    """Configuration for data paths and loading."""
    # Primary data paths (in order of preference)
    data_paths: List[str] = field(default_factory=lambda: [
        str(PROJECT_ROOT / "data" / "raw" / "donnees_hopital.csv"),
        str(PROJECT_ROOT / "data" / "donnees_hopital.csv"),
        str(PROJECT_ROOT / "donnees_hopital.csv"),
        "/mnt/data/donnees_hopital.csv",
    ])
    csv_separator: str = ";"
    encoding: str = "utf-8"
    date_column: str = "date"
    date_format: str = "%d/%m/%Y"


@dataclass
class ModelConfig:
    """Configuration for model training."""
    target_column: str = "total_admissions"
    test_size: float = 0.2
    random_state: int = 42
    n_estimators: int = 300
    model_save_path: str = str(PROJECT_ROOT / "models")

    # Realistic features (no data leakage)
    feature_columns: List[str] = field(default_factory=lambda: [
        "epidemic_level",
        "temperature_c",
        "heatwave_event",
        "strike_level",
        "staff_absence_rate",
        "available_staff",
        "available_beds",
        "scheduled_surgeries",
        "medical_stock_level_pct",
        "accident_event",
        "external_alert_level",
        "month",
        "dow",
        "is_weekend",
        "season",
        "day_of_week",
    ])


@dataclass
class CapacityConfig:
    """Configuration for hospital capacity assumptions."""
    # Baseline capacity values (typical hospital)
    total_beds: int = 1500
    total_staff: int = 430
    normal_admission_capacity: int = 450  # Max admissions per day
    critical_occupancy_threshold: float = 0.85
    warning_occupancy_threshold: float = 0.75
    min_stock_level: float = 50.0  # Minimum acceptable stock %
    staff_patient_ratio: float = 0.12  # Staff per patient


@dataclass
class ScenarioDefaults:
    """Default values for scenario parameters."""
    epidemic_intensity: float = 0.0  # % increase in demand
    staffing_reduction: float = 0.0  # % decrease in staff capacity
    seasonal_multiplier: float = 1.0  # Multiplier for seasonal effects
    shock_day_spike: float = 0.0  # % spike for specific day
    shock_day_index: Optional[int] = None  # Day index for shock


# Global configuration instances
DATA_CONFIG = DataConfig()
MODEL_CONFIG = ModelConfig()
CAPACITY_CONFIG = CapacityConfig()
SCENARIO_DEFAULTS = ScenarioDefaults()


def get_available_data_path() -> Optional[str]:
    """Find the first available data file path."""
    for path in DATA_CONFIG.data_paths:
        if os.path.exists(path):
            return path
    return None
