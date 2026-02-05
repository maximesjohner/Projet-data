import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional
import os

PROJECT_ROOT = Path(__file__).parent.parent.absolute()

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("hospital_dss")


@dataclass
class DataConfig:
    data_paths: List[str] = field(default_factory=lambda: [
        str(PROJECT_ROOT / "data" / "processed" / "donnees_hopital.csv"),
    ])
    csv_separator: str = ";"
    encoding: str = "utf-8"
    date_column: str = "date"
    date_format: str = "%d/%m/%Y"


@dataclass
class ModelConfig:
    target_column: str = "total_admissions"
    test_size: float = 0.2
    random_state: int = 42
    n_estimators: int = 300
    model_save_path: str = str(PROJECT_ROOT / "models")
    feature_columns: List[str] = field(default_factory=lambda: [
        "hospital_id", "epidemic_level", "temperature_c", "heatwave_event", "strike_level",
        "staff_absence_rate", "available_staff", "available_beds",
        "scheduled_surgeries", "medical_stock_level_pct", "accident_event",
        "external_alert_level", "month", "dow", "is_weekend", "season", "day_of_week",
    ])


@dataclass
class CapacityConfig:
    total_beds: int = 1450
    total_staff: int = 425
    normal_admission_capacity: int = 450
    critical_occupancy_threshold: float = 0.85
    warning_occupancy_threshold: float = 0.75
    min_stock_level: float = 50.0


@dataclass
class ScenarioDefaults:
    epidemic_intensity: float = 0.0
    staffing_reduction: float = 0.0
    seasonal_multiplier: float = 1.0
    shock_day_spike: float = 0.0
    shock_day_index: Optional[int] = None


DATA_CONFIG = DataConfig()
MODEL_CONFIG = ModelConfig()
CAPACITY_CONFIG = CapacityConfig()
SCENARIO_DEFAULTS = ScenarioDefaults()


def get_available_data_path() -> Optional[str]:
    for path in DATA_CONFIG.data_paths:
        if os.path.exists(path):
            return path
    return None


def check_data_exists() -> bool:
    return get_available_data_path() is not None
