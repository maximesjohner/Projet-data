from dataclasses import dataclass, field
from typing import Dict, List
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"

HOSPITALS = [
    {"id": "PITIE", "name": "Pitié-Salpêtrière", "file": "donnees_hopital_reference.csv"},
    {"id": "HEGP", "name": "Hôpital Européen Georges Pompidou", "file": "donnees_hopital_reference_01_HEGP.csv"},
    {"id": "STLOUIS", "name": "Hôpital Saint-Louis", "file": "donnees_hopital_reference_02_STLOUIS.csv"},
    {"id": "BICHAT", "name": "Hôpital Bichat", "file": "donnees_hopital_reference_03_BICHAT.csv"},
    {"id": "COCHIN", "name": "Hôpital Cochin", "file": "donnees_hopital_reference_04_COCHIN.csv"},
    {"id": "NECKER", "name": "Hôpital Necker", "file": "donnees_hopital_reference_05_NECKER.csv"},
    {"id": "LARIBO", "name": "Hôpital Lariboisière", "file": "donnees_hopital_reference_06_LARIBO.csv"},
    {"id": "BEAUJON", "name": "Hôpital Beaujon", "file": "donnees_hopital_reference_07_BEAUJON.csv"},
    {"id": "CHUBDX", "name": "CHU de Bordeaux", "file": "donnees_hopital_reference_08_CHUBDX.csv"},
    {"id": "CHULYON", "name": "CHU de Lyon", "file": "donnees_hopital_reference_09_CHULYON.csv"},
    {"id": "CHULILLE", "name": "CHU de Lille", "file": "donnees_hopital_reference_10_CHULILLE.csv"},
]


def get_available_hospitals() -> List[Dict]:
    """Return list of hospitals with existing reference files."""
    available = []
    for h in HOSPITALS:
        ref_path = REFERENCE_DIR / h["file"]
        if ref_path.exists():
            available.append(h)
    return available


@dataclass
class EventConfig:
    epidemic_prob: float = 0.08
    epidemic_duration_range: tuple = (7, 21)
    strike_prob: float = 0.01
    strike_duration_range: tuple = (1, 5)
    strike_staff_reduction: float = 0.15
    heatwave_prob: float = 0.04
    heatwave_duration_range: tuple = (3, 10)
    heatwave_months: tuple = (6, 7, 8)
    accident_prob: float = 0.005


@dataclass
class SeasonalityConfig:
    weekday_factors: Dict[int, float] = field(default_factory=lambda: {
        0: 1.12, 1: 1.08, 2: 1.04, 3: 1.00, 4: 0.96, 5: 0.88, 6: 0.92
    })
    month_factors: Dict[int, float] = field(default_factory=lambda: {
        1: 1.12, 2: 1.08, 3: 1.04, 4: 0.96, 5: 0.92, 6: 0.88,
        7: 0.88, 8: 0.92, 9: 1.00, 10: 1.04, 11: 1.08, 12: 1.12
    })


@dataclass
class CapacityConfig:
    total_beds: int = 1450
    beds_std: float = 50.0
    total_staff: int = 425
    staff_std: float = 12.0
    base_admissions: float = 400.0
    admissions_std: float = 55.0
    emergency_ratio: float = 0.39
    pediatric_ratio: float = 0.08
    icu_ratio: float = 0.05


@dataclass
class GeneratorConfig:
    seed: int = 42
    start_date: date = field(default_factory=lambda: date(2012, 1, 1))
    end_date: date = field(default_factory=lambda: date(2025, 12, 31))
    events: EventConfig = field(default_factory=EventConfig)
    seasonality: SeasonalityConfig = field(default_factory=SeasonalityConfig)
    capacity: CapacityConfig = field(default_factory=CapacityConfig)
    noise_level: float = 0.12
    trend_annual_growth: float = 0.008
    reference_path: str = str(PROJECT_ROOT / "data" / "reference" / "donnees_hopital_reference.csv")
    output_path: str = str(PROJECT_ROOT / "data" / "processed" / "donnees_hopital.csv")


DEFAULT_CONFIG = GeneratorConfig()
