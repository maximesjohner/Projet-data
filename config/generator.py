from dataclasses import dataclass, field
from typing import Dict
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


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
