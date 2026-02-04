import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import SCENARIO_DEFAULTS, CAPACITY_CONFIG, logger


@dataclass
class ScenarioParams:
    epidemic_intensity: float = 0.0
    staffing_reduction: float = 0.0
    seasonal_multiplier: float = 1.0
    shock_day_spike: float = 0.0
    shock_day_index: Optional[int] = None
    beds_reduction: float = 0.0
    stock_reduction: float = 0.0

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> "ScenarioParams":
        return cls(
            epidemic_intensity=params.get("epidemic_intensity", 0.0),
            staffing_reduction=params.get("staffing_reduction", 0.0),
            seasonal_multiplier=params.get("seasonal_multiplier", 1.0),
            shock_day_spike=params.get("shock_day_spike", 0.0),
            shock_day_index=params.get("shock_day_index"),
            beds_reduction=params.get("beds_reduction", 0.0),
            stock_reduction=params.get("stock_reduction", 0.0)
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "epidemic_intensity": self.epidemic_intensity,
            "staffing_reduction": self.staffing_reduction,
            "seasonal_multiplier": self.seasonal_multiplier,
            "shock_day_spike": self.shock_day_spike,
            "shock_day_index": self.shock_day_index,
            "beds_reduction": self.beds_reduction,
            "stock_reduction": self.stock_reduction
        }


def apply_scenario(forecast_df: pd.DataFrame, scenario: ScenarioParams, prediction_col: str = "predicted_admissions") -> pd.DataFrame:
    logger.info(f"Application du scénario: épidémie={scenario.epidemic_intensity}%, personnel={scenario.staffing_reduction}%")

    df = forecast_df.copy()
    baseline = df[prediction_col].copy()

    epidemic_factor = 1 + (scenario.epidemic_intensity / 100)
    seasonal_factor = scenario.seasonal_multiplier
    df["scenario_admissions"] = baseline * epidemic_factor * seasonal_factor

    if scenario.shock_day_index is not None and scenario.shock_day_spike > 0:
        if 0 <= scenario.shock_day_index < len(df):
            shock_factor = 1 + (scenario.shock_day_spike / 100)
            df.loc[df.index[scenario.shock_day_index], "scenario_admissions"] *= shock_factor

    base_staff = CAPACITY_CONFIG.total_staff
    base_beds = CAPACITY_CONFIG.total_beds
    base_capacity = CAPACITY_CONFIG.normal_admission_capacity

    df["effective_staff"] = base_staff * (1 - scenario.staffing_reduction / 100)
    df["effective_beds"] = base_beds * (1 - scenario.beds_reduction / 100)

    staff_capacity = df["effective_staff"] * 3
    bed_capacity = df["effective_beds"] * CAPACITY_CONFIG.critical_occupancy_threshold

    df["effective_capacity"] = np.minimum(
        base_capacity * (1 - scenario.staffing_reduction / 100),
        np.minimum(staff_capacity, bed_capacity)
    )

    df["capacity_gap"] = df["scenario_admissions"] - df["effective_capacity"]
    df["effective_stock_pct"] = 75.0 * (1 - scenario.stock_reduction / 100)
    df["occupancy_rate"] = df["scenario_admissions"] / df["effective_beds"]
    df["is_overcapacity"] = df["capacity_gap"] > 0
    df["is_critical"] = df["occupancy_rate"] > CAPACITY_CONFIG.critical_occupancy_threshold
    df["baseline_admissions"] = baseline
    df["demand_change_pct"] = (df["scenario_admissions"] - baseline) / baseline * 100

    logger.info(f"Scénario appliqué: admissions moyennes={df['scenario_admissions'].mean():.0f}/jour")
    return df


def compare_scenarios(forecast_df: pd.DataFrame, scenarios: Dict[str, ScenarioParams], prediction_col: str = "predicted_admissions") -> pd.DataFrame:
    result = forecast_df[["date", prediction_col]].copy()
    result = result.rename(columns={prediction_col: "baseline"})

    for name, scenario in scenarios.items():
        scenario_df = apply_scenario(forecast_df, scenario, prediction_col)
        result[f"{name}_admissions"] = scenario_df["scenario_admissions"]
        result[f"{name}_gap"] = scenario_df["capacity_gap"]
        result[f"{name}_occupancy"] = scenario_df["occupancy_rate"]

    return result


def create_preset_scenarios() -> Dict[str, ScenarioParams]:
    return {
        "Référence": ScenarioParams(),
        "Épidémie Légère": ScenarioParams(epidemic_intensity=15, staffing_reduction=5),
        "Épidémie Sévère": ScenarioParams(epidemic_intensity=40, staffing_reduction=15),
        "Grève du Personnel": ScenarioParams(staffing_reduction=30, stock_reduction=10),
        "Pic Hivernal": ScenarioParams(epidemic_intensity=25, seasonal_multiplier=1.2),
        "Canicule Estivale": ScenarioParams(epidemic_intensity=10, seasonal_multiplier=0.9, staffing_reduction=10),
        "Accident Majeur": ScenarioParams(shock_day_spike=80, shock_day_index=0)
    }


def summarize_scenario_impact(scenario_df: pd.DataFrame) -> Dict[str, Any]:
    return {
        "avg_daily_admissions": scenario_df["scenario_admissions"].mean(),
        "max_daily_admissions": scenario_df["scenario_admissions"].max(),
        "total_admissions": scenario_df["scenario_admissions"].sum(),
        "days_overcapacity": scenario_df["is_overcapacity"].sum(),
        "days_critical": scenario_df["is_critical"].sum(),
        "avg_capacity_gap": scenario_df["capacity_gap"].mean(),
        "max_capacity_gap": scenario_df["capacity_gap"].max(),
        "avg_occupancy_rate": scenario_df["occupancy_rate"].mean(),
        "max_occupancy_rate": scenario_df["occupancy_rate"].max(),
        "avg_demand_change_pct": scenario_df["demand_change_pct"].mean()
    }
