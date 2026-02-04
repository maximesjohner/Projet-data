"""
Scenario simulation for the Hospital Decision Support System.
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import SCENARIO_DEFAULTS, CAPACITY_CONFIG


@dataclass
class ScenarioParams:
    """Parameters for scenario simulation."""
    epidemic_intensity: float = 0.0  # % increase in demand (0-100)
    staffing_reduction: float = 0.0  # % reduction in staff capacity (0-100)
    seasonal_multiplier: float = 1.0  # Multiplier for seasonal effects
    shock_day_spike: float = 0.0  # % spike for a specific day (0-200)
    shock_day_index: Optional[int] = None  # Index of shock day (0 = first day)
    beds_reduction: float = 0.0  # % reduction in available beds (0-100)
    stock_reduction: float = 0.0  # % reduction in medical stock (0-100)

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> "ScenarioParams":
        """Create ScenarioParams from a dictionary."""
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
        """Convert to dictionary."""
        return {
            "epidemic_intensity": self.epidemic_intensity,
            "staffing_reduction": self.staffing_reduction,
            "seasonal_multiplier": self.seasonal_multiplier,
            "shock_day_spike": self.shock_day_spike,
            "shock_day_index": self.shock_day_index,
            "beds_reduction": self.beds_reduction,
            "stock_reduction": self.stock_reduction
        }


def apply_scenario(
    forecast_df: pd.DataFrame,
    scenario: ScenarioParams,
    prediction_col: str = "predicted_admissions"
) -> pd.DataFrame:
    """
    Apply scenario modifications to a baseline forecast.

    This function takes a baseline forecast and applies deterministic
    transformations based on scenario parameters.

    Args:
        forecast_df: DataFrame with baseline predictions.
        scenario: ScenarioParams defining the scenario.
        prediction_col: Name of the prediction column.

    Returns:
        DataFrame with scenario-adjusted predictions and capacity metrics.
    """
    df = forecast_df.copy()

    # Start with baseline predictions
    baseline = df[prediction_col].copy()

    # 1. Apply epidemic intensity (% increase in demand)
    epidemic_factor = 1 + (scenario.epidemic_intensity / 100)

    # 2. Apply seasonal multiplier
    seasonal_factor = scenario.seasonal_multiplier

    # 3. Combined demand adjustment
    df["scenario_admissions"] = baseline * epidemic_factor * seasonal_factor

    # 4. Apply shock day spike if specified
    if scenario.shock_day_index is not None and scenario.shock_day_spike > 0:
        if 0 <= scenario.shock_day_index < len(df):
            shock_factor = 1 + (scenario.shock_day_spike / 100)
            df.loc[df.index[scenario.shock_day_index], "scenario_admissions"] *= shock_factor

    # 5. Calculate capacity adjustments
    base_staff = CAPACITY_CONFIG.total_staff
    base_beds = CAPACITY_CONFIG.total_beds
    base_capacity = CAPACITY_CONFIG.normal_admission_capacity

    # Effective staff after reduction
    df["effective_staff"] = base_staff * (1 - scenario.staffing_reduction / 100)

    # Effective beds after reduction
    df["effective_beds"] = base_beds * (1 - scenario.beds_reduction / 100)

    # Effective capacity (limited by both staff and beds)
    # Staff can handle ~3 patients each, beds limit total patients
    staff_capacity = df["effective_staff"] * 3
    bed_capacity = df["effective_beds"] * CAPACITY_CONFIG.critical_occupancy_threshold

    df["effective_capacity"] = np.minimum(
        base_capacity * (1 - scenario.staffing_reduction / 100),
        np.minimum(staff_capacity, bed_capacity)
    )

    # 6. Calculate gap (demand - capacity)
    df["capacity_gap"] = df["scenario_admissions"] - df["effective_capacity"]

    # 7. Stock level adjustments
    base_stock = 75.0  # Default stock level %
    df["effective_stock_pct"] = base_stock * (1 - scenario.stock_reduction / 100)

    # 8. Calculate stress indicators
    df["occupancy_rate"] = df["scenario_admissions"] / df["effective_beds"]
    df["is_overcapacity"] = df["capacity_gap"] > 0
    df["is_critical"] = df["occupancy_rate"] > CAPACITY_CONFIG.critical_occupancy_threshold

    # Keep the baseline for comparison
    df["baseline_admissions"] = baseline
    df["demand_change_pct"] = (
        (df["scenario_admissions"] - baseline) / baseline * 100
    )

    return df


def compare_scenarios(
    forecast_df: pd.DataFrame,
    scenarios: Dict[str, ScenarioParams],
    prediction_col: str = "predicted_admissions"
) -> pd.DataFrame:
    """
    Compare multiple scenarios against baseline.

    Args:
        forecast_df: DataFrame with baseline predictions.
        scenarios: Dictionary mapping scenario names to ScenarioParams.
        prediction_col: Name of the prediction column.

    Returns:
        DataFrame with all scenario results side by side.
    """
    result = forecast_df[["date", prediction_col]].copy()
    result = result.rename(columns={prediction_col: "baseline"})

    for name, scenario in scenarios.items():
        scenario_df = apply_scenario(forecast_df, scenario, prediction_col)
        result[f"{name}_admissions"] = scenario_df["scenario_admissions"]
        result[f"{name}_gap"] = scenario_df["capacity_gap"]
        result[f"{name}_occupancy"] = scenario_df["occupancy_rate"]

    return result


def create_preset_scenarios() -> Dict[str, ScenarioParams]:
    """
    Create preset scenario configurations.

    Returns:
        Dictionary of preset scenarios.
    """
    return {
        "Baseline": ScenarioParams(),
        "Mild Epidemic": ScenarioParams(
            epidemic_intensity=15,
            staffing_reduction=5
        ),
        "Severe Epidemic": ScenarioParams(
            epidemic_intensity=40,
            staffing_reduction=15
        ),
        "Staff Strike": ScenarioParams(
            staffing_reduction=30,
            stock_reduction=10
        ),
        "Winter Peak": ScenarioParams(
            epidemic_intensity=25,
            seasonal_multiplier=1.2
        ),
        "Summer Heatwave": ScenarioParams(
            epidemic_intensity=10,
            seasonal_multiplier=0.9,
            staffing_reduction=10
        ),
        "Major Accident": ScenarioParams(
            shock_day_spike=80,
            shock_day_index=0
        )
    }


def summarize_scenario_impact(
    scenario_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Summarize the impact of a scenario.

    Args:
        scenario_df: DataFrame with scenario results.

    Returns:
        Dictionary with summary statistics.
    """
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
