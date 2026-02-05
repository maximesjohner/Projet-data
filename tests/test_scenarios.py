import pytest
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.scenarios.simulate import (
    ScenarioParams, apply_scenario, create_preset_scenarios, summarize_scenario_impact
)


class TestScenarioParams:
    def test_default_params(self):
        params = ScenarioParams()
        assert params.epidemic_intensity == 0.0
        assert params.staffing_reduction == 0.0
        assert params.seasonal_multiplier == 1.0

    def test_from_dict(self):
        data = {"epidemic_intensity": 25, "staffing_reduction": 10}
        params = ScenarioParams.from_dict(data)
        assert params.epidemic_intensity == 25
        assert params.staffing_reduction == 10

    def test_to_dict(self):
        params = ScenarioParams(epidemic_intensity=30)
        result = params.to_dict()
        assert isinstance(result, dict)
        assert result["epidemic_intensity"] == 30


class TestPresetScenarios:
    def test_create_preset_scenarios_returns_dict(self):
        scenarios = create_preset_scenarios()
        assert isinstance(scenarios, dict)
        assert len(scenarios) > 0

    def test_preset_scenarios_have_params(self):
        scenarios = create_preset_scenarios()
        for name, params in scenarios.items():
            assert isinstance(params, ScenarioParams)


class TestApplyScenario:
    @pytest.fixture
    def sample_forecast(self):
        return pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=7),
            "predicted_admissions": [400, 410, 420, 415, 405, 380, 370]
        })

    def test_apply_scenario_returns_dataframe(self, sample_forecast):
        params = ScenarioParams()
        result = apply_scenario(sample_forecast, params)
        assert isinstance(result, pd.DataFrame)

    def test_apply_scenario_adds_columns(self, sample_forecast):
        params = ScenarioParams()
        result = apply_scenario(sample_forecast, params)
        expected_cols = ["scenario_admissions", "capacity_gap", "occupancy_rate"]
        for col in expected_cols:
            assert col in result.columns

    def test_epidemic_increases_admissions(self, sample_forecast):
        params = ScenarioParams(epidemic_intensity=50)
        result = apply_scenario(sample_forecast, params)
        assert result["scenario_admissions"].mean() > sample_forecast["predicted_admissions"].mean()

    def test_baseline_scenario_unchanged(self, sample_forecast):
        params = ScenarioParams()
        result = apply_scenario(sample_forecast, params)
        pd.testing.assert_series_equal(
            result["scenario_admissions"],
            sample_forecast["predicted_admissions"].astype(float),
            check_names=False
        )


class TestSummarizeImpact:
    def test_summarize_returns_dict(self):
        df = pd.DataFrame({
            "scenario_admissions": [400, 500, 600],
            "is_overcapacity": [False, True, True],
            "is_critical": [False, False, True],
            "capacity_gap": [-50, 50, 150],
            "occupancy_rate": [0.7, 0.9, 1.1],
            "demand_change_pct": [0, 25, 50]
        })
        result = summarize_scenario_impact(df)
        assert isinstance(result, dict)
        assert "avg_daily_admissions" in result
        assert "days_overcapacity" in result
