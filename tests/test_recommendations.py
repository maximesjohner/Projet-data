import pytest
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.reco.recommend import (
    generate_recommendations, get_priority_actions, summarize_recommendations, Priority
)


class TestGenerateRecommendations:
    @pytest.fixture
    def normal_scenario(self):
        return pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=3),
            "scenario_admissions": [350, 360, 355],
            "predicted_admissions": [350, 360, 355],
            "capacity_gap": [-100, -90, -95],
            "occupancy_rate": [0.6, 0.65, 0.62],
            "effective_staff": [430, 430, 430],
            "effective_stock_pct": [75, 75, 75],
            "is_overcapacity": [False, False, False],
            "is_critical": [False, False, False]
        })

    @pytest.fixture
    def crisis_scenario(self):
        return pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=3),
            "scenario_admissions": [550, 600, 580],
            "predicted_admissions": [400, 400, 400],
            "capacity_gap": [100, 150, 130],
            "occupancy_rate": [0.95, 1.0, 0.97],
            "effective_staff": [300, 280, 290],
            "effective_stock_pct": [40, 35, 38],
            "is_overcapacity": [True, True, True],
            "is_critical": [True, True, True]
        })

    def test_returns_dataframe(self, normal_scenario):
        result = generate_recommendations(normal_scenario)
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self, normal_scenario):
        result = generate_recommendations(normal_scenario)
        required = ["date", "action", "priority", "description"]
        for col in required:
            assert col in result.columns

    def test_normal_scenario_low_priority(self, normal_scenario):
        result = generate_recommendations(normal_scenario)
        priorities = result["priority"].unique()
        assert "CRITICAL" not in priorities

    def test_crisis_generates_critical(self, crisis_scenario):
        result = generate_recommendations(crisis_scenario)
        assert "CRITICAL" in result["priority"].values


class TestPriorityActions:
    def test_filters_by_priority(self):
        df = pd.DataFrame({
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "action": ["A", "B", "C"],
            "priority": ["CRITICAL", "HIGH", "LOW"],
            "description": ["d1", "d2", "d3"]
        })
        result = get_priority_actions(df, max_priority=Priority.HIGH)
        assert len(result) == 2
        assert "LOW" not in result["priority"].values


class TestSummarize:
    def test_returns_dict(self):
        df = pd.DataFrame({
            "date": ["2024-01-01", "2024-01-02"],
            "action": ["A", "B"],
            "priority": ["CRITICAL", "HIGH"],
            "description": ["d1", "d2"]
        })
        result = summarize_recommendations(df)
        assert isinstance(result, dict)
        assert "total_recommendations" in result
        assert "critical_count" in result

    def test_counts_correctly(self):
        df = pd.DataFrame({
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "action": ["A", "B", "C"],
            "priority": ["CRITICAL", "CRITICAL", "HIGH"],
            "description": ["d1", "d2", "d3"]
        })
        result = summarize_recommendations(df)
        assert result["total_recommendations"] == 3
        assert result["critical_count"] == 2
        assert result["high_count"] == 1
