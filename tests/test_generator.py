"""Tests for synthetic data generator."""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from datetime import date

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.generator import GeneratorConfig
from src.generator.core import HospitalDataGenerator, generate_dataset
from src.generator.validators import validate_dataset


class TestReproducibility:
    """Test that generation is reproducible with same seed."""

    def test_same_seed_same_output(self):
        config1 = GeneratorConfig(seed=42, start_date=date(2024, 1, 1), end_date=date(2024, 1, 31))
        config2 = GeneratorConfig(seed=42, start_date=date(2024, 1, 1), end_date=date(2024, 1, 31))

        gen1 = HospitalDataGenerator(config1)
        gen2 = HospitalDataGenerator(config2)

        df1 = gen1.generate()
        df2 = gen2.generate()

        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seed_different_output(self):
        config1 = GeneratorConfig(seed=42, start_date=date(2024, 1, 1), end_date=date(2024, 1, 31))
        config2 = GeneratorConfig(seed=123, start_date=date(2024, 1, 1), end_date=date(2024, 1, 31))

        gen1 = HospitalDataGenerator(config1)
        gen2 = HospitalDataGenerator(config2)

        df1 = gen1.generate()
        df2 = gen2.generate()

        assert not df1["total_admissions"].equals(df2["total_admissions"])


class TestNoNegativeValues:
    """Test that count columns have no negative values."""

    @pytest.fixture
    def generated_df(self):
        config = GeneratorConfig(seed=42, start_date=date(2024, 1, 1), end_date=date(2024, 12, 31))
        gen = HospitalDataGenerator(config)
        return gen.generate()

    def test_total_admissions_non_negative(self, generated_df):
        assert (generated_df["total_admissions"] >= 0).all()

    def test_emergency_admissions_non_negative(self, generated_df):
        assert (generated_df["emergency_admissions"] >= 0).all()

    def test_available_beds_non_negative(self, generated_df):
        assert (generated_df["available_beds"] >= 0).all()

    def test_available_staff_non_negative(self, generated_df):
        assert (generated_df["available_staff"] >= 0).all()

    def test_waiting_time_non_negative(self, generated_df):
        assert (generated_df["waiting_time_avg_min"] >= 0).all()

    def test_all_count_columns_non_negative(self, generated_df):
        count_cols = [
            "total_admissions", "emergency_admissions", "pediatric_admissions",
            "icu_admissions", "available_beds", "available_staff",
            "waiting_time_avg_min", "scheduled_surgeries", "ambulance_arrivals"
        ]
        for col in count_cols:
            assert (generated_df[col] >= 0).all(), f"{col} has negative values"


class TestCapacityConstraints:
    """Test capacity-related constraints."""

    @pytest.fixture
    def generated_df(self):
        config = GeneratorConfig(seed=42, start_date=date(2024, 1, 1), end_date=date(2024, 12, 31))
        gen = HospitalDataGenerator(config)
        return gen.generate()

    def test_occupancy_rate_reasonable(self, generated_df):
        assert (generated_df["bed_occupancy_rate"] >= 0).all()
        assert (generated_df["bed_occupancy_rate"] <= 1.0).all()

    def test_staff_absence_rate_valid(self, generated_df):
        assert (generated_df["staff_absence_rate"] >= 0).all()
        assert (generated_df["staff_absence_rate"] <= 1.0).all()

    def test_stock_level_valid(self, generated_df):
        assert (generated_df["medical_stock_level_pct"] >= 0).all()
        assert (generated_df["medical_stock_level_pct"] <= 100).all()


class TestColumnPresence:
    """Test that all expected columns are present."""

    @pytest.fixture
    def generated_df(self):
        config = GeneratorConfig(seed=42, start_date=date(2024, 1, 1), end_date=date(2024, 1, 31))
        gen = HospitalDataGenerator(config)
        return gen.generate()

    def test_all_expected_columns_present(self, generated_df):
        expected_cols = [
            "date", "dow", "month", "day_of_week", "season", "temperature_c",
            "heatwave_event", "epidemic_level", "strike_level", "accident_event",
            "total_admissions", "emergency_admissions", "pediatric_admissions", "icu_admissions",
            "available_beds", "available_staff", "medical_stock_level_pct", "waiting_time_avg_min",
            "scheduled_surgeries", "avg_patient_severity", "staff_absence_rate", "bed_occupancy_rate",
            "ambulance_arrivals", "external_alert_level", "supply_delivery_day", "it_system_outage",
            "estimated_cost_per_day", "patient_satisfaction_score"
        ]

        for col in expected_cols:
            assert col in generated_df.columns, f"Missing column: {col}"


class TestDateRange:
    """Test that date range is correct."""

    def test_date_range_2025(self):
        config = GeneratorConfig(
            seed=42,
            start_date=date(2023, 1, 1),
            end_date=date(2025, 12, 31)
        )
        gen = HospitalDataGenerator(config)
        df = gen.generate()

        dates = pd.to_datetime(df["date"], format="%d/%m/%Y")
        assert dates.min().year == 2023
        assert dates.max().year == 2025
        assert dates.max().month == 12
        assert dates.max().day == 31

    def test_correct_number_of_days(self):
        config = GeneratorConfig(
            seed=42,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31)
        )
        gen = HospitalDataGenerator(config)
        df = gen.generate()

        assert len(df) == 366  # 2024 is a leap year


class TestValidation:
    """Test the validation functions."""

    def test_validation_passes_for_valid_data(self):
        config = GeneratorConfig(
            seed=42,
            start_date=date(2023, 1, 1),
            end_date=date(2025, 12, 31)
        )
        gen = HospitalDataGenerator(config)
        df = gen.generate()

        report = validate_dataset(df, expected_end_year=2025)
        assert report.is_valid

    def test_validation_detects_missing_column(self):
        config = GeneratorConfig(seed=42, start_date=date(2024, 1, 1), end_date=date(2024, 1, 31))
        gen = HospitalDataGenerator(config)
        df = gen.generate()

        df_incomplete = df.drop(columns=["total_admissions"])
        report = validate_dataset(df_incomplete)

        assert not report.is_valid
        assert any("missing" in issue.issue_type.lower() for issue in report.issues)


class TestSeasonality:
    """Test that seasonality patterns are present."""

    @pytest.fixture
    def generated_df(self):
        config = GeneratorConfig(seed=42, start_date=date(2024, 1, 1), end_date=date(2024, 12, 31))
        gen = HospitalDataGenerator(config)
        return gen.generate()

    def test_weekend_lower_admissions(self, generated_df):
        weekend_mean = generated_df[generated_df["dow"] >= 5]["total_admissions"].mean()
        weekday_mean = generated_df[generated_df["dow"] < 5]["total_admissions"].mean()
        assert weekend_mean < weekday_mean

    def test_winter_higher_admissions(self, generated_df):
        winter_mean = generated_df[generated_df["season"] == "winter"]["total_admissions"].mean()
        summer_mean = generated_df[generated_df["season"] == "summer"]["total_admissions"].mean()
        assert winter_mean > summer_mean
