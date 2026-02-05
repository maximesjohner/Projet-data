"""Core synthetic data generation logic."""
import pandas as pd
import numpy as np
from datetime import date, timedelta
from typing import Dict, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.generator import GeneratorConfig, DEFAULT_CONFIG


class HospitalDataGenerator:
    """Generator for synthetic hospital data."""

    def __init__(self, config: Optional[GeneratorConfig] = None):
        self.config = config or DEFAULT_CONFIG
        self.rng = np.random.default_rng(self.config.seed)
        self.reference_stats = self._load_reference_stats()

    def _load_reference_stats(self) -> Dict:
        """Load statistics from reference file for calibration."""
        ref_path = Path(self.config.reference_path)
        if not ref_path.exists():
            return {}

        df = pd.read_csv(ref_path, sep=";")
        stats = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            stats[col] = {
                "mean": df[col].mean(),
                "std": df[col].std(),
                "min": df[col].min(),
                "max": df[col].max(),
            }
        return stats

    def generate(self) -> pd.DataFrame:
        """Generate complete synthetic dataset."""
        dates = self._generate_dates()
        n_days = len(dates)

        df = pd.DataFrame({"date": dates})
        df = self._add_temporal_features(df)
        df = self._generate_events(df, n_days)
        df = self._generate_capacity(df, n_days)
        df = self._generate_admissions(df, n_days)
        df = self._generate_derived_metrics(df, n_days)
        df = self._format_output(df)

        return df

    def _generate_dates(self) -> list:
        current = self.config.start_date
        dates = []
        while current <= self.config.end_date:
            dates.append(current)
            current += timedelta(days=1)
        return dates

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["_date_dt"] = pd.to_datetime(df["date"])
        df["dow"] = df["_date_dt"].dt.dayofweek
        df["month"] = df["_date_dt"].dt.month
        df["year"] = df["_date_dt"].dt.year

        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        df["day_of_week"] = df["dow"].map(lambda x: day_names[x])

        def get_season(m):
            if m in [12, 1, 2]: return "winter"
            if m in [3, 4, 5]: return "spring"
            if m in [6, 7, 8]: return "summer"
            return "autumn"

        df["season"] = df["month"].map(get_season)
        return df

    def _generate_events(self, df: pd.DataFrame, n: int) -> pd.DataFrame:
        cfg = self.config.events

        df["epidemic_level"] = self._generate_epidemic_events(n)
        df["strike_level"] = self._generate_clustered_events(n, cfg.strike_prob, cfg.strike_duration_range)
        df["heatwave_event"] = self._generate_seasonal_events(df, cfg.heatwave_prob, cfg.heatwave_duration_range, cfg.heatwave_months)
        df["accident_event"] = (self.rng.random(n) < cfg.accident_prob).astype(int)
        df["it_system_outage"] = (self.rng.random(n) < 0.015).astype(int)
        df["supply_delivery_day"] = (self.rng.random(n) < 0.28).astype(int)

        return df

    def _generate_epidemic_events(self, n: int) -> np.ndarray:
        cfg = self.config.events
        levels = np.zeros(n, dtype=int)
        i = 0
        while i < n:
            if self.rng.random() < cfg.epidemic_prob / 60:
                duration = self.rng.integers(*cfg.epidemic_duration_range)
                intensity = self.rng.integers(1, 4)
                end = min(i + duration, n)
                levels[i:end] = intensity
                i = end + self.rng.integers(30, 90)
            else:
                i += 1
        return levels

    def _generate_clustered_events(self, n: int, prob: float, duration_range: tuple) -> np.ndarray:
        events = np.zeros(n, dtype=int)
        i = 0
        while i < n:
            if self.rng.random() < prob / 20:
                duration = self.rng.integers(*duration_range)
                end = min(i + duration, n)
                events[i:end] = 1
                i = end + self.rng.integers(60, 180)
            else:
                i += 1
        return events

    def _generate_seasonal_events(self, df: pd.DataFrame, prob: float, duration_range: tuple, months: tuple) -> np.ndarray:
        n = len(df)
        events = np.zeros(n, dtype=int)
        in_season = df["month"].isin(months).values
        i = 0
        while i < n:
            if in_season[i] and self.rng.random() < prob / 10:
                duration = self.rng.integers(*duration_range)
                end = min(i + duration, n)
                events[i:end] = 1
                i = end + self.rng.integers(20, 60)
            else:
                i += 1
        return events

    def _generate_capacity(self, df: pd.DataFrame, n: int) -> pd.DataFrame:
        cfg = self.config.capacity
        ref = self.reference_stats

        beds_mean = ref.get("available_beds", {}).get("mean", cfg.total_beds)
        beds_std = ref.get("available_beds", {}).get("std", cfg.beds_std)
        staff_mean = ref.get("available_staff", {}).get("mean", cfg.total_staff)
        staff_std = ref.get("available_staff", {}).get("std", cfg.staff_std)

        base_beds = beds_mean + self.rng.normal(0, beds_std / 5, n).cumsum() / 100
        beds_noise = self.rng.normal(0, beds_std, n)
        strike_reduction = df["strike_level"].values * 0.10 * beds_mean
        df["available_beds"] = np.clip(base_beds + beds_noise - strike_reduction, 1200, 1600).astype(int)

        staff_noise = self.rng.normal(0, staff_std, n)
        strike_effect = df["strike_level"].values * staff_mean * 0.15
        epidemic_effect = df["epidemic_level"].values * 3
        df["available_staff"] = np.clip(staff_mean + staff_noise - strike_effect - epidemic_effect, 370, 460).astype(int)

        stock = np.zeros(n)
        stock[0] = 72.0
        for i in range(1, n):
            usage = 1.8 + self.rng.normal(0, 0.4)
            delivery = 7.0 if df["supply_delivery_day"].iloc[i] else 0
            stock[i] = np.clip(stock[i-1] - usage + delivery + self.rng.normal(0, 0.8), 40, 95)
        df["medical_stock_level_pct"] = np.round(stock, 1)

        return df

    def _generate_admissions(self, df: pd.DataFrame, n: int) -> pd.DataFrame:
        cfg = self.config.capacity
        ref = self.reference_stats
        seasonality = self.config.seasonality

        base = ref.get("total_admissions", {}).get("mean", cfg.base_admissions)

        years_elapsed = (df["year"] - df["year"].min()).values
        trend = 1 + self.config.trend_annual_growth * years_elapsed

        weekday_factor = df["dow"].map(seasonality.weekday_factors).values
        month_factor = df["month"].map(seasonality.month_factors).values

        epidemic_mult = 1 + df["epidemic_level"].values * 0.10
        heatwave_mult = 1 + df["heatwave_event"].values * 0.06
        accident_mult = 1 + df["accident_event"].values * 0.40

        combined = trend * weekday_factor * month_factor * epidemic_mult * heatwave_mult * accident_mult
        noise = self.rng.normal(0, cfg.admissions_std * self.config.noise_level, n)

        df["total_admissions"] = np.clip(base * combined + noise, 250, 620).astype(int)

        emerg_ratio = ref.get("emergency_admissions", {}).get("mean", 155) / base
        df["emergency_admissions"] = (df["total_admissions"] * emerg_ratio * (1 + self.rng.normal(0, 0.06, n))).astype(int)

        ped_ratio = ref.get("pediatric_admissions", {}).get("mean", 32) / base
        df["pediatric_admissions"] = (df["total_admissions"] * ped_ratio * (1 + self.rng.normal(0, 0.10, n))).astype(int)

        icu_ratio = ref.get("icu_admissions", {}).get("mean", 20) / base
        df["icu_admissions"] = (df["total_admissions"] * icu_ratio * (1 + df["epidemic_level"].values * 0.12 + self.rng.normal(0, 0.12, n))).astype(int)

        df["ambulance_arrivals"] = (df["emergency_admissions"] * 0.6 * (1 + self.rng.normal(0, 0.08, n))).astype(int)

        surgery_base = ref.get("scheduled_surgeries", {}).get("mean", 42) * (1 - df["strike_level"].values * 0.25)
        weekend_reduction = (df["dow"] >= 5).astype(int) * 18
        df["scheduled_surgeries"] = np.clip(surgery_base - weekend_reduction + self.rng.normal(0, 4, n), 15, 55).astype(int)

        return df

    def _generate_derived_metrics(self, df: pd.DataFrame, n: int) -> pd.DataFrame:
        ref = self.reference_stats

        df["bed_occupancy_rate"] = np.round(np.clip(df["total_admissions"] / df["available_beds"] * 0.85, 0.30, 0.72), 3)

        base_absence = 0.07
        epidemic_effect = df["epidemic_level"].values * 0.012
        strike_effect = df["strike_level"].values * 0.025
        df["staff_absence_rate"] = np.round(np.clip(base_absence + epidemic_effect + strike_effect + self.rng.normal(0, 0.008, n), 0.04, 0.14), 2)

        load_ratio = df["total_admissions"] / df["available_staff"]
        wait_base = ref.get("waiting_time_avg_min", {}).get("mean", 47)
        load_effect = (load_ratio - load_ratio.mean()) * 25
        epidemic_effect = df["epidemic_level"].values * 8
        df["waiting_time_avg_min"] = np.clip(wait_base + load_effect + epidemic_effect + self.rng.normal(0, 6, n), 15, 115).astype(int)

        df["avg_patient_severity"] = np.round(np.clip(2.3 + df["epidemic_level"].values * 0.20 + self.rng.normal(0, 0.12, n), 2.0, 3.8), 1)

        df["external_alert_level"] = np.clip(df["epidemic_level"] + df["accident_event"] + (df["heatwave_event"] * 2) - 1, 0, 2).astype(int)

        month_temp = {1: 4, 2: 5, 3: 9, 4: 13, 5: 17, 6: 21, 7: 24, 8: 23, 9: 19, 10: 13, 11: 8, 12: 5}
        base_temp = df["month"].map(month_temp).values.astype(float)
        heatwave_effect = df["heatwave_event"].values * 7
        df["temperature_c"] = np.round(np.clip(base_temp + heatwave_effect + self.rng.normal(0, 2.5, n), -6, 38), 1)

        df["naive_pred_total_admissions"] = (df["total_admissions"].rolling(7, min_periods=1).mean().shift(1).fillna(400) + self.rng.normal(0, 15, n)).astype(int)

        cost_base = ref.get("estimated_cost_per_day", {}).get("mean", 60000)
        df["estimated_cost_per_day"] = (cost_base * 0.7 + df["total_admissions"] * 35 + df["icu_admissions"] * 180 + self.rng.normal(0, 1500, n)).astype(int)

        wait_effect = -0.015 * (df["waiting_time_avg_min"] - 47)
        epidemic_effect = -0.08 * df["epidemic_level"].values
        df["patient_satisfaction_score"] = np.round(np.clip(7.5 + wait_effect + epidemic_effect + self.rng.normal(0, 0.18, n), 5.5, 8.5), 2)

        return df

    def _format_output(self, df: pd.DataFrame) -> pd.DataFrame:
        df["date"] = df["date"].apply(lambda d: d.strftime("%d/%m/%Y"))

        columns_order = [
            "date", "dow", "month", "day_of_week", "season", "temperature_c",
            "heatwave_event", "epidemic_level", "strike_level", "accident_event",
            "total_admissions", "emergency_admissions", "pediatric_admissions", "icu_admissions",
            "available_beds", "available_staff", "medical_stock_level_pct", "waiting_time_avg_min",
            "naive_pred_total_admissions", "scheduled_surgeries", "avg_patient_severity",
            "staff_absence_rate", "bed_occupancy_rate", "ambulance_arrivals",
            "external_alert_level", "supply_delivery_day", "it_system_outage",
            "estimated_cost_per_day", "patient_satisfaction_score"
        ]

        return df[[c for c in columns_order if c in df.columns]]


def generate_dataset(seed: int = 42, start_date: Optional[date] = None, end_date: Optional[date] = None, output_path: Optional[str] = None) -> pd.DataFrame:
    config = GeneratorConfig(seed=seed)
    if start_date:
        config.start_date = start_date
    if end_date:
        config.end_date = end_date
    if output_path:
        config.output_path = output_path

    generator = HospitalDataGenerator(config)
    return generator.generate()
