"""Validation functions for generated data."""
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path


@dataclass
class ValidationIssue:
    """Single validation issue."""
    column: str
    issue_type: str
    message: str
    severity: str = "warning"


@dataclass
class ValidationReport:
    """Complete validation report."""
    is_valid: bool = True
    n_rows: int = 0
    n_cols: int = 0
    date_range: Tuple[str, str] = ("", "")
    issues: List[ValidationIssue] = field(default_factory=list)
    column_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    comparison: Optional[pd.DataFrame] = None

    def add_issue(self, column: str, issue_type: str, message: str, severity: str = "warning"):
        self.issues.append(ValidationIssue(column, issue_type, message, severity))
        if severity == "error":
            self.is_valid = False


def validate_no_negative_counts(df: pd.DataFrame, count_cols: List[str]) -> List[ValidationIssue]:
    """Check that count columns have no negative values."""
    issues = []
    for col in count_cols:
        if col in df.columns:
            neg_count = (df[col] < 0).sum()
            if neg_count > 0:
                issues.append(ValidationIssue(
                    col, "negative_values",
                    f"{neg_count} negative values found",
                    severity="error"
                ))
    return issues


def validate_capacity_constraints(df: pd.DataFrame) -> List[ValidationIssue]:
    """Check capacity-related constraints."""
    issues = []

    if "bed_occupancy_rate" in df.columns:
        over_100 = (df["bed_occupancy_rate"] > 1.0).sum()
        if over_100 > 0:
            issues.append(ValidationIssue(
                "bed_occupancy_rate", "constraint_violation",
                f"{over_100} days with occupancy > 100%",
                severity="warning"
            ))

    if "staff_absence_rate" in df.columns:
        invalid = ((df["staff_absence_rate"] < 0) | (df["staff_absence_rate"] > 1)).sum()
        if invalid > 0:
            issues.append(ValidationIssue(
                "staff_absence_rate", "out_of_range",
                f"{invalid} values outside [0, 1]",
                severity="error"
            ))

    return issues


def validate_missing_values(df: pd.DataFrame) -> List[ValidationIssue]:
    """Check for unexpected missing values."""
    issues = []
    for col in df.columns:
        na_count = df[col].isna().sum()
        if na_count > 0:
            issues.append(ValidationIssue(
                col, "missing_values",
                f"{na_count} missing values",
                severity="warning"
            ))
    return issues


def validate_date_range(df: pd.DataFrame, expected_end_year: int = 2025) -> List[ValidationIssue]:
    """Validate date range coverage."""
    issues = []

    if "date" in df.columns:
        dates = pd.to_datetime(df["date"], format="%d/%m/%Y", errors="coerce")
        max_year = dates.dt.year.max()

        if max_year < expected_end_year:
            issues.append(ValidationIssue(
                "date", "incomplete_range",
                f"Data ends at {max_year}, expected {expected_end_year}",
                severity="error"
            ))

    return issues


def validate_column_presence(df: pd.DataFrame, expected_cols: List[str]) -> List[ValidationIssue]:
    """Check that all expected columns are present."""
    issues = []
    missing = set(expected_cols) - set(df.columns)
    if missing:
        issues.append(ValidationIssue(
            "schema", "missing_columns",
            f"Missing columns: {missing}",
            severity="error"
        ))
    return issues


def compare_with_reference(
    generated: pd.DataFrame, reference: pd.DataFrame, tolerance: float = 0.25
) -> pd.DataFrame:
    """Compare generated data with reference dataset."""
    comparison_data = []

    numeric_cols = generated.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        if col in reference.columns:
            gen_mean = generated[col].mean()
            gen_std = generated[col].std()
            ref_mean = reference[col].mean()
            ref_std = reference[col].std()

            mean_diff_pct = abs(gen_mean - ref_mean) / (ref_mean + 1e-9) * 100
            std_diff_pct = abs(gen_std - ref_std) / (ref_std + 1e-9) * 100

            status = "OK" if mean_diff_pct < tolerance * 100 else "DRIFT"

            comparison_data.append({
                "column": col,
                "gen_mean": round(gen_mean, 2),
                "ref_mean": round(ref_mean, 2),
                "mean_diff_%": round(mean_diff_pct, 1),
                "gen_std": round(gen_std, 2),
                "ref_std": round(ref_std, 2),
                "std_diff_%": round(std_diff_pct, 1),
                "status": status
            })

    return pd.DataFrame(comparison_data)


def validate_dataset(
    df: pd.DataFrame,
    reference_df: Optional[pd.DataFrame] = None,
    expected_end_year: int = 2025
) -> ValidationReport:
    """Complete dataset validation."""
    report = ValidationReport()
    report.n_rows = len(df)
    report.n_cols = len(df.columns)

    if "date" in df.columns:
        dates = pd.to_datetime(df["date"], format="%d/%m/%Y", errors="coerce")
        report.date_range = (
            dates.min().strftime("%Y-%m-%d"),
            dates.max().strftime("%Y-%m-%d")
        )

    expected_cols = [
        "date", "dow", "month", "day_of_week", "season", "temperature_c",
        "heatwave_event", "epidemic_level", "strike_level", "accident_event",
        "total_admissions", "emergency_admissions", "pediatric_admissions", "icu_admissions",
        "available_beds", "available_staff", "medical_stock_level_pct", "waiting_time_avg_min",
        "scheduled_surgeries", "avg_patient_severity", "staff_absence_rate", "bed_occupancy_rate",
        "ambulance_arrivals", "external_alert_level", "supply_delivery_day", "it_system_outage",
        "estimated_cost_per_day", "patient_satisfaction_score"
    ]

    count_cols = [
        "total_admissions", "emergency_admissions", "pediatric_admissions", "icu_admissions",
        "available_beds", "available_staff", "waiting_time_avg_min", "scheduled_surgeries",
        "ambulance_arrivals", "estimated_cost_per_day"
    ]

    for issue in validate_column_presence(df, expected_cols):
        report.issues.append(issue)
        if issue.severity == "error":
            report.is_valid = False

    for issue in validate_no_negative_counts(df, count_cols):
        report.issues.append(issue)
        if issue.severity == "error":
            report.is_valid = False

    for issue in validate_capacity_constraints(df):
        report.issues.append(issue)
        if issue.severity == "error":
            report.is_valid = False

    for issue in validate_missing_values(df):
        report.issues.append(issue)

    for issue in validate_date_range(df, expected_end_year):
        report.issues.append(issue)
        if issue.severity == "error":
            report.is_valid = False

    for col in df.select_dtypes(include=[np.number]).columns:
        report.column_stats[col] = {
            "min": df[col].min(),
            "max": df[col].max(),
            "mean": df[col].mean(),
            "std": df[col].std(),
        }

    if reference_df is not None:
        report.comparison = compare_with_reference(df, reference_df)

    return report


def print_validation_report(report: ValidationReport) -> None:
    """Print validation report to console."""
    print("=" * 60)
    print("VALIDATION REPORT")
    print("=" * 60)

    status = "PASSED" if report.is_valid else "FAILED"
    print(f"\nStatus: {status}")
    print(f"Rows: {report.n_rows}")
    print(f"Columns: {report.n_cols}")
    print(f"Date range: {report.date_range[0]} to {report.date_range[1]}")

    if report.issues:
        print(f"\nIssues ({len(report.issues)}):")
        for issue in report.issues:
            print(f"  [{issue.severity.upper()}] {issue.column}: {issue.message}")
    else:
        print("\nNo issues found.")

    if report.comparison is not None and len(report.comparison) > 0:
        print("\n--- Comparison with Reference ---")
        drifted = report.comparison[report.comparison["status"] == "DRIFT"]
        if len(drifted) > 0:
            print(f"Columns with drift: {list(drifted['column'])}")
        else:
            print("All columns within tolerance.")


def save_validation_report(report: ValidationReport, output_path: str) -> None:
    """Save validation report to markdown file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        f.write("# Data Generation Validation Report\n\n")
        f.write(f"**Status:** {'PASSED' if report.is_valid else 'FAILED'}\n\n")
        f.write(f"- Rows: {report.n_rows}\n")
        f.write(f"- Columns: {report.n_cols}\n")
        f.write(f"- Date range: {report.date_range[0]} to {report.date_range[1]}\n\n")

        if report.issues:
            f.write("## Issues\n\n")
            for issue in report.issues:
                f.write(f"- **{issue.severity.upper()}** [{issue.column}]: {issue.message}\n")
            f.write("\n")

        if report.comparison is not None:
            f.write("## Comparison with Reference\n\n")
            f.write("| Column | Gen Mean | Ref Mean | Diff % | Status |\n")
            f.write("|--------|----------|----------|--------|--------|\n")
            for _, row in report.comparison.iterrows():
                f.write(f"| {row['column']} | {row['gen_mean']} | {row['ref_mean']} | "
                        f"{row['mean_diff_%']} | {row['status']} |\n")
