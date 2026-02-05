#!/usr/bin/env python
"""
Generate synthetic hospital datasets for all hospitals.

Usage:
    python scripts/generate_dataset.py
    python scripts/generate_dataset.py --seed 123 --end-date 2025-12-31
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from config.generator import GeneratorConfig, REFERENCE_DIR, get_available_hospitals
from src.generator.core import HospitalDataGenerator
from src.generator.validators import validate_dataset, print_validation_report, save_validation_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic hospital datasets")

    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--start-date", type=str, default="2012-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, default="2025-12-31", help="End date YYYY-MM-DD")
    parser.add_argument("--output", type=str, default=None, help="Output path (default: data/processed/donnees_hopital.csv)")
    parser.add_argument("--no-validate", action="store_true", help="Skip validation")
    parser.add_argument("--report", type=str, default="reports/generation_report.md", help="Report path")

    return parser.parse_args()


def load_reference(hospital_file: str = "donnees_hopital_reference.csv") -> pd.DataFrame:
    ref_path = REFERENCE_DIR / hospital_file
    if ref_path.exists():
        return pd.read_csv(ref_path, sep=";")
    return None


def main():
    args = parse_args()

    print("=" * 60)
    print("MULTI-HOSPITAL DATA GENERATOR")
    print("=" * 60)

    start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date()

    hospitals = get_available_hospitals()
    print(f"\nFound {len(hospitals)} hospitals with reference data:")
    for h in hospitals:
        print(f"  - {h['id']}: {h['name']}")

    print(f"\nConfiguration:")
    print(f"  Base seed: {args.seed}")
    print(f"  Period: {start_date} to {end_date}")

    all_datasets = []

    for i, hospital in enumerate(hospitals):
        print(f"\n{'=' * 40}")
        print(f"Generating data for {hospital['id']} ({hospital['name']})...")
        print(f"{'=' * 40}")

        hospital_seed = args.seed + i

        ref_path = str(REFERENCE_DIR / hospital["file"])
        config = GeneratorConfig(
            seed=hospital_seed,
            start_date=start_date,
            end_date=end_date,
            reference_path=ref_path
        )

        generator = HospitalDataGenerator(config, hospital_id=hospital["id"])
        df = generator.generate()

        print(f"  Generated {len(df)} rows")
        all_datasets.append(df)

    print(f"\n{'=' * 60}")
    print("COMBINING DATASETS")
    print("=" * 60)

    combined_df = pd.concat(all_datasets, ignore_index=True)
    print(f"Combined dataset: {len(combined_df)} rows, {len(combined_df.columns)} columns")
    print(f"Hospitals: {combined_df['hospital_id'].nunique()}")

    output_path = Path(args.output) if args.output else PROJECT_ROOT / "data" / "processed" / "donnees_hopital.csv"
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_path, sep=";", index=False)
    print(f"Saved combined data to: {output_path}")

    print("\n--- Sample (first 5 rows) ---")
    sample_cols = ["hospital_id", "date", "total_admissions", "emergency_admissions", "available_beds"]
    print(combined_df[sample_cols].head().to_string())

    print("\n--- Per-hospital summary ---")
    summary = combined_df.groupby("hospital_id").agg({
        "total_admissions": ["mean", "std"],
        "emergency_admissions": "mean",
        "available_beds": "mean"
    }).round(1)
    print(summary.to_string())

    if not args.no_validate:
        print("\n" + "=" * 60)
        print("VALIDATION (first hospital - PITIE)")
        print("=" * 60)

        first_hospital_df = all_datasets[0]
        reference_df = load_reference(hospitals[0]["file"])
        report = validate_dataset(first_hospital_df, reference_df=reference_df, expected_end_year=end_date.year)
        print_validation_report(report)

        report_path = PROJECT_ROOT / args.report
        save_validation_report(report, str(report_path))
        print(f"\nReport saved to: {report_path}")

    print("\n" + "=" * 60)
    print(f"DONE - Generated data for {len(hospitals)} hospitals")
    print(f"Total records: {len(combined_df):,}")
    print("Streamlit will now use this combined data")
    print("=" * 60)


if __name__ == "__main__":
    main()
