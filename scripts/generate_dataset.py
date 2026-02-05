#!/usr/bin/env python
"""
Generate synthetic hospital dataset.

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
from config.generator import GeneratorConfig, DEFAULT_CONFIG
from src.generator.core import HospitalDataGenerator
from src.generator.validators import validate_dataset, print_validation_report, save_validation_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic hospital dataset")

    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--start-date", type=str, default="2012-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, default="2025-12-31", help="End date YYYY-MM-DD")
    parser.add_argument("--output", type=str, default=None, help="Output path (default: data/processed/donnees_hopital.csv)")
    parser.add_argument("--no-validate", action="store_true", help="Skip validation")
    parser.add_argument("--report", type=str, default="reports/generation_report.md", help="Report path")

    return parser.parse_args()


def load_reference() -> pd.DataFrame:
    ref_path = PROJECT_ROOT / "data" / "reference" / "donnees_hopital_reference.csv"
    if ref_path.exists():
        return pd.read_csv(ref_path, sep=";")
    return None


def main():
    args = parse_args()

    print("=" * 60)
    print("HOSPITAL DATA GENERATOR")
    print("=" * 60)

    start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date()

    config = GeneratorConfig(seed=args.seed, start_date=start_date, end_date=end_date)

    if args.output:
        config.output_path = args.output

    print(f"\nConfiguration:")
    print(f"  Seed: {config.seed}")
    print(f"  Period: {config.start_date} to {config.end_date}")
    print(f"  Reference: {Path(config.reference_path).name}")
    print(f"  Output: {config.output_path}")

    print("\nGenerating synthetic data...")
    generator = HospitalDataGenerator(config)
    df = generator.generate()

    print(f"\nGenerated {len(df)} rows, {len(df.columns)} columns")

    output_path = Path(config.output_path)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, sep=";", index=False)
    print(f"Saved to: {output_path}")

    print("\n--- Sample (first 5 rows) ---")
    print(df[["date", "total_admissions", "emergency_admissions", "available_beds", "epidemic_level"]].head().to_string())

    if not args.no_validate:
        print("\n" + "=" * 60)
        print("VALIDATION")
        print("=" * 60)

        reference_df = load_reference()
        report = validate_dataset(df, reference_df=reference_df, expected_end_year=end_date.year)
        print_validation_report(report)

        report_path = PROJECT_ROOT / args.report
        save_validation_report(report, str(report_path))
        print(f"\nReport saved to: {report_path}")

    print("\n" + "=" * 60)
    print("DONE - Streamlit will now use this data")
    print("=" * 60)


if __name__ == "__main__":
    main()
