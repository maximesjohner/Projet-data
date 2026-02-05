#!/usr/bin/env python
"""
Generate synthetic hospital datasets for all hospitals.

Data structure:
  - data/processed/donnees_hopital.csv         -> Pitié-Salpêtrière only (for frontend)
  - data/processed/training/donnees_PITIE.csv  -> Pitié-Salpêtrière (for training)
  - data/processed/training/donnees_HEGP.csv   -> Other hospitals (for training)
  - ...

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
    parser.add_argument("--output", type=str, default=None, help="Output path for main file")
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

    # Create training directory
    training_dir = PROJECT_ROOT / "data" / "processed" / "training"
    training_dir.mkdir(parents=True, exist_ok=True)

    pitie_df = None
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

        # Save individual hospital file for training
        hospital_file = training_dir / f"donnees_{hospital['id']}.csv"
        df.to_csv(hospital_file, sep=";", index=False)
        print(f"  Saved to: {hospital_file.name}")

        all_datasets.append(df)

        # Keep Pitié-Salpêtrière for main frontend file
        if hospital["id"] == "PITIE":
            pitie_df = df.copy()

    # Save Pitié-Salpêtrière as main file (for frontend)
    print(f"\n{'=' * 60}")
    print("SAVING FRONTEND DATA (Pitié-Salpêtrière only)")
    print("=" * 60)

    output_path = Path(args.output) if args.output else PROJECT_ROOT / "data" / "processed" / "donnees_hopital.csv"
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove hospital_id column for frontend (not needed, single hospital)
    pitie_frontend = pitie_df.drop(columns=["hospital_id"], errors="ignore")
    pitie_frontend.to_csv(output_path, sep=";", index=False)
    print(f"Saved Pitié-Salpêtrière data to: {output_path}")
    print(f"  Rows: {len(pitie_frontend)}, Columns: {len(pitie_frontend.columns)}")

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)

    print("\n--- Frontend data (donnees_hopital.csv) ---")
    print(f"Hospital: Pitié-Salpêtrière")
    print(f"Rows: {len(pitie_frontend)}")

    print("\n--- Training data (data/processed/training/) ---")
    combined_df = pd.concat(all_datasets, ignore_index=True)
    print(f"Total rows: {len(combined_df):,}")
    print(f"Hospitals: {combined_df['hospital_id'].nunique()}")

    print("\n--- Per-hospital summary ---")
    summary = combined_df.groupby("hospital_id").agg({
        "total_admissions": ["mean", "std"],
        "emergency_admissions": "mean",
        "available_beds": "mean"
    }).round(1)
    print(summary.to_string())

    if not args.no_validate:
        print("\n" + "=" * 60)
        print("VALIDATION (Pitié-Salpêtrière)")
        print("=" * 60)

        reference_df = load_reference(hospitals[0]["file"])
        report = validate_dataset(pitie_df, reference_df=reference_df, expected_end_year=end_date.year)
        print_validation_report(report)

        report_path = PROJECT_ROOT / args.report
        save_validation_report(report, str(report_path))
        print(f"\nReport saved to: {report_path}")

    print("\n" + "=" * 60)
    print(f"DONE")
    print(f"  Frontend: donnees_hopital.csv (Pitié only)")
    print(f"  Training: {len(hospitals)} files in data/processed/training/")
    print("=" * 60)


if __name__ == "__main__":
    main()
