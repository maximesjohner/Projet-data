#!/usr/bin/env python
"""
Simple CLI to run the Hospital Decision Support System.

Usage:
    python run.py app                  # Run Streamlit app
    python run.py generate             # Generate data (2012-2025)
    python run.py generate --seed 123  # Generate with custom seed
    python run.py test                 # Run tests
    python run.py health               # Run health check
"""
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent


def run_app():
    """Run the Streamlit application."""
    try:
        print("Starting Streamlit app...")
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            str(PROJECT_ROOT / "app" / "Home.py")
        ])
    except KeyboardInterrupt:
        return


def run_generate(args):
    """Generate synthetic data."""
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "generate_dataset.py")]
    cmd.extend(args)
    subprocess.run(cmd)


def run_tests():
    """Run pytest tests."""
    print("Running tests...")
    subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"])


def run_health():
    """Run health check."""
    subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "health_check.py")])


def show_help():
    """Show help message."""
    print("""
Hospital Decision Support System - CLI

Commands:
    python run.py app                    Run Streamlit application
    python run.py generate               Generate data (2012-2025, seed=42)
    python run.py generate --seed 123    Generate with custom seed
    python run.py test                   Run all tests
    python run.py health                 Run health check

Examples:
    python run.py app
    python run.py generate
    python run.py generate --seed 99 --start-date 2015-01-01
    """)


def main():
    if len(sys.argv) < 2:
        show_help()
        return

    command = sys.argv[1].lower()
    extra_args = sys.argv[2:]

    if command in ["app", "start", "run"]:
        run_app()
    elif command in ["generate", "gen", "data"]:
        run_generate(extra_args)
    elif command in ["test", "tests"]:
        run_tests()
    elif command in ["health", "check"]:
        run_health()
    elif command in ["help", "-h", "--help"]:
        show_help()
    else:
        print(f"Unknown command: {command}")
        show_help()


if __name__ == "__main__":
    main()
