# Hospital Decision Support System

Interactive dashboard for hospital capacity planning, forecasting, and scenario analysis.

[![CI Pipeline](https://github.com/maximesjohner/Projet-data/actions/workflows/ci.yml/badge.svg)](https://github.com/maximesjohner/Projet-data/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.30+-red.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)

## Quick Start

### Option 1: Docker (Recommended)

```bash
# Run with Docker Compose
docker-compose up --build

# Access the app at http://localhost:8501
```

### Option 2: Local Installation

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate data (2012-2025)
python run.py generate

# 3. Run the app
python run.py app
```

## Commands

| Command | Description |
|---------|-------------|
| `python run.py app` | Run Streamlit application |
| `python run.py generate` | Generate synthetic data (2012-2025) |
| `python run.py test` | Run all tests |
| `python run.py health` | Run health check |

### Generate Options

```bash
python run.py generate                              # Default (2012-2025, seed=42)
python run.py generate --seed 123                   # Custom seed
python run.py generate --start-date 2020-01-01      # Custom start date
python run.py generate --end-date 2030-12-31        # Custom end date
```

## Features

- **Dashboard**: KPIs, time series, distributions, correlations
- **Forecast**: ML-based predictions (7-90 days)
- **Scenarios**: Epidemic, strike, seasonal peaks, shock events
- **Recommendations**: Prioritized actions for capacity management

## Project Structure

```
├── app/                    # Streamlit pages
├── src/
│   ├── data/              # Data loading
│   ├── features/          # Feature engineering
│   ├── models/            # ML training & prediction
│   ├── scenarios/         # Scenario simulation
│   ├── reco/              # Recommendations
│   └── generator/         # Synthetic data generation
├── data/
│   ├── reference/         # Calibration data (11 hospitals)
│   └── processed/
│       ├── donnees_hopital.csv  # Frontend (Pitié only)
│       └── training/            # Model training (all hospitals)
├── config/                # Configuration
├── scripts/               # CLI scripts
├── tests/                 # Pytest tests
├── .github/workflows/     # CI/CD pipeline
├── Dockerfile             # Docker image definition
├── docker-compose.yml     # Docker orchestration
└── run.py                 # Main CLI entry point
```

## Data Generation

The generator creates realistic synthetic hospital data with:

- **Seasonality**: Higher admissions in winter, lower on weekends
- **Events**: Epidemics (+10%/level), strikes (-15% staff), heatwaves
- **Capacity**: Beds/staff vary slowly, stock depletes and restocks
- **Correlations**: Wait time increases with load, severity with epidemics
- **Constraints**: No negative values, valid rates

### Data Structure

```
data/
├── reference/                    # Calibration data (11 hospitals)
│   ├── donnees_hopital_reference.csv
│   └── donnees_hopital_reference_*_*.csv
└── processed/
    ├── donnees_hopital.csv       # Frontend data (Pitié-Salpêtrière only)
    └── training/                 # Model training data (all hospitals)
        ├── donnees_PITIE.csv
        ├── donnees_HEGP.csv
        └── ...
```

- **Frontend**: Uses `donnees_hopital.csv` (single hospital)
- **Model training**: Uses all files in `training/` (11 hospitals, ~56k rows)

## Model

- **Algorithm**: Random Forest (300 trees)
- **Features**: Temporal + operational (no data leakage)
- **Training data**: 11 hospitals (~56k rows)
- **Validation**: Time-based split (80/20)
- **Performance**: R² ≈ 0.89

## Docker

### Running with Docker Compose

```bash
# Build and start the application
docker-compose up --build

# Run in detached mode
docker-compose up -d --build

# Stop the application
docker-compose down

# View logs
docker-compose logs -f hospital-dss
```

### Docker Utilities

```bash
# Generate fresh data
docker-compose --profile tools run generate-data

# Run tests in container
docker-compose --profile tools run test

# Run health check
docker-compose --profile tools run health-check
```

### Pull from GitHub Container Registry

```bash
# Pull the latest image
docker pull ghcr.io/YOUR_USERNAME/projet-data:latest

# Run directly
docker run -p 8501:8501 ghcr.io/YOUR_USERNAME/projet-data:latest
```

### Build Manually

```bash
# Build the image
docker build -t hospital-dss .

# Run the container
docker run -p 8501:8501 hospital-dss
```

## Requirements

- Python 3.9+
- See `requirements.txt`
