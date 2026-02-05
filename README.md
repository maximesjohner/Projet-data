# Hospital Decision Support System

Interactive dashboard for hospital capacity planning, forecasting, and scenario analysis.

## Quick Start

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
│   ├── reference/         # Calibration data
│   └── processed/         # Generated data (used by app)
├── config/                # Configuration
├── scripts/               # CLI scripts
├── tests/                 # Pytest tests
└── run.py                 # Main CLI entry point
```

## Data Generation

The generator creates realistic synthetic hospital data with:

- **Seasonality**: Higher admissions in winter, lower on weekends
- **Events**: Epidemics (+10%/level), strikes (-15% staff), heatwaves
- **Capacity**: Beds/staff vary slowly, stock depletes and restocks
- **Correlations**: Wait time increases with load, severity with epidemics
- **Constraints**: No negative values, valid rates

Reference file (`data/reference/`) provides calibration statistics.

## Model

- **Algorithm**: Random Forest (300 trees)
- **Features**: Temporal + operational (no data leakage)
- **Validation**: Time-based split (80/20)
- **Performance**: R² ≈ 0.82

## Requirements

- Python 3.9+
- See `requirements.txt`
