# Hospital Decision Support System

An interactive decision-support prototype for hospital capacity planning, forecasting, and scenario analysis.

## Features

- **Interactive Dashboard**: Explore historical hospital data with KPIs and visualizations
- **Predictive Forecasting**: ML-based prediction of future admissions
- **Scenario Simulation**: Test "what-if" scenarios (epidemics, strikes, seasonal peaks)
- **Action Recommendations**: Get prioritized suggestions for resource management

## Quick Start

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository** (or download the project):
   ```bash
   git clone <repository-url>
   cd Projet-data
   ```

2. **Create a virtual environment**:
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # Linux/Mac
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```bash
   streamlit run app/Home.py
   ```

5. **Open in browser**: The app will automatically open at `http://localhost:8501`

## Project Structure

```
Projet-data/
├── app/                    # Streamlit application
│   ├── Home.py            # Main entry point
│   └── pages/             # Application pages
│       ├── 1_Dashboard.py
│       ├── 2_Forecast.py
│       ├── 3_Scenarios.py
│       └── 4_Recommendations.py
├── src/                    # Source code modules
│   ├── config.py          # Configuration settings
│   ├── data/              # Data loading & preprocessing
│   ├── features/          # Feature engineering
│   ├── models/            # ML training & prediction
│   ├── scenarios/         # Scenario simulation
│   └── reco/              # Recommendation engine
├── data/                   # Data files
│   ├── raw/               # Raw data files
│   └── processed/         # Processed data files
├── models/                 # Saved model artifacts
├── notebooks/              # Jupyter notebooks
├── reports/                # Generated reports & metrics
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Usage Guide

### Dashboard Page
- View key performance indicators (KPIs)
- Explore time series of admissions
- Analyze distributions and correlations
- Filter by date range

### Forecast Page
- Generate predictions for 7-90 days ahead
- Choose between ML model and baseline
- View model performance metrics
- Download forecast data

### Scenarios Page
- Select preset scenarios (epidemic, strike, etc.)
- Customize scenario parameters with sliders
- Compare baseline vs scenario forecasts
- Analyze capacity gaps and occupancy rates

### Recommendations Page
- Get prioritized action recommendations
- View critical/high/medium/low priority items
- Download reports in CSV or TXT format
- See resource requirements summary

## Data

The system uses daily hospital operations data with the following key variables:

**Target Variable:**
- `total_admissions`: Daily total patient admissions

**Features:**
- Temporal: date, day of week, month, season
- Operational: available beds, staff, medical stock
- External: epidemic level, temperature, events

**Data Period:** 2023-2028 (6 years of daily data)

## Models

### Random Forest Regressor
- **Algorithm**: Ensemble of 300 decision trees
- **Features**: Temporal + operational features (no data leakage)
- **Validation**: Time-based train/test split (80/20)
- **Performance**: R² ≈ 0.82 on test set

### Baseline Model
- **Method**: Seasonal naive (day-of-week + monthly averages)
- **Use case**: Simple, interpretable benchmark

## Scenario Parameters

| Parameter | Range | Description |
|-----------|-------|-------------|
| Epidemic Intensity | 0-100% | Increase in patient demand |
| Staff Reduction | 0-50% | Decrease in available staff |
| Seasonal Multiplier | 0.5-1.5 | Adjust for seasonal patterns |
| Shock Day Spike | 0-200% | Single-day surge |
| Beds Reduction | 0-30% | Decrease in available beds |
| Stock Reduction | 0-50% | Decrease in medical supplies |

## Capacity Assumptions

- Total beds: 1,500
- Total staff: 430
- Normal admission capacity: 450/day
- Critical occupancy threshold: 85%
- Warning threshold: 75%
- Minimum stock level: 50%

## Development

### Running Tests
```bash
pytest tests/
```

### Code Formatting
```bash
black src/ app/
flake8 src/ app/
```

## Acknowledgments

This is an academic project for hospital decision support and capacity planning.

## License

This project is for educational purposes.
