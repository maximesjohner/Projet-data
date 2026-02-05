# Data Generation Validation Report

**Status:** PASSED

- Rows: 5114
- Columns: 29
- Date range: 2012-01-01 to 2025-12-31

## Comparison with Reference

| Column | Gen Mean | Ref Mean | Diff % | Status |
|--------|----------|----------|--------|--------|
| dow | 3.0 | 3.0 | 0.1 | OK |
| month | 6.52 | 6.52 | 0.0 | OK |
| temperature_c | 13.46 | 12.07 | 11.6 | OK |
| heatwave_event | 0.01 | 0.0 | 997262416.9 | DRIFT |
| epidemic_level | 0.03 | 0.13 | 79.4 | DRIFT |
| strike_level | 0.0 | 0.02 | 97.6 | DRIFT |
| accident_event | 0.01 | 0.01 | 61.6 | DRIFT |
| total_admissions | 423.05 | 399.42 | 5.9 | OK |
| emergency_admissions | 163.3 | 154.74 | 5.5 | OK |
| pediatric_admissions | 33.38 | 31.97 | 4.4 | OK |
| icu_admissions | 17.13 | 16.59 | 3.3 | OK |
| available_beds | 1423.83 | 1417.52 | 0.4 | OK |
| available_staff | 422.18 | 422.99 | 0.2 | OK |
| medical_stock_level_pct | 75.75 | 72.54 | 4.4 | OK |
| waiting_time_avg_min | 46.82 | 47.19 | 0.8 | OK |
| naive_pred_total_admissions | 422.41 | 399.33 | 5.8 | OK |
| scheduled_surgeries | 32.63 | 38.27 | 14.7 | OK |
| avg_patient_severity | 2.3 | 2.41 | 4.4 | OK |
| staff_absence_rate | 0.07 | 0.08 | 14.6 | OK |
| bed_occupancy_rate | 0.3 | 0.48 | 36.6 | DRIFT |
| ambulance_arrivals | 97.45 | 88.77 | 9.8 | OK |
| external_alert_level | 0.02 | 0.06 | 67.5 | DRIFT |
| supply_delivery_day | 0.28 | 0.24 | 17.1 | OK |
| it_system_outage | 0.02 | 0.0 | 302.9 | DRIFT |
| estimated_cost_per_day | 61052.74 | 61643.6 | 1.0 | OK |
| patient_satisfaction_score | 7.5 | 7.7 | 2.6 | OK |
