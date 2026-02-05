#!/usr/bin/env python
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import load_data, preprocess_data

df = load_data()
original_dow = df['dow'].copy()
original_month = df['month'].copy()

df = preprocess_data(df)

dow_match = (original_dow == df['dow']).all()
month_match = (original_month == df['month']).all()

print('=== CHECKING TEMPORAL COLUMN CONSISTENCY ===')
print()
print('dow (day of week):')
print('  Original values:', sorted(original_dow.unique()))
print('  Computed values:', sorted(df['dow'].unique()))
print('  Match:', dow_match)
print()
print('month:')
print('  Original range:', original_month.min(), '-', original_month.max())
print('  Computed range:', df['month'].min(), '-', df['month'].max())
print('  Match:', month_match)
print()

# Check day_of_week vs dow consistency
print('=== CHECKING day_of_week vs dow ===')
dow_mapping = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
df['expected_day'] = df['dow'].map(dow_mapping)
mismatch = df[df['day_of_week'] != df['expected_day']]
print('Mismatches:', len(mismatch))

if len(mismatch) > 0:
    print('Sample mismatches:')
    print(mismatch[['date', 'dow', 'day_of_week', 'expected_day']].head())
else:
    print('All day_of_week values match computed dow')

# Check is_weekend consistency
print()
print('=== CHECKING is_weekend ===')
df['computed_weekend'] = (df['dow'] >= 5).astype(int)
weekend_in_data = df[df['day_of_week'].isin(['Saturday', 'Sunday'])]
print('Weekend days in data:', len(weekend_in_data))
print('is_weekend=1 count:', (df['is_weekend'] == 1).sum())

# Final summary
print()
print('=== SUMMARY ===')
issues = []
if not dow_match:
    issues.append('dow mismatch')
if not month_match:
    issues.append('month mismatch')
if len(mismatch) > 0:
    issues.append('day_of_week inconsistency')

if issues:
    print('ISSUES FOUND:', ', '.join(issues))
else:
    print('All data processing validated successfully!')
