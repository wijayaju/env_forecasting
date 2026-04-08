#!/usr/bin/env python3
"""
Generate state_data.json for the updated website with crypto vs AI separation.
"""

import pandas as pd
import json

# Load categorized data
dc = pd.read_csv('../data/datacenter_categorized.csv')

# Load EIA data for state electricity totals
eia = pd.read_csv('../data/eia_state_electricity_real.csv')
latest_eia = eia[eia['year'] == eia['year'].max()][['state', 'total_consumption_mwh']]

# State name to abbreviation mapping
STATE_ABBREV = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA',
    'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'District of Columbia': 'DC',
    'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL',
    'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA',
    'Maine': 'ME', 'Maryland': 'MD', 'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN',
    'Mississippi': 'MS', 'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV',
    'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM', 'New York': 'NY',
    'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK', 'Oregon': 'OR',
    'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC', 'South Dakota': 'SD',
    'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT', 'Virginia': 'VA',
    'Washington': 'WA', 'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY'
}

# Aggregate by state
state_summary = dc.groupby('state').agg({
    'data_center_id': 'count',
    'capacity_mw_est': 'sum',
    'estimated_energy_mwh': 'sum',
}).reset_index()
state_summary.columns = ['state', 'dc_count', 'capacity_mw', 'energy_mwh']

# Add category breakdowns
for cat in ['small', 'decent', 'crypto', 'big_ai']:
    cat_df = dc[dc['dc_category'] == cat].groupby('state').agg({
        'data_center_id': 'count',
        'capacity_mw_est': 'sum',
        'estimated_energy_mwh': 'sum',
    }).reset_index()
    cat_df.columns = ['state', f'{cat}_count', f'{cat}_capacity_mw', f'{cat}_energy_mwh']
    state_summary = state_summary.merge(cat_df, on='state', how='left')

state_summary = state_summary.fillna(0)

# Add EIA data
state_summary = state_summary.merge(latest_eia, on='state', how='left')

# Convert to JSON format
state_data = {}
for _, row in state_summary.iterrows():
    abbrev = STATE_ABBREV.get(row['state'])
    if abbrev:
        state_data[abbrev] = {
            'name': row['state'],
            'dc_count': int(row['dc_count']),
            'capacity_mw': round(row['capacity_mw'], 1),
            'energy_mwh': round(row['energy_mwh']),
            'energy_twh': round(row['energy_mwh'] / 1e6, 2),
            'small_count': int(row['small_count']),
            'decent_count': int(row['decent_count']),
            'crypto_count': int(row['crypto_count']),
            'ai_count': int(row['big_ai_count']),
            'crypto_energy_mwh': round(row['crypto_energy_mwh']),
            'ai_energy_mwh': round(row['big_ai_energy_mwh']),
            'state_total_mwh': round(row['total_consumption_mwh']) if pd.notna(row['total_consumption_mwh']) else None,
        }

# Save to JSON
with open('state_data.json', 'w') as f:
    json.dump(state_data, f, indent=2)

print(f"Generated state_data.json with {len(state_data)} states")

# Also create summary.json
summary = {
    'total_dcs': int(dc['data_center_id'].count()),
    'small_dcs': int((dc['dc_category'] == 'small').sum()),
    'decent_dcs': int((dc['dc_category'] == 'decent').sum()),
    'crypto_dcs': int((dc['dc_category'] == 'crypto').sum()),
    'ai_dcs': int((dc['dc_category'] == 'big_ai').sum()),
    'total_capacity_mw': round(dc['capacity_mw_est'].sum(), 1),
    'total_energy_twh': round(dc['estimated_energy_mwh'].sum() / 1e6, 2),
    'crypto_energy_twh': round(dc[dc['dc_category'] == 'crypto']['estimated_energy_mwh'].sum() / 1e6, 2),
    'ai_energy_twh': round(dc[dc['dc_category'] == 'big_ai']['estimated_energy_mwh'].sum() / 1e6, 2),
}

with open('summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"Generated summary.json")
print(f"\nSummary:")
for k, v in summary.items():
    print(f"  {k}: {v}")
