#!/usr/bin/env python3
"""
Generate state-level aggregated data for the website visualization
"""

import pandas as pd
import json

# State name to abbreviation mapping
STATE_ABBREV = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR',
    'California': 'CA', 'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE',
    'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI', 'Idaho': 'ID',
    'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS',
    'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
    'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS',
    'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV',
    'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM', 'New York': 'NY',
    'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK',
    'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC',
    'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT',
    'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV',
    'Wisconsin': 'WI', 'Wyoming': 'WY', 'District of Columbia': 'DC'
}

def main():
    # Load data
    df = pd.read_csv('../data/datacenter_energy_estimates.csv')
    
    # Aggregate by state
    state_data = df.groupby('state').agg({
        'data_center_id': 'count',
        'capacity_mw_est': 'sum',
        'estimated_energy_mwh': 'sum',
    }).reset_index()
    
    state_data.columns = ['state', 'dc_count', 'total_capacity_mw', 'total_energy_mwh']
    
    # Convert to TWh
    state_data['total_energy_twh'] = state_data['total_energy_mwh'] / 1e6
    
    # Add state abbreviations
    state_data['abbrev'] = state_data['state'].map(STATE_ABBREV)
    
    # Fill missing with state name
    state_data['abbrev'] = state_data['abbrev'].fillna(state_data['state'])
    
    # Convert to dict for JSON
    result = {}
    for _, row in state_data.iterrows():
        result[row['abbrev']] = {
            'name': row['state'],
            'dc_count': int(row['dc_count']),
            'capacity_mw': round(row['total_capacity_mw'], 1),
            'energy_mwh': round(row['total_energy_mwh'], 0),
            'energy_twh': round(row['total_energy_twh'], 2),
        }
    
    # Save JSON
    with open('state_data.json', 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"Generated data for {len(result)} states")
    print(f"Total data centers: {state_data['dc_count'].sum():,}")
    print(f"Total energy: {state_data['total_energy_twh'].sum():.1f} TWh/year")
    
    # Also generate summary stats
    summary = {
        'total_datacenters': int(state_data['dc_count'].sum()),
        'total_capacity_gw': round(state_data['total_capacity_mw'].sum() / 1000, 1),
        'total_energy_twh': round(state_data['total_energy_twh'].sum(), 1),
        'max_energy_twh': round(state_data['total_energy_twh'].max(), 1),
        'max_dc_count': int(state_data['dc_count'].max()),
    }
    
    with open('summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

if __name__ == '__main__':
    main()
