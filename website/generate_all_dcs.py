import pandas as pd
import json

# Load all data centers (original scraped)
df_all = pd.read_csv('../data_centers.csv')
print(f'Total scraped: {len(df_all)}')

# Load categorized (621 with energy data)
df_cat = pd.read_csv('../data/datacenter_categorized.csv')
print(f'With energy data: {len(df_cat)}')

# Create lookup from categorized data by name
cat_lookup = {}
for _, row in df_cat.iterrows():
    cat_lookup[row['data_center_name']] = {
        'year': int(row['year_operational']) if pd.notna(row['year_operational']) else None,
        'capacity_mw': round(float(row['capacity_mw_est']), 1) if pd.notna(row['capacity_mw_est']) else None,
        'energy_mwh': round(float(row['estimated_energy_mwh'])) if pd.notna(row['estimated_energy_mwh']) else None,
        'category': row['dc_category']
    }

# Build full list - ensure no NaN values (use None instead)
dcs = []
for _, row in df_all.iterrows():
    name = str(row['Name']) if pd.notna(row['Name']) else 'Unknown'
    state = str(row['State']) if pd.notna(row['State']) else 'Unknown'
    city = str(row['City']) if pd.notna(row['City']) else 'Unknown'
    
    # Check if we have categorized data
    if name in cat_lookup:
        cat = cat_lookup[name]
        dcs.append({
            'name': name,
            'state': state,
            'city': city,
            'year': cat['year'],
            'capacity_mw': cat['capacity_mw'],
            'energy_mwh': cat['energy_mwh'],
            'category': cat['category'],
            'has_energy_data': True
        })
    else:
        dcs.append({
            'name': name,
            'state': state,
            'city': city,
            'year': None,
            'capacity_mw': None,
            'energy_mwh': None,
            'category': 'unknown',
            'has_energy_data': False
        })

with open('datacenters.json', 'w') as f:
    json.dump(dcs, f)

print(f'Created datacenters.json with {len(dcs)} entries')
print(f'With energy data: {sum(1 for d in dcs if d["has_energy_data"])}')
print(f'Unknown: {sum(1 for d in dcs if not d["has_energy_data"])}')

# Also update state_data.json to show full counts
state_counts = {}
for dc in dcs:
    state = dc['state']
    if state not in state_counts:
        state_counts[state] = {'total': 0, 'with_data': 0}
    state_counts[state]['total'] += 1
    if dc['has_energy_data']:
        state_counts[state]['with_data'] += 1

# Load existing state_data and add total counts
with open('state_data.json', 'r') as f:
    state_data = json.load(f)

# State name to abbrev mapping
name_to_abbrev = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA',
    'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'District of Columbia': 'DC', 'Florida': 'FL',
    'Georgia': 'GA', 'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN',
    'Iowa': 'IA', 'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME',
    'Maryland': 'MD', 'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS',
    'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH',
    'New Jersey': 'NJ', 'New Mexico': 'NM', 'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND',
    'Ohio': 'OH', 'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI',
    'South Carolina': 'SC', 'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT',
    'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY'
}

for state_name, counts in state_counts.items():
    abbrev = name_to_abbrev.get(state_name)
    if abbrev and abbrev in state_data:
        state_data[abbrev]['total_dc_count'] = counts['total']
        state_data[abbrev]['dc_with_data'] = counts['with_data']
    elif abbrev:
        # State not in state_data yet - add it
        state_data[abbrev] = {
            'name': state_name,
            'dc_count': counts['with_data'],
            'total_dc_count': counts['total'],
            'dc_with_data': counts['with_data'],
            'capacity_mw': 0,
            'energy_twh': 0,
            'crypto_count': 0,
            'ai_count': 0,
            'crypto_energy_mwh': 0,
            'ai_energy_mwh': 0
        }

with open('state_data.json', 'w') as f:
    json.dump(state_data, f, indent=2)

print(f'Updated state_data.json')
print(f'Total DCs across all states: {sum(c["total"] for c in state_counts.values())}')
