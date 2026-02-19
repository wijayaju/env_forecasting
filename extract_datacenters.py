#!/usr/bin/env python3
"""
Extract data center information from city txt files
Saves to data_centers.csv
"""

import os
import re
import json
import csv

def extract_data_centers():
    """Extract data center info from all city txt files and save to CSV"""
    
    state_dir = 'html/state'
    all_datacenters = []
    
    # Get all state folders
    states = [d for d in os.listdir(state_dir) if os.path.isdir(os.path.join(state_dir, d))]
    
    print(f"Found {len(states)} state folders")
    
    for state in sorted(states):
        city_dir = f'{state_dir}/{state}/city'
        
        if not os.path.exists(city_dir):
            continue
        
        cities = [d for d in os.listdir(city_dir) if os.path.isdir(os.path.join(city_dir, d))]
        
        for city in cities:
            city_file = f'{city_dir}/{city}/{city}.txt'
            
            if not os.path.exists(city_file):
                continue
            
            with open(city_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Skip rate-limited files
            if 'Page View Limit Reached' in content:
                print(f"  Skipping {state}/{city} (rate limited)")
                continue
            
            # Extract JSON data from __NEXT_DATA__ script tag
            json_match = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.+?)</script>', content)
            
            if not json_match:
                print(f"  Skipping {state}/{city} (no JSON data)")
                continue
            
            try:
                data = json.loads(json_match.group(1))
                
                # Navigate to the data centers list
                dcs = data.get('props', {}).get('pageProps', {}).get('mapdata', {}).get('dcs', [])
                
                for dc in dcs:
                    props = dc.get('properties', {})
                    
                    name = props.get('name', '')
                    company = props.get('companyname', '')
                    address = props.get('address', '')
                    postal = props.get('postal', '')
                    city_name = props.get('city', '')
                    state_name = props.get('state', '')
                    country = props.get('country', '')
                    
                    all_datacenters.append({
                        'name': name,
                        'company': company,
                        'address': address,
                        'postal': postal,
                        'city': city_name,
                        'state': state_name,
                        'country': country
                    })
                
                if dcs:
                    print(f"  {state}/{city}: {len(dcs)} data centers")
                    
            except json.JSONDecodeError as e:
                print(f"  Error parsing {state}/{city}: {e}")
    
    # Remove duplicates based on name + address
    seen = set()
    unique_dcs = []
    for dc in all_datacenters:
        key = (dc['name'], dc['address'])
        if key not in seen:
            seen.add(key)
            unique_dcs.append(dc)
    
    # Save to CSV
    csv_file = 'data_centers.csv'
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(['Name', 'Company', 'Address', 'Postal', 'City', 'State', 'Country'])
        # Write data
        for dc in unique_dcs:
            writer.writerow([
                dc['name'],
                dc['company'],
                dc['address'],
                dc['postal'],
                dc['city'],
                dc['state'],
                dc['country']
            ])
    
    print("\n" + "="*50)
    print(f"EXTRACTION COMPLETE")
    print("="*50)
    print(f"Total data centers found: {len(all_datacenters)}")
    print(f"Unique data centers: {len(unique_dcs)}")
    print(f"Saved to: {csv_file}")

if __name__ == "__main__":
    extract_data_centers()
