#!/usr/bin/env python3
"""
Extract specs from data center specs.txt files
Parses capacity (power) and operational date from spec pages
Outputs to specs_data.csv
"""

import os
import re
import csv
import json

def extract_specs_from_html(content):
    """Extract capacity and operational date from specs page HTML"""
    
    capacity = "NA"
    operational_date = "NA"
    
    # Check if it's a "No data supplied" page
    if "No data supplied by" in content:
        return capacity, operational_date
    
    # Check for rate-limited content or security checkpoint
    if 'Page View Limit Reached' in content or 'Vercel Security Checkpoint' in content:
        return None, None  # Signal to skip this file
    
    # Try to extract from __NEXT_DATA__ JSON first (more reliable)
    json_match = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.+?)</script>', content)
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            dc = data.get('props', {}).get('pageProps', {}).get('dc', {})
            
            # Extract capacity from meta_power.totalmw
            meta_power = dc.get('meta_power', {}) or {}
            power = meta_power.get('totalmw', '')
            if power and str(power) not in ['', '-', 'N/A', 'null', 'None', '0']:
                capacity = f"{power} MW"
            
            # Extract year from meta_building.year_operational
            meta_building = dc.get('meta_building', {}) or {}
            year = meta_building.get('year_operational', '')
            if year and str(year) not in ['', '-', 'N/A', 'null', 'None', '0']:
                operational_date = str(year)
            
            return capacity, operational_date
                
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
    
    # Fallback: try HTML table patterns (less reliable)
    
    # Extract Fully Built-Out Power (capacity)
    # Pattern: Fully Built-Out Power</td><td ...>6.8 MW</td>
    # Or in table format: | Fully Built-Out Power | 6.8 MW |
    power_patterns = [
        r'Fully Built-Out Power[^<]*</td>[^<]*<td[^>]*>([^<]+)</td>',
        r'Fully Built-Out Power[^|]*\|[^|]*\|\s*([^\|<]+)',
        r'"Fully Built-Out Power"[^"]*"([^"]+)"',
        r'>Fully Built-Out Power</[^>]+>[^>]*>([^<]+)<',
    ]
    
    for pattern in power_patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            power_val = match.group(1).strip()
            if power_val and power_val not in ['', '-', 'N/A']:
                capacity = power_val
                break
    
    # Also try to find power in JSON data
    if capacity == "NA":
        # Look for power in JSON
        json_match = re.search(r'"power"\s*:\s*"([^"]+)"', content)
        if json_match:
            power_val = json_match.group(1).strip()
            if power_val and power_val not in ['', '-', 'N/A', 'null']:
                capacity = power_val
    
    # Extract Year Operational
    # Pattern: Year Operational</td><td ...>2011</td>
    year_patterns = [
        r'Year Operational[^<]*</td>[^<]*<td[^>]*>([^<]+)</td>',
        r'Year Operational[^|]*\|[^|]*\|\s*([^\|<]+)',
        r'"Year Operational"[^"]*"([^"]+)"',
        r'>Year Operational</[^>]+>[^>]*>([^<]+)<',
        r'Year Operational[:\s]+(\d{4})',
    ]
    
    for pattern in year_patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            year_val = match.group(1).strip()
            if year_val and year_val not in ['', '-', 'N/A']:
                operational_date = year_val
                break
    
    # Also try to find year in JSON data
    if operational_date == "NA":
        json_match = re.search(r'"yearOperational"\s*:\s*"?(\d{4})"?', content)
        if json_match:
            operational_date = json_match.group(1)
    
    return capacity, operational_date

def extract_all_specs():
    """Extract specs from all data center spec files"""
    
    state_dir = '../data/raw/html/state'
    all_specs = []
    
    # Get all state folders
    states = [d for d in os.listdir(state_dir) if os.path.isdir(os.path.join(state_dir, d))]
    
    print(f"Found {len(states)} state folders")
    
    total_found = 0
    total_with_capacity = 0
    total_with_date = 0
    total_skipped = 0
    
    for state in sorted(states):
        city_dir = f'{state_dir}/{state}/city'
        
        if not os.path.exists(city_dir):
            continue
        
        cities = [d for d in os.listdir(city_dir) if os.path.isdir(os.path.join(city_dir, d))]
        
        for city in sorted(cities):
            dc_dir = f'{city_dir}/{city}/dc'
            
            if not os.path.exists(dc_dir):
                continue
            
            dcs = [d for d in os.listdir(dc_dir) if os.path.isdir(os.path.join(dc_dir, d))]
            
            for dc in dcs:
                specs_file = f'{dc_dir}/{dc}/specs.txt'
                
                if not os.path.exists(specs_file):
                    continue
                
                with open(specs_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                capacity, operational_date = extract_specs_from_html(content)
                
                # Skip rate-limited files
                if capacity is None:
                    total_skipped += 1
                    continue
                
                total_found += 1
                if capacity != "NA":
                    total_with_capacity += 1
                if operational_date != "NA":
                    total_with_date += 1
                
                all_specs.append({
                    'dc_link': dc,
                    'state': state,
                    'city': city,
                    'capacity': capacity,
                    'operational_date': operational_date
                })
    
    # Save to CSV
    csv_file = 'specs_data.csv'
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['DC_Link', 'State', 'City', 'Capacity', 'Operational_Date'])
        for spec in all_specs:
            writer.writerow([
                spec['dc_link'],
                spec['state'],
                spec['city'],
                spec['capacity'],
                spec['operational_date']
            ])
    
    print("\n" + "="*50)
    print("EXTRACTION COMPLETE")
    print("="*50)
    print(f"Total specs files processed: {total_found}")
    print(f"With capacity data: {total_with_capacity}")
    print(f"With operational date: {total_with_date}")
    print(f"Skipped (rate limited): {total_skipped}")
    print(f"Saved to: {csv_file}")

if __name__ == "__main__":
    extract_all_specs()
