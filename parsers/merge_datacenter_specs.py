#!/usr/bin/env python3
"""
Merge Data Centers with Specs
Combines data_centers.csv with specs data from html/state/{state}/city/{city}/dc/{dc}/specs.txt
Creates data_centers_complete.csv with all original data plus Capacity and Operational_Date
"""

import os
import re
import json
import csv

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
    
    # Extract Fully Built-Out Power (capacity)
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
        json_match = re.search(r'"power"\s*:\s*"([^"]+)"', content)
        if json_match:
            power_val = json_match.group(1).strip()
            if power_val and power_val not in ['', '-', 'N/A', 'null']:
                capacity = power_val
    
    # Extract Year Operational
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

def build_specs_lookup():
    """Build a lookup dict mapping dc_link -> (capacity, operational_date)"""
    
    state_dir = '../data/raw/html/state'
    specs_lookup = {}
    
    states = [d for d in os.listdir(state_dir) if os.path.isdir(os.path.join(state_dir, d))]
    
    for state in states:
        city_dir = f'{state_dir}/{state}/city'
        
        if not os.path.exists(city_dir):
            continue
        
        cities = [d for d in os.listdir(city_dir) if os.path.isdir(os.path.join(city_dir, d))]
        
        for city in cities:
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
                    continue
                
                # Store by dc_link
                specs_lookup[dc] = (capacity, operational_date)
    
    return specs_lookup

def extract_dc_with_link(city_file_path):
    """Extract data center info including the link field from a city txt file"""
    if not os.path.exists(city_file_path):
        return []
    
    with open(city_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if 'Page View Limit Reached' in content:
        return []
    
    json_match = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.+?)</script>', content)
    
    if not json_match:
        return []
    
    try:
        data = json.loads(json_match.group(1))
        dcs = data.get('props', {}).get('pageProps', {}).get('mapdata', {}).get('dcs', [])
        
        results = []
        for dc in dcs:
            props = dc.get('properties', {})
            
            results.append({
                'name': props.get('name', ''),
                'company': props.get('companyname', ''),
                'address': props.get('address', ''),
                'postal': props.get('postal', ''),
                'city': props.get('city', ''),
                'state': props.get('state', ''),
                'country': props.get('country', ''),
                'link': props.get('link', '')  # This is the URL-safe dc name
            })
        
        return results
        
    except json.JSONDecodeError:
        return []

def merge_data():
    """Extract all data centers with their specs and save to data_centers_complete.csv"""
    
    print("Building specs lookup from scraped data...")
    specs_lookup = build_specs_lookup()
    print(f"Found specs for {len(specs_lookup)} data centers")
    
    state_dir = '../data/raw/html/state'
    all_datacenters = []
    
    states = [d for d in os.listdir(state_dir) if os.path.isdir(os.path.join(state_dir, d))]
    
    print(f"\nExtracting data centers from {len(states)} states...")
    
    for state in sorted(states):
        city_dir = f'{state_dir}/{state}/city'
        
        if not os.path.exists(city_dir):
            continue
        
        cities = [d for d in os.listdir(city_dir) if os.path.isdir(os.path.join(city_dir, d))]
        
        for city in cities:
            city_file = f'{city_dir}/{city}/{city}.txt'
            
            dcs = extract_dc_with_link(city_file)
            
            for dc in dcs:
                link = dc['link']
                
                # Look up specs
                if link in specs_lookup:
                    capacity, operational_date = specs_lookup[link]
                else:
                    capacity, operational_date = "NA", "NA"
                
                all_datacenters.append({
                    'name': dc['name'],
                    'company': dc['company'],
                    'address': dc['address'],
                    'postal': dc['postal'],
                    'city': dc['city'],
                    'state': dc['state'],
                    'country': dc['country'],
                    'capacity': capacity,
                    'operational_date': operational_date
                })
    
    # Remove duplicates based on name + address
    seen = set()
    unique_dcs = []
    for dc in all_datacenters:
        key = (dc['name'], dc['address'])
        if key not in seen:
            seen.add(key)
            unique_dcs.append(dc)
    
    # Save to CSV
    csv_file = 'data_centers_complete.csv'
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(['Name', 'Company', 'Address', 'Postal', 'City', 'State', 'Country', 'Capacity', 'Operational_Date'])
        # Write data
        for dc in unique_dcs:
            writer.writerow([
                dc['name'],
                dc['company'],
                dc['address'],
                dc['postal'],
                dc['city'],
                dc['state'],
                dc['country'],
                dc['capacity'],
                dc['operational_date']
            ])
    
    # Count stats
    total_with_capacity = sum(1 for dc in unique_dcs if dc['capacity'] != "NA")
    total_with_date = sum(1 for dc in unique_dcs if dc['operational_date'] != "NA")
    
    print("\n" + "="*50)
    print("MERGE COMPLETE")
    print("="*50)
    print(f"Total unique data centers: {len(unique_dcs)}")
    print(f"With capacity data: {total_with_capacity}")
    print(f"With operational date: {total_with_date}")
    print(f"Saved to: {csv_file}")

if __name__ == "__main__":
    merge_data()
