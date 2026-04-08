#!/usr/bin/env python3
"""
Parse all specs.txt files and extract data center info to CSV.
"""

import os
import re
import json
import csv
from glob import glob

def extract_dc_info(file_path):
    """Extract data center info from a specs.txt file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Skip rate-limited files
        if 'Page View Limit Reached' in content:
            return None
        
        # Extract JSON from __NEXT_DATA__
        match = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.+?)</script>', content)
        if not match:
            return None
        
        data = json.loads(match.group(1))
        dc = data.get('props', {}).get('pageProps', {}).get('dc', {})
        
        if not dc:
            return None
        
        # Extract fields
        meta_building = dc.get('meta_building') or {}
        meta_capacity = dc.get('meta_capacity') or {}
        
        return {
            'data_center_id': dc.get('link', ''),
            'data_center_name': dc.get('name', ''),
            'state': dc.get('state', ''),
            'city': dc.get('city', ''),
            'latitude': dc.get('latitude', ''),
            'longitude': dc.get('longitude', ''),
            'year_operational': meta_building.get('year_operational', ''),
            'capacity_mw': meta_capacity.get('mw_builtout', ''),
            'capacity_sqft': meta_capacity.get('whitespace_builtout', ''),
        }
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def main():
    # Find all specs.txt files
    specs_files = glob('html/state/*/city/*/dc/*/specs.txt')
    print(f"Found {len(specs_files)} specs files")
    
    results = []
    errors = 0
    
    for file_path in specs_files:
        info = extract_dc_info(file_path)
        if info:
            results.append(info)
        else:
            errors += 1
    
    print(f"Extracted {len(results)} data centers ({errors} errors/skipped)")
    
    # Write to CSV
    output_file = 'data/datacenter_specs.csv'
    os.makedirs('data', exist_ok=True)
    
    fieldnames = [
        'data_center_id',
        'data_center_name',
        'state',
        'city',
        'latitude',
        'longitude',
        'year_operational',
        'capacity_mw',
        'capacity_sqft',
    ]
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Saved to {output_file}")
    
    # Stats
    with_year = sum(1 for r in results if r['year_operational'] and r['year_operational'] not in ('', '0', 0))
    with_capacity = sum(1 for r in results if r['capacity_mw'] and r['capacity_mw'] not in ('', '0', 0))
    print(f"\nStats:")
    print(f"  With operational year: {with_year}")
    print(f"  With capacity (MW): {with_capacity}")

if __name__ == '__main__':
    main()
