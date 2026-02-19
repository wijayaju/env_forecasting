#!/usr/bin/env python3
"""
Extract data centers with coordinates and look up HUC12 codes
Uses USGS National Map API to get HUC12 watershed IDs
Saves to huc_12.csv
"""

import os
import re
import json
import csv
import requests
import time

def get_huc12(lat, lon):
    """Query USGS API to get HUC12 code for given coordinates"""
    url = "https://hydro.nationalmap.gov/arcgis/rest/services/wbd/MapServer/6/query"
    params = {
        'geometry': f'{lon},{lat}',
        'geometryType': 'esriGeometryPoint',
        'inSR': '4326',
        'spatialRel': 'esriSpatialRelIntersects',
        'outFields': 'huc12,name',
        'returnGeometry': 'false',
        'f': 'json'
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        features = data.get('features', [])
        if features:
            attrs = features[0].get('attributes', {})
            huc12 = attrs.get('huc12', '')
            huc_name = attrs.get('name', '')
            return huc12, huc_name
    except Exception as e:
        print(f"    Error looking up HUC12: {e}")
    
    return '', ''

def extract_datacenters_with_huc12():
    """Extract data center info with coordinates and look up HUC12 codes"""
    
    state_dir = 'html/state'
    all_datacenters = []
    
    # Get all state folders
    states = [d for d in os.listdir(state_dir) if os.path.isdir(os.path.join(state_dir, d))]
    
    print(f"Found {len(states)} state folders")
    print("Extracting data centers with coordinates...\n")
    
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
                continue
            
            # Extract JSON data from __NEXT_DATA__ script tag
            json_match = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.+?)</script>', content)
            
            if not json_match:
                continue
            
            try:
                data = json.loads(json_match.group(1))
                
                # Navigate to the data centers list
                dcs = data.get('props', {}).get('pageProps', {}).get('mapdata', {}).get('dcs', [])
                
                for dc in dcs:
                    props = dc.get('properties', {})
                    geom = dc.get('geometry', {})
                    coords = geom.get('coordinates', [])
                    
                    if len(coords) >= 2:
                        lon, lat = coords[0], coords[1]
                    else:
                        lon, lat = None, None
                    
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
                        'country': country,
                        'latitude': lat,
                        'longitude': lon
                    })
                    
            except json.JSONDecodeError:
                continue
    
    # Remove duplicates based on name + address
    seen = set()
    unique_dcs = []
    for dc in all_datacenters:
        key = (dc['name'], dc['address'])
        if key not in seen:
            seen.add(key)
            unique_dcs.append(dc)
    
    print(f"Found {len(unique_dcs)} unique data centers")
    print("Looking up HUC12 codes (this will take a while)...\n")
    
    # Look up HUC12 for each data center
    for i, dc in enumerate(unique_dcs):
        lat = dc.get('latitude')
        lon = dc.get('longitude')
        
        if lat and lon:
            print(f"[{i+1}/{len(unique_dcs)}] {dc['name'][:50]}...", end=" ")
            huc12, huc_name = get_huc12(lat, lon)
            dc['huc12'] = huc12
            dc['huc12_name'] = huc_name
            print(f"HUC12: {huc12}")
            
            # Be polite to the API
            time.sleep(0.3)
        else:
            dc['huc12'] = ''
            dc['huc12_name'] = ''
            print(f"[{i+1}/{len(unique_dcs)}] {dc['name'][:50]}... No coordinates")
    
    # Save to CSV
    csv_file = 'huc_12.csv'
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(['Name', 'Company', 'Address', 'Postal', 'City', 'State', 'Country', 
                        'Latitude', 'Longitude', 'HUC12', 'HUC12_Name'])
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
                dc.get('latitude', ''),
                dc.get('longitude', ''),
                dc.get('huc12', ''),
                dc.get('huc12_name', '')
            ])
    
    print("\n" + "="*50)
    print(f"COMPLETE")
    print("="*50)
    print(f"Total data centers: {len(unique_dcs)}")
    print(f"Saved to: {csv_file}")

if __name__ == "__main__":
    extract_datacenters_with_huc12()
