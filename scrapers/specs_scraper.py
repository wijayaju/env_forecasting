#!/usr/bin/env python3
"""
Data Center Specs Scraper
Loops through all city txt files, extracts data center URLs,
and scrapes each data center's /specs/ page.
Saves to data/raw/html/state/{state}/city/{city}/dc/{dc-name}/specs.txt

Implements rate limiting strategies:
- Configurable delay between requests
- Skip already-scraped content
- Detect and stop on rate limit
- Resumable (run multiple times)
- Uses requests Session for cookie persistence
"""

import requests
import os
import time
import re
import json
import random

# Delay between requests in seconds (increase to avoid rate limiting)
REQUEST_DELAY = 3
# Add random jitter to delay (0 to this value in seconds)
JITTER_MAX = 1
# Wait time in minutes when rate limited before retrying
RATE_LIMIT_WAIT_MINUTES = 0.25
# Maximum retries after rate limit before giving up
MAX_RATE_LIMIT_RETRIES = 10

# Rotate user agents to reduce detection
USER_AGENTS = [
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
]

def is_rate_limited(file_path):
    """Check if a file contains rate limit message"""
    if not os.path.exists(file_path):
        return False
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    return 'Page View Limit Reached' in content

def has_valid_content(file_path):
    """Check if file exists and has valid (non-rate-limited) content"""
    if not os.path.exists(file_path):
        return False
    return not is_rate_limited(file_path)

def extract_dc_urls_from_city(city_file_path):
    """Extract data center URLs from a city txt file"""
    if not os.path.exists(city_file_path):
        return []
    
    with open(city_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Skip rate-limited files
    if 'Page View Limit Reached' in content:
        return []
    
    # Extract JSON data from __NEXT_DATA__ script tag
    json_match = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.+?)</script>', content)
    
    if not json_match:
        return []
    
    try:
        data = json.loads(json_match.group(1))
        dcs = data.get('props', {}).get('pageProps', {}).get('mapdata', {}).get('dcs', [])
        
        urls = []
        for dc in dcs:
            props = dc.get('properties', {})
            url = props.get('url', '')
            name = props.get('name', '')
            link = props.get('link', '')  # This is the URL-safe dc name
            
            if url and name:
                urls.append({
                    'url': url,
                    'name': name,
                    'link': link,
                    'specs_url': f"https://www.datacentermap.com{url}specs/"
                })
        
        return urls
        
    except json.JSONDecodeError:
        return []

def scrape_specs():
    """Scrape all data center specs pages"""
    
    # Use a session to maintain cookies across requests
    session = requests.Session()
    
    # Set initial headers
    session.headers.update({
        'User-Agent': random.choice(USER_AGENTS),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    })
    
    state_dir = '../data/raw/html/state'
    
    # Get all state folders
    states = [d for d in os.listdir(state_dir) if os.path.isdir(os.path.join(state_dir, d))]
    
    print(f"Found {len(states)} state folders")
    print(f"Using {REQUEST_DELAY}s delay (+0-{JITTER_MAX}s jitter) between requests")
    
    total_success = 0
    total_skipped = 0
    total_errors = 0
    total_dcs = 0
    
    for state in sorted(states):
        city_dir = f'{state_dir}/{state}/city'
        
        if not os.path.exists(city_dir):
            continue
        
        cities = [d for d in os.listdir(city_dir) if os.path.isdir(os.path.join(city_dir, d))]
        
        state_dcs = 0
        
        for city in sorted(cities):
            city_file = f'{city_dir}/{city}/{city}.txt'
            
            # Get all DC URLs from this city
            dc_urls = extract_dc_urls_from_city(city_file)
            
            if not dc_urls:
                continue
            
            for dc_info in dc_urls:
                total_dcs += 1
                state_dcs += 1
                
                dc_link = dc_info['link']
                specs_url = dc_info['specs_url']
                dc_name = dc_info['name']
                
                # Create directory html/state/{state}/city/{city}/dc/{dc-link}/
                dc_dir = f'{city_dir}/{city}/dc/{dc_link}'
                output_path = f'{dc_dir}/specs.txt'
                
                # Check if we already have valid content
                if has_valid_content(output_path):
                    total_skipped += 1
                    continue
                
                print(f"  Scraping: {state}/{city}/{dc_link}...", end=" ", flush=True)
                
                # Retry loop for rate limiting
                retries = 0
                while retries <= MAX_RATE_LIMIT_RETRIES:
                    try:
                        # Rotate user agent occasionally
                        if total_success % 50 == 0:
                            session.headers.update({'User-Agent': random.choice(USER_AGENTS)})
                        
                        response = session.get(specs_url, timeout=30)
                        response.raise_for_status()
                        
                        # Check if we got rate limited or Vercel security checkpoint
                        if 'Page View Limit Reached' in response.text or 'Vercel Security Checkpoint' in response.text:
                            retries += 1
                            if retries > MAX_RATE_LIMIT_RETRIES:
                                print(f"RATE LIMITED - max retries ({MAX_RATE_LIMIT_RETRIES}) exceeded, stopping")
                                print("\n" + "="*50)
                                print(f"Progress: {total_success} scraped, {total_skipped} skipped, {total_errors} errors")
                                print("="*50)
                                return
                            
                            wait_time = RATE_LIMIT_WAIT_MINUTES * 60
                            print(f"RATE LIMITED - waiting {RATE_LIMIT_WAIT_MINUTES} minutes (retry {retries}/{MAX_RATE_LIMIT_RETRIES})...")
                            time.sleep(wait_time)
                            # Rotate user agent after rate limit
                            session.headers.update({'User-Agent': random.choice(USER_AGENTS)})
                            continue
                        
                        # Success - break out of retry loop
                        break
                        
                    except requests.RequestException as e:
                        print(f"ERROR: {e}")
                        total_errors += 1
                        break
                else:
                    # Max retries exceeded in while loop
                    continue
                
                # Only save if we got a valid response (not rate limited)
                if 'Page View Limit Reached' not in response.text and 'Vercel Security Checkpoint' not in response.text:
                    # Create directory
                    os.makedirs(dc_dir, exist_ok=True)
                    
                    # Save HTML as specs.txt
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(response.text)
                    
                    print(f"OK ({len(response.text)} chars)")
                    total_success += 1
                    
                    # Delay with jitter between requests
                    delay = REQUEST_DELAY + random.uniform(0, JITTER_MAX)
                    time.sleep(delay)
        
        if state_dcs > 0:
            print(f"\n{state.upper()}: {state_dcs} data centers processed\n")
    
    print("\n" + "="*50)
    print("SCRAPING COMPLETE")
    print("="*50)
    print(f"Total data centers found: {total_dcs}")
    print(f"Successful: {total_success}")
    print(f"Skipped (already done): {total_skipped}")
    print(f"Errors: {total_errors}")
    print(f"Files saved to: data/raw/html/state/{{state}}/city/{{city}}/dc/{{dc}}/specs.txt")

if __name__ == "__main__":
    scrape_specs()
