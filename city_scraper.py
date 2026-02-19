#!/usr/bin/env python3
"""
City Web Scraper
Loops through all city_links.txt files in each state folder
and scrapes each city page, saving to html/state/{state}/city/{city}/{city}.txt
"""

import requests
import os
import time

# Delay between requests in seconds (increase to avoid rate limiting)
REQUEST_DELAY = 5

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

def scrape_cities():
    """Scrape all city pages from city_links.txt files"""
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    state_dir = 'html/state'
    
    # Get all state folders
    states = [d for d in os.listdir(state_dir) if os.path.isdir(os.path.join(state_dir, d))]
    
    print(f"Found {len(states)} state folders")
    print(f"Using {REQUEST_DELAY} second delay between requests")
    
    total_success = 0
    total_skipped = 0
    total_errors = 0
    
    for state in sorted(states):
        city_links_file = f'{state_dir}/{state}/city_links.txt'
        
        if not os.path.exists(city_links_file):
            print(f"Skipping {state}: no city_links.txt found")
            continue
        
        # Read city URLs
        with open(city_links_file, 'r', encoding='utf-8') as f:
            urls = [line.strip() for line in f if line.strip()]
        
        if not urls:
            print(f"Skipping {state}: no city links")
            continue
        
        print(f"\n{state.upper()}: {len(urls)} cities")
        
        for url in urls:
            # Extract city name from URL (e.g., "abilene" from "/usa/texas/abilene/")
            parts = url.rstrip('/').split('/')
            city_name = parts[-1]
            
            # Check if we already have valid content
            city_dir = f'{state_dir}/{state}/city/{city_name}'
            output_path = f'{city_dir}/{city_name}.txt'
            
            if has_valid_content(output_path):
                print(f"  Skipping: {city_name} (already scraped)")
                total_skipped += 1
                continue
            
            print(f"  Scraping: {city_name}...", end=" ")
            
            try:
                response = requests.get(url, headers=headers, timeout=30)
                response.raise_for_status()
                
                # Check if we got rate limited
                if 'Page View Limit Reached' in response.text:
                    print("RATE LIMITED - stopping")
                    print("\nRate limit hit! Wait a while and run again.")
                    return
                
                # Create directory html/state/{state}/city/{city}/
                os.makedirs(city_dir, exist_ok=True)
                
                # Save HTML as {city}.txt
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(response.text)
                
                print(f"OK ({len(response.text)} chars)")
                total_success += 1
                
                # Longer delay between requests to avoid rate limiting
                time.sleep(REQUEST_DELAY)
                
            except requests.RequestException as e:
                print(f"ERROR: {e}")
                total_errors += 1
    
    print("\n" + "="*50)
    print("SCRAPING COMPLETE")
    print("="*50)
    print(f"Successful: {total_success}")
    print(f"Skipped (already done): {total_skipped}")
    print(f"Errors: {total_errors}")
    print(f"Files saved to: html/state/{{state}}/city/{{city}}/{{city}}.txt")

if __name__ == "__main__":
    scrape_cities()
