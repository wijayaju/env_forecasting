#!/usr/bin/env python3
"""
State Web Scraper
Loops through state_links.txt and scrapes each state page,
saving the HTML to html/state/{state}/{state}.txt
"""

import requests
import os
import time

def scrape_states():
    """Scrape all state pages from state_links.txt"""
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    # Read the state links
    with open('html/state_links.txt', 'r', encoding='utf-8') as f:
        urls = [line.strip() for line in f if line.strip()]
    
    print(f"Found {len(urls)} URLs to scrape")
    
    success_count = 0
    error_count = 0
    
    for url in urls:
        # Extract state name from URL (e.g., "alabama" from "/usa/alabama/")
        # Skip non-state URLs like /usa/quote/
        parts = url.rstrip('/').split('/')
        state_name = parts[-1]
        
        if state_name == 'quote':
            print(f"Skipping non-state URL: {url}")
            continue
        
        print(f"Scraping: {state_name}...", end=" ")
        
        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Create directory html/state/{state}/
            state_dir = f'html/state/{state_name}'
            os.makedirs(state_dir, exist_ok=True)
            
            # Save HTML as {state}.txt
            output_path = f'{state_dir}/{state_name}.txt'
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(response.text)
            
            print(f"OK ({len(response.text)} chars)")
            success_count += 1
            
            # Be polite - add a small delay between requests
            time.sleep(0.5)
            
        except requests.RequestException as e:
            print(f"ERROR: {e}")
            error_count += 1
    
    print("\n" + "="*50)
    print("SCRAPING COMPLETE")
    print("="*50)
    print(f"Successful: {success_count}")
    print(f"Errors: {error_count}")
    print(f"Files saved to: html/state/{{state}}/{{state}}.txt")

if __name__ == "__main__":
    scrape_states()
