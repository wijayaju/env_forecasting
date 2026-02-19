#!/usr/bin/env python3
"""
Parse city links from state txt files
Loops through html/state/{state}/{state}.txt and extracts city links
Saves them to html/state/{state}/city_links.txt
"""

import re
import os

def parse_city_links():
    """Extract city links from all state txt files"""
    
    state_dir = 'html/state'
    
    # Get all state folders
    states = [d for d in os.listdir(state_dir) if os.path.isdir(os.path.join(state_dir, d))]
    
    print(f"Found {len(states)} state folders")
    
    total_cities = 0
    
    for state in sorted(states):
        state_file = f'{state_dir}/{state}/{state}.txt'
        
        if not os.path.exists(state_file):
            print(f"Skipping {state}: no txt file found")
            continue
        
        # Read the HTML content
        with open(state_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find all href attributes with /usa/{state}/{city}/ pattern
        # Pattern matches href="/usa/state-name/city-name/"
        pattern = rf'href="(/usa/{state}/[^"]+)"'
        matches = re.findall(pattern, content)
        
        # Remove duplicates and filter out quote links and the state itself
        unique_links = set()
        for link in matches:
            # Skip quote links and the state link itself
            if '/quote' in link:
                continue
            # Make sure it's a city link (has something after /usa/state/)
            parts = link.rstrip('/').split('/')
            if len(parts) >= 4:  # ['', 'usa', 'state', 'city']
                unique_links.add(link)
        
        # Convert to full URLs and sort
        full_urls = sorted([f"https://www.datacentermap.com{link}" for link in unique_links])
        
        # Save to city_links.txt in the state folder
        output_file = f'{state_dir}/{state}/city_links.txt'
        with open(output_file, 'w', encoding='utf-8') as f:
            for url in full_urls:
                f.write(url + '\n')
        
        print(f"{state}: {len(full_urls)} cities")
        total_cities += len(full_urls)
    
    print("\n" + "="*50)
    print(f"COMPLETE: Found {total_cities} total city links across {len(states)} states")
    print("="*50)

if __name__ == "__main__":
    parse_city_links()
