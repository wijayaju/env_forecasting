#!/usr/bin/env python3
"""
Parse state links from usa.txt and save as full URLs to state_links.txt
"""

import re

def parse_state_links():
    """Extract /usa/ links from usa.txt and save as full URLs"""
    
    # Read the HTML content
    with open('html/usa.txt', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find all href attributes with /usa/ pattern
    # Pattern matches href="/usa/state-name/"
    pattern = r'href="(/usa/[^"]+)"'
    matches = re.findall(pattern, content)
    
    # Remove duplicates and filter out just "/usa/" itself
    unique_links = set()
    for link in matches:
        # Skip the base /usa/ link
        if link != '/usa/' and link.startswith('/usa/'):
            unique_links.add(link)
    
    # Convert to full URLs and sort
    full_urls = sorted([f"https://www.datacentermap.com{link}" for link in unique_links])
    
    # Save to html/state_links.txt
    with open('html/state_links.txt', 'w', encoding='utf-8') as f:
        for url in full_urls:
            f.write(url + '\n')
    
    print(f"Found {len(full_urls)} unique state links")
    print(f"Saved to html/state_links.txt")
    
    # Print the links
    print("\nLinks:")
    for url in full_urls:
        print(f"  {url}")

if __name__ == "__main__":
    parse_state_links()
