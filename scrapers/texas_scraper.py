#!/usr/bin/env python3
"""
Simple Web Scraper for Texas Data Centers
Saves HTML and extracts data center listings from datacentermap.com
"""

import requests
from bs4 import BeautifulSoup
import os
import json
from datetime import datetime

URL = "https://www.datacentermap.com/usa/texas/"

def scrape_texas_datacenters():
    """Fetch and save everything from the Texas datacenter page"""
    
    print(f"Fetching: {URL}")
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    }
    
    response = requests.get(URL, headers=headers)
    
    if response.status_code != 200:
        print(f"Error: HTTP {response.status_code}")
        return
    
    print(f"Success! Status: {response.status_code}")
    
    # Save raw HTML
    html_filename = "texas_datacenters.html"
    with open(html_filename, 'w', encoding='utf-8') as f:
        f.write(response.text)
    print(f"Saved raw HTML to: {html_filename}")
    
    # Parse the HTML
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Save page title
    title = soup.title.string if soup.title else "No title"
    print(f"Page title: {title}")
    
    # Extract all text content
    text_filename = "texas_datacenters.txt"
    text_content = soup.get_text(separator='\n', strip=True)
    with open(text_filename, 'w', encoding='utf-8') as f:
        f.write(f"URL: {URL}\n")
        f.write(f"Scraped: {datetime.now()}\n")
        f.write(f"Title: {title}\n")
        f.write("="*50 + "\n\n")
        f.write(text_content)
    print(f"Saved text content to: {text_filename}")
    
    # Extract all links
    links = []
    for a in soup.find_all('a', href=True):
        link_text = a.get_text(strip=True)
        link_url = a['href']
        if link_text or link_url:
            links.append({
                'text': link_text,
                'url': link_url
            })
    
    # Extract data center listings (look for common patterns)
    datacenters = []
    
    # Try to find datacenter entries - common patterns on this site
    # Look for list items, table rows, or divs with datacenter info
    
    # Method 1: Find all city/location links
    for a in soup.find_all('a', href=True):
        href = a['href']
        text = a.get_text(strip=True)
        if '/usa/texas/' in href and text and len(text) > 2:
            if href != '/usa/texas/' and 'datacentermap.com' not in text.lower():
                datacenters.append({
                    'name': text,
                    'url': href if href.startswith('http') else f"https://www.datacentermap.com{href}"
                })
    
    # Method 2: Look for structured content
    # Find tables
    tables = soup.find_all('table')
    for i, table in enumerate(tables):
        rows = table.find_all('tr')
        for row in rows:
            cells = row.find_all(['td', 'th'])
            row_data = [cell.get_text(strip=True) for cell in cells]
            if row_data and any(row_data):
                datacenters.append({'table_row': row_data})
    
    # Find lists
    lists = soup.find_all(['ul', 'ol'])
    for lst in lists:
        items = lst.find_all('li')
        for item in items:
            item_text = item.get_text(strip=True)
            item_link = item.find('a')
            if item_text and len(item_text) > 3:
                entry = {'list_item': item_text}
                if item_link and item_link.get('href'):
                    entry['url'] = item_link['href']
                datacenters.append(entry)
    
    # Save extracted data as JSON
    json_filename = "texas_datacenters.json"
    data = {
        'url': URL,
        'scraped_at': str(datetime.now()),
        'title': title,
        'total_links': len(links),
        'links': links,
        'datacenters': datacenters
    }
    
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved structured data to: {json_filename}")
    
    # Summary
    print("\n" + "="*50)
    print("SCRAPING COMPLETE")
    print("="*50)
    print(f"Files saved:")
    print(f"  - {html_filename} (raw HTML)")
    print(f"  - {text_filename} (text content)")
    print(f"  - {json_filename} (structured data)")
    print(f"\nExtracted:")
    print(f"  - {len(links)} links")
    print(f"  - {len(datacenters)} potential datacenter entries")

if __name__ == "__main__":
    scrape_texas_datacenters()
