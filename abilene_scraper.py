#!/usr/bin/env python3
"""
Simple Web Scraper for Abilene Data Centers
Saves HTML and extracts data center listings from datacentermap.com
"""

import requests
from bs4 import BeautifulSoup
import os
import json
from datetime import datetime

URL = "https://www.datacentermap.com/usa/texas/abilene/"

def scrape_abilene_datacenters():
    """Fetch and save everything from the Abilene datacenter page"""
    
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
    html_filename = "abilene_datacenters.html"
    with open(html_filename, 'w', encoding='utf-8') as f:
        f.write(response.text)
    print(f"Saved raw HTML to: {html_filename}")
    
    # Parse the HTML
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Save page title
    title = soup.title.string if soup.title else "No title"
    print(f"Page title: {title}")
    
    # Extract all text content
    text_filename = "abilene_datacenters.txt"
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
    
    # Extract data center listings
    datacenters = []
    
    # Look for datacenter cards/entries - typically in cards or table rows
    cards = soup.find_all('div', class_='card')
    for card in cards:
        dc_info = {}
        # Get title/name
        title_elem = card.find(['h2', 'h3', 'h4', 'a'])
        if title_elem:
            dc_info['name'] = title_elem.get_text(strip=True)
            if title_elem.name == 'a' and title_elem.get('href'):
                dc_info['url'] = title_elem['href']
        # Get description/details
        desc = card.get_text(separator=' | ', strip=True)
        dc_info['details'] = desc
        if dc_info:
            datacenters.append(dc_info)
    
    # Look for table data
    tables = soup.find_all('table')
    for table in tables:
        rows = table.find_all('tr')
        for row in rows:
            cells = row.find_all(['td', 'th'])
            row_data = [cell.get_text(strip=True) for cell in cells]
            # Get links from cells
            row_links = []
            for cell in cells:
                link = cell.find('a')
                if link and link.get('href'):
                    row_links.append({
                        'text': link.get_text(strip=True),
                        'url': link['href']
                    })
            if row_data and any(row_data):
                datacenters.append({
                    'table_row': row_data,
                    'links': row_links
                })
    
    # Look for list items with datacenter info
    for li in soup.find_all('li'):
        li_text = li.get_text(strip=True)
        li_link = li.find('a')
        if li_text and len(li_text) > 5:
            entry = {'list_item': li_text}
            if li_link and li_link.get('href'):
                entry['url'] = li_link['href']
            # Check if it looks like a datacenter (not nav)
            if '/usa/texas/' in str(li_link) if li_link else False:
                datacenters.append(entry)
    
    # Extract embedded JSON data (Next.js pages often have this)
    script_data = soup.find('script', id='__NEXT_DATA__')
    embedded_data = None
    if script_data:
        try:
            embedded_data = json.loads(script_data.string)
            print("Found embedded Next.js data!")
        except:
            pass
    
    # Save extracted data as JSON
    json_filename = "abilene_datacenters.json"
    data = {
        'url': URL,
        'scraped_at': str(datetime.now()),
        'title': title,
        'total_links': len(links),
        'links': links,
        'datacenters': datacenters,
    }
    
    # Include embedded data if found
    if embedded_data:
        data['embedded_nextjs_data'] = embedded_data
    
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
    print(f"  - {len(datacenters)} datacenter entries")
    if embedded_data:
        print(f"  - Embedded Next.js data included")

if __name__ == "__main__":
    scrape_abilene_datacenters()
