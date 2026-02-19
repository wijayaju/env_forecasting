import requests
import os

def scrape_usa_datacenters():
    """Scrape the USA datacenters page from datacentermap.com and save as txt."""
    url = "https://www.datacentermap.com/usa/"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        # Create html folder if it doesn't exist
        os.makedirs('html', exist_ok=True)
        
        # Save the HTML content as txt file in html folder
        with open('html/usa.txt', 'w', encoding='utf-8') as f:
            f.write(response.text)
        
        print(f"Successfully saved HTML to html/usa.txt")
        print(f"File size: {len(response.text)} characters")
        
    except requests.RequestException as e:
        print(f"Error fetching the page: {e}")

if __name__ == "__main__":
    scrape_usa_datacenters()
