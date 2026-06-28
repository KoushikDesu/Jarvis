import urllib.request
import urllib.parse
import re
import json
import socket
from bs4 import BeautifulSoup

# Standard user agent to avoid bot detection blocks
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

def is_connected():
    """Check if there is an active internet connection."""
    try:
        # Check connection to a reliable DNS server (Cloudflare DNS)
        socket.setdefaulttimeout(2)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(("1.1.1.1", 53))
        return True
    except socket.error:
        return False

def search_google(query, num_results=5):
    """
    Search Google and extract search result titles, links, and snippets.
    Does not require any API keys.
    """
    if not is_connected():
        print("[Offline] Internet connection unavailable. Skipping web search.")
        return []

    try:
        encoded_query = urllib.parse.quote(query)
        url = f"https://www.google.com/search?q={encoded_query}&num={num_results * 2}"
        
        req = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
        with urllib.request.urlopen(req, timeout=5) as response:
            html = response.read()
            
        soup = BeautifulSoup(html, 'html.parser')
        results = []
        
        # Google search results are typically structured in divs with class 'g'
        search_divs = soup.find_all('div', class_='g')
        
        for div in search_divs:
            title_elem = div.find('h3')
            link_elem = div.find('a')
            
            # Find snippet elements
            # Google snippets are usually contained in divs with classes like 'VwiC3b' or similar text containers
            snippet_elem = None
            # Search for typical snippet classes or common structural elements
            snippet_div = div.find('div', class_=re.compile(r'(VwiC3b|yXM1m|lEB1nd|MUFPAc)'))
            if snippet_div:
                snippet_elem = snippet_div
            else:
                # Fallback: look for spans or divs inside the main block containing substantial text
                text_containers = div.find_all(['div', 'span'])
                for container in text_containers:
                    text = container.get_text().strip()
                    if len(text) > 40 and not text.startswith("http") and title_elem and title_elem.get_text() not in text:
                        snippet_elem = container
                        break
            
            if title_elem and link_elem:
                title = title_elem.get_text().strip()
                link = link_elem.get('href')
                
                # Filter out google internal links
                if link and link.startswith('/url?q='):
                    link = urllib.parse.parse_qs(urllib.parse.urlparse(link).query).get('q', [None])[0]
                
                if link and (link.startswith('http://') or link.startswith('https://')):
                    snippet = snippet_elem.get_text().strip() if snippet_elem else "No description available."
                    results.append({
                        "title": title,
                        "link": link,
                        "snippet": snippet
                    })
                    
            if len(results) >= num_results:
                break
                
        return results
    except Exception as e:
        print(f"[Search Error] Google search scraping failed: {e}. Attempting fallback...")
        return search_duckduckgo(query, num_results)

def search_duckduckgo(query, num_results=5):
    """Fallback search using DuckDuckGo HTML/lite interface."""
    if not is_connected():
        return []
        
    try:
        # We query the DDG lite version because it is simple to scrape and doesn't run complex JavaScript
        encoded_query = urllib.parse.quote(query)
        url = f"https://lite.duckduckgo.com/lite/"
        data = urllib.parse.urlencode({'q': query}).encode('utf-8')
        
        req = urllib.request.Request(url, data=data, headers={'User-Agent': USER_AGENT})
        with urllib.request.urlopen(req, timeout=5) as response:
            html = response.read()
            
        soup = BeautifulSoup(html, 'html.parser')
        results = []
        
        # DuckDuckGo HTML Lite table structure
        rows = soup.find_all('td', class_='result-snippet')
        for row in rows:
            # Find the previous result link row
            prev_row = row.parent.find_previous_sibling('tr')
            if prev_row:
                link_elem = prev_row.find('a', class_='result-link')
                if link_elem:
                    title = link_elem.get_text().strip()
                    link = link_elem.get('href')
                    # Clean DDG redirect links
                    if link and link.startswith('//duckduckgo.com/l/?uddg='):
                        url_param = re.search(r'uddg=(.*?)(?:&|$)', link)
                        if url_param:
                            link = urllib.parse.unquote(url_param.group(1))
                            
                    snippet = row.get_text().strip()
                    results.append({
                        "title": title,
                        "link": link,
                        "snippet": snippet
                    })
            if len(results) >= num_results:
                break
        return results
    except Exception as e:
        print(f"[Search Error] Fallback DuckDuckGo search failed: {e}")
        return []

def get_search_context(query, num_results=4):
    """Perform search and format results into a context block for the model."""
    results = search_google(query, num_results)
    
    if not results:
        if not is_connected():
            return "\n[SYSTEM NOTE: The system is currently offline. Internet search is disabled. Relying entirely on pre-trained internal knowledge.]\n"
        return "\n[SYSTEM NOTE: Web search returned no results for this query.]\n"
        
    context = "\n--- GOOGLE SEARCH REAL-TIME RESULTS ---\n"
    for idx, res in enumerate(results, 1):
        context += f"[{idx}] Title: {res['title']}\n"
        context += f"    Source: {res['link']}\n"
        context += f"    Snippet: {res['snippet']}\n\n"
    context += "Use these search results to provide a fresh, factual response. Cite references as [1], [2], etc., where appropriate.\n"
    return context

if __name__ == "__main__":
    test_query = "who won the latest Cricket World Cup"
    print(f"Testing search for: '{test_query}'...")
    ctx = get_search_context(test_query, num_results=3)
    print(ctx)
