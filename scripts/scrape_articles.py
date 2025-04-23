import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from datetime import datetime
import time
import xml.etree.ElementTree as ET

def scrape_article(url):
    """Scrapes a news article and stores it in DuckDB."""
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=5)
        response.encoding = 'utf-8'  # Ensure UTF-8 encoding
    except (requests.exceptions.Timeout, requests.exceptions.RequestException):
        print(f"Failed to fetch {url}: Timeout/Connection error")
        return None

    if response.status_code != 200:
        print(f"Failed to fetch {url}: {response.status_code}")
        return None

    soup = BeautifulSoup(response.text, "lxml")  # Using lxml parser for better performance

    # Initialize article data dictionary
    article_data = {
        'content_id': url.split('-')[-1] if '-' in url else url.split('/')[-1],
        'pubtime': '',
        'pubdate': '',
        'article_link': url,
        'medium_code': '20min_de' if '/fr/' not in url else '20min_fr',
        'medium_name': '20 Minuten Online',
        'language': 'de' if '/fr/' not in url else 'fr',
        'char_count': '',
        'head': '',
        'subhead': '',
        'content': ''
    }

    # Extract metadata from meta tags more efficiently
    meta_tags = {tag.get("property"): tag.get("content") for tag in soup.find_all("meta", property=["article:published_time", "article:section"])}
    
    if "article:published_time" in meta_tags:
        timestamp = meta_tags["article:published_time"]
        article_data['pubtime'] = timestamp.split('T')[1].strip().split('.')[0]
        article_data['pubdate'] = timestamp.split('T')[0].strip()
        
        # Skip date check for now since it's commented out
        # date = datetime.strptime(article_data['pubdate'], '%Y-%m-%d')
        
    if "article:section" in meta_tags:
        article_data['rubric'] = meta_tags["article:section"]

    # Extract title and subhead
    if title_tag := soup.find("title"):
        article_data['head'] = escape_xml_chars(title_tag.text.strip().split(' - 20')[0])

    if subhead_tag := soup.find("meta", {"property": "og:description"}):
        article_data['subhead'] = escape_xml_chars(subhead_tag.get("content"))

    # Extract article content more efficiently
    content_divs = soup.find_all("p")
    article_text = []
    
    # Handle lead paragraph
    if content_divs and (lead := escape_xml_chars(content_divs[0].get_text(strip=True))):
        article_text.append(f"<ld><p>{lead}</p></ld>")
        content_divs = content_divs[1:]

    # Process remaining paragraphs with improved text handling
    current_paragraph = []
    for paragraph in content_divs:
        if text := escape_xml_chars(paragraph.get_text(strip=True)):
            # Check if this paragraph starts with lowercase (potential split)
            if current_paragraph and text and text[0].islower():
                # Append to previous paragraph
                current_paragraph[-1] = current_paragraph[-1].rstrip('&apos;') + text
            else:
                current_paragraph.append(text)
            
            if paragraph.get('class') and any('title' in c.lower() for c in paragraph['class']):
                article_text.append(f"<zt>{text}</zt>")
            else:
                # Join split paragraphs and then split by length if needed
                full_text = ' '.join(current_paragraph)
                current_paragraph = []
                
                # Split on sentence boundaries if possible when breaking long paragraphs
                if len(full_text) > 1000:
                    sentences = full_text.split('. ')
                    current_chunk = []
                    current_length = 0
                    
                    for sentence in sentences:
                        if current_length + len(sentence) > 1000:
                            article_text.append(f"<p>{'. '.join(current_chunk)}.</p>")
                            current_chunk = [sentence]
                            current_length = len(sentence)
                        else:
                            current_chunk.append(sentence)
                            current_length += len(sentence) + 2  # +2 for '. '
                    
                    if current_chunk:
                        article_text.append(f"<p>{'. '.join(current_chunk)}.</p>")
                else:
                    article_text.append(f"<p>{full_text}</p>")

    stop_marker = '<p>Deine Meinung zählt' if '/fr/' not in url else '<p>Ton opinion'
    content = ''.join(article_text).split(stop_marker)[0]
    
    # Validate XML structure
    try:
        article_data['content'] = f"<tx>{content}</tx>"
        ET.fromstring(article_data['content'])  # Validate XML
    except ET.ParseError:
        print(f"Invalid XML structure in article {url}")
        with open(f"invalid_xml_articles.txt", "a") as f:
            f.write(f"{article_data['content']}\n")
        return None

    article_data['char_count'] = len(article_data['content'])

    if article_data['char_count'] == 0 or not article_data['pubdate']:
        print(f"Skipping article {url} due to missing content or pubdate")
        return None

    return article_data

def escape_xml_chars(text):
    """Escape special characters for XML and fix split words."""
    if not isinstance(text, str):
        return text
    
    # First escape XML special characters
    escaped = text.translate(str.maketrans({
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&apos;'
    }))
    
    # Remove any soft hyphens that might cause word splits
    escaped = escaped.replace('\u00AD', '')
    
    return escaped

def save_as_tsv(article):
    """Saves article as a TSV file locally based on language."""
    filename = f"./data/scraped_articles/{'fr' if article['language'] == 'fr' else 'de'}/articles.tsv"
    
    mode = "a" if os.path.exists(filename) else "w"
    
    with open(filename, mode, encoding="utf-8") as f:
        if mode == "w":
            f.write("\t".join(article.keys()) + "\n")
        f.write("\t".join(str(v) for v in article.values()) + "\n")

def scrape_and_store(url):
    """Scrapes an article and saves it as TSV."""
    if article := scrape_article(url):
        save_as_tsv(article)
        return True
    return False

# Generate URLs for all possible articles
# first run
max_fr_id, min_fr_id = 103230035, 26977512
max_de_id, min_de_id = 103229590, 17111041

# second run
max_fr_id, min_fr_id = 102892040, 26977512
max_de_id, min_de_id = 102882918, 17111041

# Get URLs from predefined files
def process_urls(filename):
    with open(filename, "r") as f:
        return [f"https://{url.strip().split('https:/')[1].rsplit('/', 1)[0] + '/' + url.strip().split('https:/')[1].split('-')[-1]}" for url in f]

fr_urls = process_urls("data/fr_urls.csv")
de_urls = process_urls("data/de_urls.csv")

urls = fr_urls + de_urls

# Use parallel scraping with optimized settings
with ThreadPoolExecutor(max_workers=64) as executor:
    # Submit all URLs at once and process both languages simultaneously
    futures = [
        executor.submit(scrape_and_store, url) 
        for url in urls
    ]
    
    completed = sum(1 for future in as_completed(futures) 
                   if future.result(timeout=60))

print(f"Scraping completed! Successfully processed {completed} articles")