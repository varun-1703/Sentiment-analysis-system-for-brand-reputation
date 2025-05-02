# data_collector/rss_fetcher.py
import feedparser
import logging
from datetime import datetime, timezone
from time import mktime # To convert feedparser time struct to timestamp
from typing import List, Dict, Optional, Any
import sys
import os
import requests # To fetch feed content with timeout/headers


import config # Import config for limits

logger = logging.getLogger(__name__) # Get logger specific to this module

# --- User Agent for Requests ---
# Some feeds might block default Python User-Agents
REQUESTS_HEADERS = {
    'User-Agent': 'BrandMonitorApp/0.1 (RSS Fetcher; +http://example.com/botinfo)' # Customize agent
}

def _parse_rss_entry(entry: feedparser.FeedParserDict, feed_title: str, keyword: str) -> Optional[Dict[str, Any]]:
    """ Helper to parse an RSS feed entry into our standard dict format. """
    try:
        # Extract standard RSS fields, providing defaults
        title = entry.get('title', '') or ''
        # Try summary, fallback to description, then content snippets
        summary = entry.get('summary', '') or entry.get('description', '') or ''
        # Content can be a list, try to get the main text value
        content_list = entry.get('content', [])
        content_text = ""
        if isinstance(content_list, list) and content_list:
            # Prioritize 'text/plain' or 'text/html', fallback to first value
            for item in content_list:
                if isinstance(item, dict) and 'type' in item and 'value' in item:
                    if item['type'] == 'text/plain':
                        content_text = item['value']
                        break
                    elif item['type'] == 'text/html': # Could contain HTML tags
                        content_text = item['value'] # Store HTML for now, maybe strip later if needed
                        # Consider using BeautifulSoup here to strip HTML:
                        # from bs4 import BeautifulSoup
                        # content_text = BeautifulSoup(item['value'], 'html.parser').get_text()
            if not content_text and isinstance(content_list[0], dict) and 'value' in content_list[0]:
                 content_text = content_list[0]['value'] # Fallback to first item's value

        # Combine available text fields for matching and analysis
        # Be mindful of potential HTML tags in content_text if not stripped
        full_text = f"{title}. {summary}. {content_text}".strip()
        MAX_TEXT_LENGTH = 2000 # Allow longer text from RSS initially
        if len(full_text) > MAX_TEXT_LENGTH:
             full_text = full_text[:MAX_TEXT_LENGTH] + "..."


        # --- Keyword Matching ---
        # Simple case-insensitive check in the combined text
        if keyword.lower() not in full_text.lower():
            return None # Skip entry if keyword not found


        # --- Unique ID ---
        # Use 'guid' if available and looks like a permalink, otherwise 'link', fallback 'id'
        source_unique_id = entry.get('guid')
        if not source_unique_id or not entry.get('guidislink', False): # Check if guid is usable as link
             source_unique_id = entry.get('link')
        if not source_unique_id:
             source_unique_id = entry.get('id') # Fallback to 'id' field

        if not source_unique_id:
            logger.warning(f"RSS entry missing guid/link/id in feed '{feed_title}', skipping: {title[:50]}...")
            return None

        # --- Timestamp Parsing ---
        timestamp = datetime.now(timezone.utc) # Default
        # feedparser provides 'published_parsed' or 'updated_parsed' as time.struct_time
        published_parsed = entry.get('published_parsed') or entry.get('updated_parsed')
        if published_parsed:
            try:
                # Convert time.struct_time to Unix timestamp, then to aware datetime
                timestamp_unix = mktime(published_parsed)
                timestamp = datetime.fromtimestamp(timestamp_unix, tz=timezone.utc)
            except (TypeError, ValueError, OverflowError) as e:
                logger.warning(f"Could not parse RSS timestamp {published_parsed} for entry {source_unique_id}: {e}. Using current time.")


        return {
            "text": full_text,
            "source": f"RSS - {feed_title}", # Use the feed's title
            "source_unique_id": source_unique_id,
            "brand_keyword": keyword,
            "timestamp": timestamp,
            "url": entry.get('link') # Store original link if available, might differ from unique_id
        }
    except Exception as e:
        logger.error(f"Error parsing RSS entry {entry.get('link', 'No Link')} from feed '{feed_title}': {e}", exc_info=False)
        return None


def fetch_rss_feed(feed_url: str, keyword: str, limit: int = config.MAX_RESULTS_RSS) -> List[Dict[str, Any]]:
    """
    Fetches entries from a given RSS feed URL using feedparser, filtering by keyword.

    Args:
        feed_url: The URL of the RSS feed.
        keyword: The search term to filter entries by.
        limit: The maximum number of matching entries to return from this feed.

    Returns:
        A list of dictionaries, each representing a matching feed entry.
    """
    results: List[Dict[str, Any]] = []
    logger.info(f"Fetching RSS feed: {feed_url} for keyword '{keyword}'")

    try:
        # Use requests to fetch the feed content first - allows timeouts & headers
        response = requests.get(feed_url, headers=REQUESTS_HEADERS, timeout=15) # 15 second timeout
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        # Check content type - basic check for XML/RSS/Atom
        content_type = response.headers.get('content-type', '').lower()
        if not ('xml' in content_type or 'rss' in content_type or 'atom' in content_type):
             logger.warning(f"Unexpected content type '{content_type}' for RSS feed {feed_url}. Attempting parse anyway.")

        # Parse the fetched content using feedparser
        feed_data = feedparser.parse(response.content) # Parse raw bytes

        # Check for parsing errors (bozo bit)
        if feed_data.bozo:
            bozo_exception = feed_data.get('bozo_exception', 'Unknown parsing error')
            # Log specific types of common errors
            if isinstance(bozo_exception, requests.exceptions.RequestException):
                 logger.error(f"Network error fetching RSS feed {feed_url}: {bozo_exception}")
            elif isinstance(bozo_exception, feedparser.exceptions.CharacterEncodingOverride):
                  logger.warning(f"Character encoding issue parsing RSS feed {feed_url}: {bozo_exception}")
            else:
                 logger.warning(f"Failed to parse RSS feed {feed_url} (bozo=1): {bozo_exception}")
            return [] # Return empty on parse failure

        # Get feed title (fallback to URL)
        feed_title = feed_data.feed.get('title', feed_url)
        logger.info(f"Processing feed '{feed_title}' with {len(feed_data.entries)} entries.")

        # Iterate through entries and parse/filter
        count = 0
        for entry in feed_data.entries:
            if count >= limit:
                logger.info(f"Reached fetch limit ({limit}) for feed '{feed_title}'.")
                break

            parsed_entry = _parse_rss_entry(entry, feed_title, keyword)
            if parsed_entry:
                results.append(parsed_entry)
                count += 1

        logger.info(f"Found {len(results)} matching entries in RSS feed: '{feed_title}'")

    except requests.exceptions.Timeout:
         logger.error(f"Timeout error fetching RSS feed: {feed_url}")
    except requests.exceptions.RequestException as e:
        # Handles connection errors, invalid URLs, status code errors (4xx, 5xx) etc.
        logger.error(f"Network/HTTP error fetching RSS feed {feed_url}: {e}")
    except Exception as e:
        # Catch other unexpected errors during fetch or parsing
        logger.error(f"Unexpected error fetching or processing RSS feed {feed_url}: {e}", exc_info=True)

    return results


# --- Example Usage (for direct testing) ---
# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     # Example: Fetch news about 'Apple' from a tech news feed
#     test_rss_url = "http://feeds.arstechnica.com/arstechnica/index/" # Ars Technica main feed
#     test_keyword = "Apple" # Or try "Microsoft", "Google" etc.
#     print(f"Testing RSS fetcher for feed '{test_rss_url}' with keyword: '{test_keyword}'")

#     entries = fetch_rss_feed(test_rss_url, test_keyword, limit=5)
#     print(f"\n--- Found {len(entries)} matching entries ---")
#     for i, entry in enumerate(entries):
#         print(f"{i+1}. Source: {entry['source']}")
#         print(f"   Timestamp: {entry['timestamp']}")
#         print(f"   URL: {entry.get('url', 'N/A')}") # Use .get as URL might be missing
#         print(f"   UniqueID: {entry['source_unique_id']}")
#         print(f"   Text: {entry['text'][:100]}...")
#     print("\nRSS fetcher test complete.")