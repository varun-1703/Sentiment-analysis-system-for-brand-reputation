# data_collector/collector.py
import logging
from typing import List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
import sys
import os
import time


from data_collector import reddit_fetcher, news_fetcher, rss_fetcher
import config # Use config for enabling/disabling sources if needed



logger = logging.getLogger(__name__) # Logger for the coordinator

# --- Max workers for concurrent fetching ---
# Adjust based on typical number of sources and API rate limits.
# Too high might lead to rate limiting or blocking.
MAX_FETCH_WORKERS = 5

def collect_data(brand_keyword: str, rss_urls: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Collects data concurrently from all configured and requested sources.

    Args:
        brand_keyword: The primary keyword to search for.
        rss_urls: A list of RSS feed URLs to fetch (can be None or empty).

    Returns:
        A list of unique dictionaries, each representing a collected item.
        Uniqueness is based on 'source_unique_id'.
    """
    all_raw_results: List[Dict[str, Any]] = []
    rss_urls = rss_urls or [] # Ensure it's a list, even if empty

    start_time = time.time()
    logger.info(f"Starting data collection for keyword: '{brand_keyword}' with {len(rss_urls)} RSS feeds.")

    # Use ThreadPoolExecutor for I/O-bound tasks like API calls
    with ThreadPoolExecutor(max_workers=MAX_FETCH_WORKERS, thread_name_prefix='FetcherThread') as executor:
        # List to hold Future objects
        futures: List[Future] = []

        # --- Submit tasks to the executor ---

        # Submit Reddit Task (if configured)
        # Check relevant config values to see if Reddit is set up
        if config.REDDIT_CLIENT_ID and config.REDDIT_CLIENT_SECRET:
             logger.info(f"Submitting Reddit fetch task for '{brand_keyword}'")
             futures.append(executor.submit(reddit_fetcher.fetch_reddit_mentions, brand_keyword))
        else:
             logger.warning("Reddit fetcher skipped (Reddit credentials not fully configured in .env).")

        # Submit NewsAPI Task (if configured)
        if config.NEWS_API_KEY:
            logger.info(f"Submitting NewsAPI fetch task for '{brand_keyword}'")
            futures.append(executor.submit(news_fetcher.fetch_news_articles, brand_keyword))
        else:
            logger.warning("NewsAPI fetcher skipped (NEWS_API_KEY not configured in .env).")

        # Submit RSS Tasks (one per URL)
        if rss_urls:
            for url in rss_urls:
                if url and isinstance(url, str): # Basic validation
                    logger.info(f"Submitting RSS fetch task for '{brand_keyword}' from {url}")
                    futures.append(executor.submit(rss_fetcher.fetch_rss_feed, url, brand_keyword))
                else:
                     logger.warning(f"Skipping invalid RSS URL: {url}")
        else:
             logger.info("No valid RSS URLs provided, skipping RSS fetch tasks.")


        # --- Process results as they complete ---
        logger.info(f"Waiting for {len(futures)} fetcher tasks to complete...")
        for future in as_completed(futures):
            try:
                # Get the result list from the completed future
                # Each fetcher function returns a list of dicts
                result_list = future.result()
                if result_list and isinstance(result_list, list):
                    all_raw_results.extend(result_list)
                    # logger.debug(f"Received {len(result_list)} items from a completed task.")
                elif result_list: # Should be a list, log if not
                     logger.warning(f"Fetcher task returned unexpected type: {type(result_list)}. Expected list.")

            except Exception as e:
                # Log errors from individual fetcher tasks
                # The specific fetcher function should have logged details already
                logger.error(f"Error retrieving result from a fetcher task: {e}", exc_info=False) # Keep coordinator log cleaner


    # --- Deduplication ---
    # Use source_unique_id to remove duplicates gathered from potentially overlapping sources or within sources
    final_unique_results: List[Dict[str, Any]] = []
    seen_ids = set()
    duplicates_found = 0

    for item in all_raw_results:
        unique_id = item.get('source_unique_id')
        if unique_id:
            if unique_id not in seen_ids:
                final_unique_results.append(item)
                seen_ids.add(unique_id)
            else:
                duplicates_found += 1
        else:
            # Log items missing the crucial unique ID - should not happen if fetchers work correctly
            logger.warning(f"Collected item missing 'source_unique_id': {item.get('url', item.get('text', 'N/A')[:50])}...")


    end_time = time.time()
    duration = end_time - start_time
    logger.info(f"Data collection finished in {duration:.2f} seconds.")
    logger.info(f"Total raw items collected: {len(all_raw_results)}. Duplicates removed: {duplicates_found}.")
    logger.info(f"Returning {len(final_unique_results)} unique items for processing.")

    return final_unique_results


# --- Example Usage (for direct testing) ---
# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     test_keyword = "Microsoft"
#     test_feeds = [
#         "http://feeds.arstechnica.com/arstechnica/index/",
#         "https://www.theverge.com/rss/index.xml"
#         # Add more feeds for testing if desired
#     ]
#     # test_feeds = [] # Test without RSS

#     print(f"\n--- Testing Collector for keyword: '{test_keyword}' ---")
#     if test_feeds:
#          print(f"Using RSS feeds: {test_feeds}")

#     collected_data = collect_data(test_keyword, test_feeds)

#     print(f"\n--- Collector finished ---")
#     print(f"Total unique items collected: {len(collected_data)}")
#     print("\n--- Sample of collected items (first 5): ---")
#     for i, item in enumerate(collected_data[:5]):
#         print(f"{i+1}. ID: {item['source_unique_id']}")
#         print(f"   Source: {item['source']}")
#         print(f"   Timestamp: {item['timestamp']}")
#         print(f"   Text: {item['text'][:100]}...")
#     print("\nCollector test complete.")