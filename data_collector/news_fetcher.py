# data_collector/news_fetcher.py
from newsapi import NewsApiClient
import logging
from datetime import datetime, timezone
from dateutil import parser as dateutil_parser # Use alias to avoid confusion with ArgumentParser
from typing import List, Dict, Optional, Any
import sys
import os


import config 

logger = logging.getLogger(__name__) # Get logger specific to this module

# --- Global NewsAPI Client ---
_news_client = None

def get_news_client() -> Optional[NewsApiClient]:
    """
    Initializes and returns a NewsApiClient instance using the API key from config.
    Uses a cached instance if available. Returns None on failure.
    """
    global _news_client
    if _news_client:
        return _news_client

    if not config.NEWS_API_KEY:
        logger.error("NEWS_API_KEY missing in config/env. Cannot initialize NewsAPI client.")
        return None

    try:
        logger.info("Initializing NewsAPI client...")
        newsapi = NewsApiClient(api_key=config.NEWS_API_KEY)
        # We can't easily test the connection without making a request,
        # so we assume initialization is successful here. Errors handled during fetch.
        _news_client = newsapi
        logger.info("NewsAPI client initialized.")
        return newsapi
    except Exception as e:
        logger.error(f"Failed to initialize NewsAPI client: {e}", exc_info=True)
        return None


def _parse_article(article: Dict[str, Any], keyword: str) -> Optional[Dict[str, Any]]:
    """ Helper to parse a news article dict into our standard format. """
    try:
        title = article.get('title', '') or ''
        description = article.get('description', '') or ''
        content = article.get('content', '') or '' # Content might be truncated by NewsAPI

        # Combine fields for analysis text, prioritizing description/content over just title
        text_content = f"{title}. {description} {content}".strip()
        # Limit length - consider source truncation already done by NewsAPI
        MAX_TEXT_LENGTH = 1500 # Adjust as needed
        if len(text_content) > MAX_TEXT_LENGTH:
             text_content = text_content[:MAX_TEXT_LENGTH] + "..."
        elif not text_content: # Skip if no text content at all
             logger.warning(f"Article skipped due to empty text content: {article.get('url', 'No URL')}")
             return None


        # Safely parse timestamp using dateutil.parser which handles various formats
        timestamp_str = article.get('publishedAt')
        timestamp = datetime.now(timezone.utc) # Default to current time if parsing fails
        if timestamp_str:
            try:
                timestamp = dateutil_parser.parse(timestamp_str)
                # Ensure timestamp is timezone-aware (assume UTC if not specified)
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=timezone.utc)
            except (ValueError, OverflowError) as e:
                 logger.warning(f"Could not parse timestamp '{timestamp_str}' for article {article.get('url', '')}: {e}. Using current time.")


        # Use URL as unique ID (most reliable)
        source_unique_id = article.get('url')
        if not source_unique_id:
            logger.warning(f"Article skipped due to missing URL: {title[:50]}...")
            return None

        # Extract source name
        source_name = article.get('source', {}).get('name', 'Unknown Source')

        return {
            "text": text_content,
            "source": f"NewsAPI - {source_name}", # Include source name
            "source_unique_id": source_unique_id,
            "brand_keyword": keyword,
            "timestamp": timestamp,
            "url": source_unique_id # URL is also the unique ID here
        }
    except Exception as e:
        logger.error(f"Error parsing article {article.get('url', 'No URL')}: {e}", exc_info=False)
        return None


def fetch_news_articles(keyword: str, limit: int = config.MAX_RESULTS_NEWS) -> List[Dict[str, Any]]:
    """
    Fetches recent news articles mentioning the keyword using NewsAPI.org.

    Args:
        keyword: The search term.
        limit: The approximate maximum number of articles to return.

    Returns:
        A list of dictionaries, each representing an article.
    """
    newsapi = get_news_client()
    if not newsapi:
        logger.error("Cannot fetch news articles: NewsAPI client not available.")
        return []

    # Ensure limit is positive
    limit = max(1, limit)
    # NewsAPI page size limit is 100, request slightly more if needed and slice later
    api_page_size = min(limit, 100)

    results: List[Dict[str, Any]] = []
    try:
        logger.info(f"Searching NewsAPI for '{keyword}' (limit ~{limit}, page_size {api_page_size})...")
        # Use 'everything' endpoint for broader search.
        # Free tier restrictions: limited lookback (1 month), maybe limited sources, dev use only.
        # Consider 'qintitle' parameter if keyword must be in the title.
        all_articles_response = newsapi.get_everything(
            q=keyword,
            language='en',      # Filter by language
            sort_by='publishedAt', # Options: 'relevancy', 'popularity', 'publishedAt'
            page_size=api_page_size,
            # page=1 # Only fetching the first page for simplicity here
        )

        # Check API response status
        if all_articles_response.get('status') == 'ok':
            articles_data = all_articles_response.get('articles', [])
            total_results = all_articles_response.get('totalResults', 0)
            logger.info(f"NewsAPI returned {len(articles_data)} articles (total available: {total_results}). Processing up to {limit}.")

            # Parse and collect articles up to the specified limit
            for article_data in articles_data:
                if len(results) >= limit:
                    break # Stop processing once limit is reached
                parsed_article = _parse_article(article_data, keyword)
                if parsed_article:
                    results.append(parsed_article)

        else:
            # Log specific API errors
            error_code = all_articles_response.get('code')
            error_message = all_articles_response.get('message')
            logger.error(f"NewsAPI request failed for '{keyword}'. Code: {error_code}, Message: {error_message}")
            # Handle specific errors if needed (e.g., rate limits, API key issues)
            if error_code == 'rateLimited':
                 logger.warning("NewsAPI rate limit exceeded.")
            elif error_code == 'apiKeyInvalid' or error_code == 'apiKeyMissing':
                 logger.error("NewsAPI key is missing or invalid. Please check .env file.")


    except Exception as e:
        # Catch potential network errors or unexpected issues with the client/response
        logger.error(f"Unexpected error fetching data from NewsAPI for '{keyword}': {e}", exc_info=True)

    logger.info(f"Finished NewsAPI fetch for '{keyword}'. Total items collected: {len(results)} (target limit: {limit}).")
    return results


# --- Example Usage (for direct testing) ---
# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     test_keyword = "NVIDIA" # Replace with a relevant keyword
#     print(f"Testing NewsAPI fetcher with keyword: '{test_keyword}'")
#     articles = fetch_news_articles(test_keyword, limit=5)
#     print(f"\n--- Found {len(articles)} articles ---")
#     for i, article in enumerate(articles):
#         print(f"{i+1}. Source: {article['source']}")
#         print(f"   Timestamp: {article['timestamp']}")
#         print(f"   URL: {article['url']}")
#         print(f"   Text: {article['text'][:100]}...")
#     print("\nNewsAPI fetcher test complete.")