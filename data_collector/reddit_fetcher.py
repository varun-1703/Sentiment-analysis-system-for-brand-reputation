# data_collector/reddit_fetcher.py
import praw
import logging
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any
import sys
import os
 
import config # Import config for credentials and limits



logger = logging.getLogger(__name__) # Get logger specific to this module

# --- Global Reddit Instance (Optional but avoids reconnecting every time) ---
# Be mindful of potential state issues if PRAW instance is not thread-safe,
# but for simple read operations, it's generally okay.
_reddit_instance = None

def get_reddit_instance() -> Optional[praw.Reddit]:
    """
    Creates and returns a Reddit API instance using credentials from config.
    Uses a cached instance if available. Returns None on failure.
    """
    global _reddit_instance
    if _reddit_instance:
        # Basic check if connection might be stale (not foolproof)
        try:
             if _reddit_instance.user.me():
                 return _reddit_instance
             else: # Authentication issue?
                 logger.warning("Reddit instance seems stale (auth failed). Reconnecting.")
                 _reddit_instance = None # Force reconnection
        except Exception as e:
             logger.warning(f"Error checking Reddit instance state ({e}). Reconnecting.")
             _reddit_instance = None


    # Check if all required credentials are present
    required_creds = [
        config.REDDIT_CLIENT_ID,
        config.REDDIT_CLIENT_SECRET,
        config.REDDIT_USER_AGENT,
        config.REDDIT_USERNAME,
        config.REDDIT_PASSWORD
    ]
    if not all(required_creds):
        logger.error("Reddit credentials missing in config/env. Cannot connect.")
        return None

    try:
        logger.info("Attempting to connect to Reddit API...")
        reddit = praw.Reddit(
            client_id=config.REDDIT_CLIENT_ID,
            client_secret=config.REDDIT_CLIENT_SECRET,
            user_agent=config.REDDIT_USER_AGENT,
            username=config.REDDIT_USERNAME,
            password=config.REDDIT_PASSWORD,
            # Optional: Set timeout for requests
            # timeout=10 # seconds
            # Optional: Disable PRAW's update check for slight speedup
            check_for_async=False
        )
        # Test connection by trying to access the authenticated user
        # This will raise an exception if authentication fails
        reddit.user.me()
        logger.info("Successfully connected to Reddit API.")
        _reddit_instance = reddit # Cache the instance
        return reddit
    except praw.exceptions.PRAWException as e:
         logger.error(f"PRAW specific error connecting to Reddit: {e}", exc_info=False) # Often auth errors
         return None
    except Exception as e:
        # Catch other potential errors (network issues, etc.)
        logger.error(f"Generic error connecting to Reddit API: {e}", exc_info=True)
        return None


def _parse_submission(submission: praw.models.Submission, keyword: str) -> Optional[Dict[str, Any]]:
    """ Helper to parse a PRAW submission object into our standard dict format. """
    try:
        # Combine title and selftext, handle potential None values gracefully
        title = submission.title or ""
        selftext = submission.selftext or ""
        # Limit combined length to avoid excessively long text for analysis
        text_content = f"{title}. {selftext}".strip()
        MAX_TEXT_LENGTH = 1500 # Adjust as needed
        if len(text_content) > MAX_TEXT_LENGTH:
             text_content = text_content[:MAX_TEXT_LENGTH] + "..."


        # Ensure timestamp is timezone-aware UTC
        timestamp = datetime.fromtimestamp(submission.created_utc, tz=timezone.utc)

        # Construct the unique ID
        source_unique_id = f"reddit_submission_{submission.id}"

        return {
            "text": text_content,
            "source": "Reddit Submission", # Be specific about the source type
            "source_unique_id": source_unique_id,
            "brand_keyword": keyword,
            "timestamp": timestamp,
            "url": f"https://www.reddit.com{submission.permalink}" # Construct full URL
        }
    except Exception as e:
        logger.error(f"Error parsing submission {getattr(submission, 'id', 'N/A')}: {e}", exc_info=False)
        return None

def _parse_comment(comment: praw.models.Comment, keyword: str) -> Optional[Dict[str, Any]]:
    """ Helper to parse a PRAW comment object into our standard dict format. """
    try:
        # Limit comment body length
        text_content = comment.body or ""
        MAX_TEXT_LENGTH = 1500 # Adjust as needed
        if len(text_content) > MAX_TEXT_LENGTH:
             text_content = text_content[:MAX_TEXT_LENGTH] + "..."

        # Ensure timestamp is timezone-aware UTC
        timestamp = datetime.fromtimestamp(comment.created_utc, tz=timezone.utc)

        # Construct the unique ID
        source_unique_id = f"reddit_comment_{comment.id}"

        return {
            "text": text_content,
            "source": "Reddit Comment", # Specific source type
            "source_unique_id": source_unique_id,
            "brand_keyword": keyword,
            "timestamp": timestamp,
            "url": f"https://www.reddit.com{comment.permalink}" # Construct full URL
        }
    except Exception as e:
        logger.error(f"Error parsing comment {getattr(comment, 'id', 'N/A')}: {e}", exc_info=False)
        return None


def fetch_reddit_mentions(keyword: str, limit: int = config.MAX_RESULTS_REDDIT) -> List[Dict[str, Any]]:
    """
    Fetches recent Reddit submissions and comments mentioning the keyword across Reddit.

    Args:
        keyword: The search term (brand name, etc.).
        limit: The approximate maximum number of items to return (split between submissions/comments).

    Returns:
        A list of dictionaries, each representing a mention.
    """
    reddit = get_reddit_instance()
    if not reddit:
        logger.error("Cannot fetch Reddit mentions: Reddit instance not available.")
        return []

    # Ensure limit is positive and reasonable
    limit = max(1, limit)
    # Split limit approximately between submissions and comments
    sub_limit = max(1, limit // 2)
    com_limit = max(1, limit - sub_limit)

    results: List[Dict[str, Any]] = []
    processed_ids = set() # Track IDs within this fetch to avoid duplicates from overlapping searches

    # --- Search Submissions ---
    try:
        logger.info(f"Searching Reddit submissions for '{keyword}' (limit ~{sub_limit})...")
        # Search across 'all' subreddits, sort by 'new'
        # PRAW handles iteration and pagination internally up to the limit
        for submission in reddit.subreddit("all").search(keyword, limit=sub_limit, sort="new"):
            if submission.id not in processed_ids:
                parsed_submission = _parse_submission(submission, keyword)
                if parsed_submission:
                    results.append(parsed_submission)
                    processed_ids.add(submission.id)
            # Break early if we somehow exceed the overall target limit
            if len(results) >= limit: break
        logger.info(f"Found {len(results)} potential submissions.")

    except praw.exceptions.PRAWException as e:
         logger.error(f"PRAW Error searching Reddit submissions for '{keyword}': {e}", exc_info=False)
    except Exception as e:
        logger.error(f"Unexpected error searching Reddit submissions for '{keyword}': {e}", exc_info=True)


    # --- Search Comments ---
    # Note: PRAW's comment search might be less comprehensive or have different limitations
    # compared to submission search or dedicated push services (like Pushshift, if available).
    # Only proceed if we haven't reached the limit yet.
    remaining_limit = limit - len(results)
    if remaining_limit > 0:
        try:
            logger.info(f"Searching Reddit comments for '{keyword}' (limit ~{min(com_limit, remaining_limit)})...")
            # Use search_comments if available and suitable
            for comment in reddit.subreddit("all").search_comments(keyword, limit=min(com_limit, remaining_limit), sort="new"):
                 if comment.id not in processed_ids:
                    parsed_comment = _parse_comment(comment, keyword)
                    if parsed_comment:
                         results.append(parsed_comment)
                         processed_ids.add(comment.id)
                 # Break early if limit reached
                 if len(results) >= limit: break
            logger.info(f"Found {len(results) - (limit - remaining_limit)} potential comments.")

        except praw.exceptions.PRAWException as e:
             logger.error(f"PRAW Error searching Reddit comments for '{keyword}': {e}", exc_info=False)
        except Exception as e:
            logger.error(f"Unexpected error searching Reddit comments for '{keyword}': {e}", exc_info=True)

    logger.info(f"Finished Reddit fetch for '{keyword}'. Total items collected: {len(results)} (target limit: {limit}).")
    return results


# --- Example Usage (for direct testing) ---
# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     test_keyword = "OpenAI" # Replace with a relevant keyword for testing
#     print(f"Testing Reddit fetcher with keyword: '{test_keyword}'")
#     mentions = fetch_reddit_mentions(test_keyword, limit=10)
#     print(f"\n--- Found {len(mentions)} mentions ---")
#     for i, mention in enumerate(mentions):
#         print(f"{i+1}. Source: {mention['source']}")
#         print(f"   Timestamp: {mention['timestamp']}")
#         print(f"   URL: {mention['url']}")
#         print(f"   Text: {mention['text'][:100]}...")
#     print("\nReddit fetcher test complete.")