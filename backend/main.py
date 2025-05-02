# backend/main.py
import sys
import os

# --- START: Ensure project root is in sys.path ---
# Calculate the project root directory (one level up from 'backend')
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# If the project root is not already in the Python path, add it
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    print(f"DEBUG: Added project root to sys.path: {PROJECT_ROOT}") # Add debug print
else:
    print(f"DEBUG: Project root already in sys.path: {PROJECT_ROOT}") # Add debug print
# --- END: Ensure project root is in sys.path ---

# --- Now proceed with other imports ---
import logging
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Query, status
from datetime import datetime, timezone
from typing import List, Optional
import pandas as pd

# Import local modules AFTER potentially modifying sys.path
try:
    from backend.models import CollectRequest, AnalyzedDataResponse, TopicInfoResponse, StatusResponse, ErrorResponse
    from backend import database # This import should trigger database.py
    from nlp import processor
    from data_collector import collector
    import config # This should now work if PROJECT_ROOT was added correctly
except ModuleNotFoundError as e:
    print(f"ERROR: Failed to import module in backend/main.py after adjusting sys.path: {e}.")
    print(f"Current sys.path: {sys.path}") # Print path for debugging
    sys.exit(1)
except ImportError as e:
     print(f"ERROR: Failed with ImportError in backend/main.py: {e}.")
     print(f"Current sys.path: {sys.path}")
     sys.exit(1)

# ... rest of your main.py code ...


# Configure logging
# Use format that includes module name for easier debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Brand Reputation Monitor API",
    description="API for collecting, analyzing text sentiment/topics for brand reputation.",
    version="0.2.1", # Updated version
    # Define potential error responses for OpenAPI documentation
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        404: {"model": ErrorResponse, "description": "Not Found"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    }
)


# --- Background Task Definition ---
# Moved inside the endpoint that uses it for clarity, or keep here if reused widely
def process_and_store_data(collected_items: List[dict]):
    """
    Background task: Analyzes collected data and stores it in the database.
    Logs progress and errors.
    """
    processed_count = 0
    skipped_due_to_duplicate = 0
    skipped_due_to_error = 0
    skipped_due_to_missing_data = 0
    error_count = 0
    total_items = len(collected_items)
    logger.info(f"BACKGROUND TASK: Starting processing for {total_items} collected items.")

    # --- Pre-computation checks (optional but can save time) ---
    # Check topic model status once before the loop
    topic_model_ready = False
    try:
        current_topic_model = processor.get_topic_model()
        topic_model_ready = current_topic_model is not None and processor.is_topic_model_fitted
        if not topic_model_ready:
             logger.warning("BACKGROUND TASK: Topic model not fitted. Topics assigned will be -1 unless model is trained/loaded successfully.")
    except Exception as e:
         logger.error(f"BACKGROUND TASK: Error checking topic model status: {e}", exc_info=True)


    # --- Processing Loop ---
    for i, item in enumerate(collected_items):
        # Log progress periodically
        if (i + 1) % 50 == 0:
            logger.info(f"BACKGROUND TASK: Processing item {i+1}/{total_items}...")

        text = item.get('text')
        unique_id = item.get('source_unique_id')
        timestamp = item.get('timestamp') # This should be a datetime object from fetchers

        # Validate essential data
        if not text or not unique_id or not timestamp:
            logger.warning(f"BACKGROUND TASK: Skipping item due to missing text, unique_id, or timestamp. ID: {unique_id or 'Unknown'}")
            skipped_due_to_missing_data += 1
            continue

        # Ensure timestamp is datetime object (should be, but double-check)
        if not isinstance(timestamp, datetime):
            logger.warning(f"BACKGROUND TASK: Timestamp for {unique_id} is not a datetime object ({type(timestamp)}). Skipping.")
            skipped_due_to_missing_data += 1
            continue

        # --- Database Duplicate Check ---
        # Check DB just before processing, reduces race conditions slightly
        # database.check_if_exists handles its own logging
        if database.check_if_exists(unique_id):
             skipped_due_to_duplicate += 1
             continue # Skip if already in DB

        # --- NLP Analysis ---
        try:
            # 1. Analyze Sentiment
            sentiment_label, sentiment_score = processor.analyze_sentiment(text)
            if sentiment_label == "error":
                logger.error(f"BACKGROUND TASK: Sentiment analysis failed for item: {unique_id}. Skipping storage.")
                skipped_due_to_error += 1
                continue # Skip if sentiment fails

            # 2. Assign Topic (handle case where model isn't fitted)
            topic_id = -1 # Default to outlier/unassigned
            if topic_model_ready:
                # get_topic handles its own internal error logging
                topic_id = processor.get_topic(text)
            # No 'else' needed, topic_id remains -1 if model wasn't ready

        except Exception as e:
             logger.error(f"BACKGROUND TASK: Unexpected error during NLP analysis for item {unique_id}: {e}", exc_info=True)
             error_count += 1
             skipped_due_to_error += 1
             continue # Skip storing this item

        # --- Store in Database ---
        try:
            db_id = database.add_analyzed_text(
                text=text,
                source=item.get('source', 'Unknown'),
                source_unique_id=unique_id,
                brand_keyword=item.get('brand_keyword', 'N/A'),
                timestamp=timestamp, # Pass the datetime object
                sentiment_label=sentiment_label,
                sentiment_score=sentiment_score,
                topic_id=topic_id
            )
            if db_id:
                processed_count += 1
            else:
                # This case typically means it was added between check_if_exists and insert (race condition)
                # add_analyzed_text logs this warning. Or DB error occurred (logged by add_analyzed_text).
                 skipped_due_to_duplicate += 1 # Assume duplicate if ID is None and no exception

        except Exception as e:
            logger.error(f"BACKGROUND TASK: Unexpected error storing item {unique_id} in database: {e}", exc_info=True)
            error_count += 1
            skipped_due_to_error += 1


    # --- Log Summary ---
    logger.info(f"BACKGROUND TASK finished. "
                f"Total items received: {total_items}. "
                f"Successfully processed & added: {processed_count}. "
                f"Skipped (already in DB): {skipped_due_to_duplicate}. "
                f"Skipped (missing data): {skipped_due_to_missing_data}. "
                f"Skipped (NLP/DB error): {skipped_due_to_error}. "
                f"Other errors: {error_count}.")


# --- API Endpoints ---

@app.on_event("startup")
async def startup_event():
    """ Actions to perform once on application startup. """
    logger.info("Application starting up...")
    # Ensure DB table exists
    database.create_table()
    logger.info("Database checked/initialized.")
    # Trigger initial load/check of NLP models (sentiment loads on first use or import)
    logger.info("Initializing NLP models...")
    try:
        _ = processor.sentiment_pipeline # Access to ensure loading is attempted if not already done
        processor.get_topic_model() # Load/create topic model instance and check status
        logger.info(f"NLP models initialized. Sentiment ready: {processor.sentiment_pipeline is not None}. Topic model ready: {processor.topic_model is not None}, Fitted: {processor.is_topic_model_fitted}")
    except Exception as e:
        logger.error(f"Critical error during NLP model initialization on startup: {e}", exc_info=True)
        # Depending on severity, could raise exception to stop server or just log warning
    logger.info("Startup complete. API ready.")


@app.post("/collect",
          status_code=status.HTTP_202_ACCEPTED, # Correct status code for accepted background task
          response_model=StatusResponse,
          summary="Trigger data collection and analysis",
          description="Accepts a keyword and optional RSS feeds, starts collecting data from configured sources (Reddit, NewsAPI, RSS), and schedules background analysis.")
async def trigger_collection(request: CollectRequest, background_tasks: BackgroundTasks):
    """
    Endpoint to initiate data collection and analysis.
    """
    logger.info(f"Received collection request for keyword: '{request.brand_keyword}' with {len(request.rss_urls or [])} RSS feeds.")
    if not request.brand_keyword:
        # Should be caught by Pydantic, but good practice to check
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Brand keyword must be provided."
        )

    # --- Data Collection Phase ---
    # This runs synchronously within the request for simplicity here.
    # For very long collections, consider making collector.py async or running it in a separate process/queue.
    try:
        # Pass URLs directly; collector handles None case
        collected_items = collector.collect_data(request.brand_keyword, request.rss_urls)
        item_count = len(collected_items)
        logger.info(f"Collection phase completed for '{request.brand_keyword}'. Found {item_count} raw items across sources.")
    except Exception as e:
         logger.error(f"Error during data collection phase for '{request.brand_keyword}': {e}", exc_info=True)
         raise HTTPException(
             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
             detail=f"Data collection failed: {e}"
         )

    # --- Schedule Background Analysis ---
    if collected_items:
        # Add the processing task to run after the response is sent
        background_tasks.add_task(process_and_store_data, collected_items)
        message = f"Collection initiated for '{request.brand_keyword}'. Scheduled background analysis for {item_count} collected items."
        logger.info(message)
        return StatusResponse(message=message, details=f"{item_count} items passed to background processing.")
    else:
        message = f"No new items collected for '{request.brand_keyword}'. Analysis task not scheduled."
        logger.info(message)
        # Still return 202, but indicate nothing was scheduled
        return StatusResponse(message=message, details="No items found or all were duplicates during collection.")


@app.get("/data",
         response_model=List[AnalyzedDataResponse],
         summary="Retrieve analyzed data",
         description="Fetches recently analyzed data stored in the database, with pagination.")
async def get_data(
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records to return"),
    offset: int = Query(0, ge=0, description="Number of records to skip for pagination")
    ):
    """
    Endpoint to retrieve stored analyzed data.
    """
    logger.info(f"Data request received: limit={limit}, offset={offset}")
    try:
        # Fetch raw data (list of Row objects)
        db_rows = database.get_all_analyzed_data(limit=limit, offset=offset)
        if db_rows is None: # Handle potential connection error in get_all_analyzed_data
             raise HTTPException(status_code=500, detail="Failed to connect to database.")

        # Convert sqlite3.Row objects to dicts first, then to Pydantic models
        results = []
        for row in db_rows:
            try:
                # Convert Row to dict
                row_dict = dict(row)
                # Ensure timestamps are properly parsed (SQLite stores them as strings)
                if 'timestamp' in row_dict and isinstance(row_dict['timestamp'], str):
                    row_dict['timestamp'] = datetime.fromisoformat(row_dict['timestamp'].replace('Z', '+00:00'))
                if 'fetch_timestamp' in row_dict and isinstance(row_dict['fetch_timestamp'], str):
                    row_dict['fetch_timestamp'] = datetime.fromisoformat(row_dict['fetch_timestamp'].replace('Z', '+00:00'))
                
                # Create Pydantic model - setting from_attributes to False since we're using a dict
                model = AnalyzedDataResponse.model_validate(row_dict)
                results.append(model)
            except Exception as model_error:
                logger.error(f"Error converting row to Pydantic model: {model_error} - Row data: {row}", exc_info=True)
                # Continue processing other rows instead of failing entirely
                continue
                
        logger.info(f"Returning {len(results)} data records.")
        return results
    except Exception as e:
        logger.error(f"Error retrieving or processing data from database: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve or process data: {str(e)}"
        )


@app.post("/retrain-topic-model",
           status_code=status.HTTP_202_ACCEPTED,
           response_model=StatusResponse,
           summary="Trigger topic model retraining",
           description="Fetches all text from the database and schedules a background task to retrain the BERTopic model.")
async def trigger_topic_model_retraining(background_tasks: BackgroundTasks):
    """
    Endpoint to initiate topic model retraining.
    """
    logger.info("Received request to retrain topic model.")

    # --- Define the Retraining Background Task ---
    def retrain_task():
        task_logger = logging.getLogger("RetrainTask") # Use a specific logger if desired
        task_logger.info("Starting background topic model retraining task...")
        try:
            texts = database.get_all_texts_for_topic_model()
            if not texts:
                task_logger.warning("No texts found in database to train topic model. Task aborted.")
                return # Exit task if no data

            task_logger.info(f"Retrieved {len(texts)} texts for retraining.")
            # processor.train_topic_model handles its own logging
            success = processor.train_topic_model(texts)

            if success:
                task_logger.info("Topic model retraining completed successfully and model saved.")
                # Optional: Trigger update of topic IDs for existing entries
                # Consider performance implications - this could be slow for large DBs
                # try:
                #     update_existing_topics()
                # except Exception as e:
                #     task_logger.error(f"Error updating existing topic IDs after retraining: {e}", exc_info=True)
            else:
                task_logger.error("Topic model retraining failed. Check NLP processor logs.")

        except Exception as e:
            task_logger.error(f"Unexpected error during retraining task: {e}", exc_info=True)

    # --- Schedule and Respond ---
    background_tasks.add_task(retrain_task)
    message = "Topic model retraining task scheduled. Check backend logs for progress and completion."
    logger.info(message)
    return StatusResponse(message=message)


@app.get("/topics",
         response_model=List[TopicInfoResponse],
         summary="Retrieve topic information",
         description="Fetches the list of topics (ID, count, name, etc.) identified by the currently fitted BERTopic model.")
async def get_topics():
    """
    Endpoint to retrieve information about the topics found by the model.
    """
    logger.info("Topic info request received.")

    # Ensure model is loaded and check if fitted
    current_topic_model = processor.get_topic_model() # Ensures model is loaded/attempted
    if current_topic_model is None or not processor.is_topic_model_fitted:
        logger.warning("Cannot retrieve topics: Model not available or not fitted.")
        # Return empty list, frontend should handle this state
        return []
        # Or raise 404:
        # raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Topic model has not been trained yet.")

    try:
        # Get DataFrame from processor
        topic_info_df = processor.get_topic_info()
        if topic_info_df is None or topic_info_df.empty:
             logger.warning("processor.get_topic_info() returned empty or None.")
             return [] # Return empty list if no topics found or error in processor

        # Convert DataFrame records (dict) to Pydantic model list
        # Ensure the keys in the dicts match the Pydantic field names (case-sensitive)
        # If using aliases in Pydantic, ensure allow_population_by_field_name=True if needed
        # Here, field names match DataFrame columns ('Topic', 'Count', 'Name')
        try:
             response_data = [TopicInfoResponse.model_validate(topic) for topic in topic_info_df.to_dict('records')]
             # Pydantic V1: response_data = [TopicInfoResponse.parse_obj(topic) for topic in topic_info_df.to_dict('records')]
        except Exception as pydantic_error:
             logger.error(f"Error converting topic DataFrame to Pydantic model: {pydantic_error}", exc_info=True)
             raise HTTPException(status_code=500, detail="Failed to format topic data.")


        logger.info(f"Returning information for {len(response_data)} topics.")
        return response_data

    except Exception as e:
        logger.error(f"Error retrieving topic information: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve topic information."
        )


# --- Optional: Function to update existing topics ---
# def update_existing_topics():
#     task_logger = logging.getLogger("UpdateTopicsTask")
#     task_logger.info("Attempting to update topic IDs for existing database entries...")
#     conn = database.get_db_connection()
#     if not conn:
#         task_logger.error("Cannot connect to database to update topics.")
#         return
#
#     updated_count = 0
#     error_count = 0
#     try:
#         cursor = conn.cursor()
#         # Fetch id and text for all entries (could be memory intensive!)
#         # Consider fetching in batches for large databases
#         cursor.execute("SELECT id, text FROM analyzed_texts")
#         rows = cursor.fetchall()
#         task_logger.info(f"Fetched {len(rows)} existing entries to re-assign topics.")
#
#         if not processor.is_topic_model_fitted:
#              task_logger.error("Topic model is not fitted. Cannot re-assign topics.")
#              return
#
#         # Prepare update statement
#         update_cursor = conn.cursor()
#         for row in rows:
#             row_id = row['id']
#             text = row['text']
#             if not text: continue
#
#             try:
#                 new_topic_id = processor.get_topic(text)
#                 update_cursor.execute("UPDATE analyzed_texts SET topic_id = ? WHERE id = ?", (new_topic_id, row_id))
#                 updated_count += update_cursor.rowcount
#             except Exception as e:
#                 error_count += 1
#                 task_logger.error(f"Error re-assigning topic for DB ID {row_id}: {e}", exc_info=False) # Keep log cleaner
#
#         conn.commit() # Commit all updates
#         task_logger.info(f"Finished re-assigning topics. Updated: {updated_count}, Errors: {error_count}")
#
#     except sqlite3.Error as e:
#         task_logger.error(f"Database error during topic update: {e}", exc_info=True)
#     except Exception as e:
#          task_logger.error(f"Unexpected error during topic update: {e}", exc_info=True)
#     finally:
#         database.close_db_connection(conn)

# --- Root endpoint for basic check ---
@app.get("/", summary="Health Check", description="Basic endpoint to check if the API is running.")
async def read_root():
    logger.info("Root endpoint accessed.")
    return {"status": "Brand Reputation API is running"}

# Note: If running with `uvicorn backend.main:app`, this __main__ block won't execute.
# It's useful for direct script execution testing, but uvicorn is the standard way to run FastAPI apps.
# if __name__ == "__main__":
#     import uvicorn
#     logger.info("Starting API server directly using uvicorn...")
#     uvicorn.run(app, host="0.0.0.0", port=8000)