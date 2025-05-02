# backend/database.py
import sqlite3
import logging
from datetime import datetime, timezone
from typing import List, Optional, Tuple, Any
import sys # Keep sys import
import os # Keep os import

# --- START DEBUG ---
# print("--- DEBUG: Inside database.py ---")
# print(f"Current working directory: {os.getcwd()}")
# print(f"Absolute path of database.py: {os.path.abspath(__file__)}")
# Check if project root is REALLY in sys.path at this point
# project_root_check = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# print(f"Calculated project root from database.py: {project_root_check}")
# print(f"Is project root in sys.path? {project_root_check in sys.path}")
# print("Current sys.path:")
# for p in sys.path:
#     print(f"  - {p}")
# --- END DEBUG ---

# Import config directly, relying on main.py having set the path
import config

# --- Use config directly ---
DATABASE_NAME = config.DATABASE_NAME
# if not config: # No longer needed
#      print("WARNING: config module failed to import in database.py! Using fallback DB name.")


logger = logging.getLogger(__name__)


def get_db_connection() -> Optional[sqlite3.Connection]:
    """ Establishes a connection to the SQLite database. Returns None on failure. """
    try:
        # PARSE_DECLTYPES allows automatic conversion of declared types (e.g., DATETIME)
        # PARSE_COLNAMES allows using column names from SELECT statements
        conn = sqlite3.connect(DATABASE_NAME, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
        conn.row_factory = sqlite3.Row # Return rows that behave like dictionaries
        # Enable Write-Ahead Logging for better concurrency (recommended for FastAPI)
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn
    except sqlite3.Error as e:
        logger.error(f"Database connection error to {DATABASE_NAME}: {e}", exc_info=True)
        return None

def close_db_connection(conn: Optional[sqlite3.Connection]):
    """ Closes the database connection if it's open. """
    if conn:
        try:
            conn.close()
        except sqlite3.Error as e:
            logger.error(f"Error closing database connection: {e}", exc_info=True)


def create_table():
    """ Creates the 'analyzed_texts' table if it doesn't exist with appropriate columns and constraints. """
    conn = get_db_connection()
    if not conn:
        return # Cannot proceed without connection

    try:
        cursor = conn.cursor()
        # Define the table schema
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analyzed_texts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,                 -- The actual text content
                source TEXT,                       -- e.g., 'Reddit Submission', 'NewsAPI - BBC', 'RSS - TechCrunch'
                source_unique_id TEXT UNIQUE NOT NULL, -- URL or API-specific ID to prevent duplicates
                brand_keyword TEXT,                -- The keyword that found this item
                timestamp DATETIME NOT NULL,         -- Parsed publication/post timestamp (stored as ISO format text or timestamp)
                fetch_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, -- When we fetched it (SQLite handles default)
                sentiment_label TEXT NOT NULL,     -- 'positive', 'negative', 'neutral', 'error'
                sentiment_score REAL NOT NULL,     -- Confidence score from model
                topic_id INTEGER NOT NULL          -- ID assigned by BERTopic (-1 for outliers/unassigned)
            )
        """)
        # Create an index on source_unique_id for faster duplicate checks and lookups
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_source_unique_id ON analyzed_texts (source_unique_id)")
        # Optional: Index on timestamp for faster time-based queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON analyzed_texts (timestamp)")
        # Optional: Index on brand_keyword if filtering by it frequently
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_brand_keyword ON analyzed_texts (brand_keyword)")

        conn.commit()
        logger.info("Database table 'analyzed_texts' checked/created successfully.")
    except sqlite3.Error as e:
        logger.error(f"Database error during table creation: {e}", exc_info=True)
    finally:
        close_db_connection(conn)


def check_if_exists(source_unique_id: str) -> bool:
    """ Checks if a record with the given source_unique_id already exists. """
    conn = get_db_connection()
    if not conn:
        return False # Assume doesn't exist if DB connection fails

    exists = False
    try:
        cursor = conn.cursor()
        # Use EXISTS for efficiency - stops searching after finding one match
        cursor.execute("SELECT EXISTS(SELECT 1 FROM analyzed_texts WHERE source_unique_id = ? LIMIT 1)", (source_unique_id,))
        result = cursor.fetchone()
        # fetchone() returns a tuple (e.g., (1,) or (0,)) or None if query fails
        exists = result[0] == 1 if result else False
    except sqlite3.Error as e:
        logger.error(f"Database error while checking existence for {source_unique_id}: {e}", exc_info=False) # Less verbose log
    finally:
        close_db_connection(conn)
    # logger.debug(f"Check exists for {source_unique_id}: {exists}") # Optional debug log
    return exists


def add_analyzed_text(text: str, source: Optional[str], source_unique_id: str, brand_keyword: Optional[str],
                        timestamp: datetime, sentiment_label: str, sentiment_score: float,
                        topic_id: int) -> Optional[int]:
    """
    Adds a new record with analyzed text data to the database.
    Returns the new record's ID if successful, None otherwise.
    Handles potential duplicate entries gracefully.
    """
    # Basic validation
    if not source_unique_id:
        logger.error("Cannot add record: source_unique_id is missing.")
        return None
    if not text:
         logger.warning(f"Attempting to add record with empty text for source_id {source_unique_id}. Proceeding, but check source.")


    # Check for duplicates before attempting insert
    if check_if_exists(source_unique_id):
        logger.debug(f"Record with source_unique_id '{source_unique_id}' already exists. Skipping insert.")
        return None

    conn = get_db_connection()
    if not conn:
        return None # Cannot proceed

    last_id = None
    try:
        cursor = conn.cursor()
        # Ensure timestamp is timezone-aware (UTC recommended) before storing
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        # Convert datetime to ISO 8601 string format for SQLite storage
        # SQLite doesn't have a native DATETIME type, stores as TEXT, REAL, or INTEGER. TEXT is standard.
        timestamp_str = timestamp.isoformat()

        cursor.execute("""
            INSERT INTO analyzed_texts
            (text, source, source_unique_id, brand_keyword, timestamp, sentiment_label, sentiment_score, topic_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            text,
            source if source else "Unknown", # Provide default if None
            source_unique_id,
            brand_keyword if brand_keyword else "N/A", # Provide default if None
            timestamp_str, # Store as ISO string
            sentiment_label,
            sentiment_score,
            topic_id
            )
        )
        conn.commit()
        last_id = cursor.lastrowid
        logger.info(f"Added analyzed text record ID: {last_id} for source ID: {source_unique_id}")
    except sqlite3.IntegrityError:
        # This handles the rare race condition where check_if_exists passed, but another process inserted concurrently
        logger.warning(f"Record with source_unique_id '{source_unique_id}' was likely inserted concurrently. Skipping.")
    except sqlite3.Error as e:
        logger.error(f"Database error while adding text for {source_unique_id}: {e}", exc_info=True)
    finally:
        close_db_connection(conn)
    return last_id


def get_all_analyzed_data(limit: int = 100, offset: int = 0) -> List[sqlite3.Row]:
    """ Retrieves analyzed text data from the database with pagination, ordered by timestamp descending. """
    conn = get_db_connection()
    if not conn:
        return []

    rows = []
    try:
        cursor = conn.cursor()
        # Select all relevant columns
        # SQLite stores DATETIME as TEXT, it will be returned as string. Parsing needed later.
        cursor.execute("""
            SELECT id, text, source, source_unique_id, brand_keyword,
                   timestamp, fetch_timestamp, -- These will be strings
                   sentiment_label, sentiment_score, topic_id
            FROM analyzed_texts
            ORDER BY timestamp DESC
            LIMIT ? OFFSET ?
        """, (limit, offset))
        rows = cursor.fetchall()
        logger.info(f"Retrieved {len(rows)} records from database (limit={limit}, offset={offset}).")
    except sqlite3.Error as e:
        logger.error(f"Database error while retrieving data: {e}", exc_info=True)
    finally:
        close_db_connection(conn)
    return rows # Returns list of Row objects


def get_all_texts_for_topic_model() -> List[str]:
    """ Retrieves only the text column for all entries, useful for topic model training. """
    conn = get_db_connection()
    if not conn:
        return []

    texts = []
    try:
        cursor = conn.cursor()
        # Fetching potentially large amounts of text, consider memory usage
        # Might add a LIMIT or filter by recent time if dataset grows very large
        cursor.execute("SELECT text FROM analyzed_texts ORDER BY timestamp DESC")
        rows = cursor.fetchall()
        # Extract the text from each Row object
        texts = [row['text'] for row in rows if row['text']] # Ensure text is not None or empty
        logger.info(f"Retrieved {len(texts)} non-empty texts for topic modeling.")
    except sqlite3.Error as e:
        logger.error(f"Database error retrieving texts for topic model: {e}", exc_info=True)
    finally:
        close_db_connection(conn)
    return texts

# --- Initialization ---
# Ensure the table exists when this module is imported.
# This is generally safe for SQLite as it's lightweight.
# For heavier DBs, might do this in startup script or manually.
create_table()