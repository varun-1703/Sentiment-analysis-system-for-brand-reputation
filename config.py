# config.py
import os
import logging

# --- Custom .env file parser to avoid python-dotenv dependency ---
def manual_load_env(dotenv_path):
    """Simple .env file parser that doesn't require python-dotenv"""
    if not os.path.exists(dotenv_path):
        return False
    
    try:
        with open(dotenv_path, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                # Parse VAR=VALUE format
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    # Remove quotes if present
                    if value and value[0] == value[-1] and value[0] in ['"', "'"]:
                        value = value[1:-1]
                    # Only set if not already in environment
                    if key and not os.environ.get(key):
                        os.environ[key] = value
        return True
    except Exception as e:
        print(f"Error reading .env file: {e}")
        return False

# Try to import dotenv, but fall back to our manual implementation
try:
    from dotenv import load_dotenv as _load_dotenv_func
    _use_manual_load = False
except ImportError:
    print("Python-dotenv not available, using manual .env file parser.")
    _load_dotenv_func = manual_load_env
    _use_manual_load = True

# Load environment variables from .env file in the project root
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(dotenv_path):
    _load_dotenv_func(dotenv_path)
else:
    # Try loading .env from parent if config.py is not in root (e.g. running script from subdir)
     dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
     if os.path.exists(dotenv_path):
         _load_dotenv_func(dotenv_path)
     else:
         print("Warning: .env file not found.")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- API Keys ---
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT")
REDDIT_USERNAME = os.getenv("REDDIT_USERNAME")
REDDIT_PASSWORD = os.getenv("REDDIT_PASSWORD")

# --- Validation (Optional but Recommended) ---
if not NEWS_API_KEY:
    logger.warning("NEWS_API_KEY not found in environment variables or .env file.")
if not all([REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT, REDDIT_USERNAME, REDDIT_PASSWORD]):
    logger.warning("One or more Reddit credentials not found in environment variables or .env file.")

# --- Other Settings ---
# Construct absolute path for the database relative to this config file's location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_NAME = os.path.join(BASE_DIR, "brand_reputation.db")
NLP_MODEL_SAVE_PATH = os.path.join(BASE_DIR, "nlp", "topic_model_cache")

# Max results per source during one fetch operation (adjust based on API limits/needs)
MAX_RESULTS_REDDIT = 25
MAX_RESULTS_NEWS = 20 # NewsAPI free tier often limited to page size
MAX_RESULTS_RSS = 50 # Per feed

# Ensure NLP model save directory exists
os.makedirs(NLP_MODEL_SAVE_PATH, exist_ok=True)

# Backend URL (primarily for dashboard)
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

logger.info(f"Database path configured: {DATABASE_NAME}")
logger.info(f"NLP model cache path configured: {NLP_MODEL_SAVE_PATH}")