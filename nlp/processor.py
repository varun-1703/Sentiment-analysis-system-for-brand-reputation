# nlp/processor.py
import logging
from transformers import pipeline
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import pandas as pd
import os
import sys

import config # Import the config file



# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Sentiment Analysis ---
# Load a pre-trained sentiment analysis pipeline from Hugging Face.
sentiment_pipeline = None # Initialize later in a try-except block
try:
    # Using a robust model suitable for social media
    # Note: First time running will download the model (may take time/bandwidth)
    logger.info("Loading sentiment analysis model...")
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        # device=0 # Uncomment this line to use GPU if available and PyTorch with CUDA is installed
    )
    logger.info("Sentiment analysis pipeline loaded successfully.")
except Exception as e:
    logger.error(f"Error loading sentiment pipeline: {e}. Sentiment analysis will not be available.", exc_info=True)
    # Consider exiting or providing a fallback if sentiment is critical


def analyze_sentiment(text: str) -> tuple[str, float]:
    """
    Analyzes the sentiment of a given text.

    Args:
        text: The input text string.

    Returns:
        A tuple containing the sentiment label ('positive', 'negative', 'neutral')
        and the confidence score. Returns ('error', 0.0) if analysis fails or model not loaded.
    """
    if not sentiment_pipeline:
        logger.error("Sentiment pipeline not available.")
        return "error", 0.0
    if not text or not isinstance(text, str) or len(text.strip()) == 0:
        logger.warning("Invalid or empty input text for sentiment analysis. Returning neutral.")
        return "neutral", 0.0 # Assign neutral to empty/invalid text

    try:
        # Limit text length for RoBERTa (max 512 tokens usually) - pipeline handles truncation
        # Be mindful of very long texts potentially losing context.
        results = sentiment_pipeline(text, truncation=True, max_length=510) # Use truncation
        if not results: # Should not happen with valid text, but check
             return "error", 0.0

        # The pipeline might return slightly different label names. Map them.
        raw_label = results[0]['label'].lower()
        score = results[0]['score']

        # Mapping based on cardiffnlp model output labels
        if raw_label == "positive" or "pos" in raw_label : # More robust check
            sentiment = "positive"
        elif raw_label == "negative" or "neg" in raw_label:
            sentiment = "negative"
        else: # Covers 'neutral' etc.
            sentiment = "neutral"

        return sentiment, score
    except Exception as e:
        # Log the specific text snippet that caused the error if possible (be mindful of PII)
        logger.error(f"Error during sentiment analysis for text snippet '{text[:100]}...': {e}", exc_info=True)
        return "error", 0.0

# --- Topic Modeling ---
# Use the path from config.py
MODEL_SAVE_PATH = config.NLP_MODEL_SAVE_PATH
topic_model = None # Initialize later
is_topic_model_fitted = False # Global flag

def get_topic_model():
    """Loads or creates the BERTopic model."""
    global topic_model, is_topic_model_fitted # Allow modification of global state
    if topic_model is not None:
        # Model already loaded in memory
        # Check if it's fitted (might have been loaded but not fitted, or fitted later)
        is_topic_model_fitted = hasattr(topic_model, 'topic_embeddings_') and topic_model.topic_embeddings_ is not None
        return topic_model

    # Check if a saved model exists
    if os.path.exists(MODEL_SAVE_PATH) and os.listdir(MODEL_SAVE_PATH): # Check if directory is not empty
        try:
            logger.info(f"Loading saved BERTopic model from {MODEL_SAVE_PATH}...")
            # Ensure the directory exists before loading
            os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
            topic_model = BERTopic.load(MODEL_SAVE_PATH)
            is_topic_model_fitted = hasattr(topic_model, 'topic_embeddings_') and topic_model.topic_embeddings_ is not None
            logger.info(f"BERTopic model loaded. Fitted: {is_topic_model_fitted}")
            return topic_model
        except Exception as e:
            logger.error(f"Error loading saved model from {MODEL_SAVE_PATH}: {e}. Creating a new one.", exc_info=True)
            # Fall through to create a new model if loading fails

    logger.info("Creating a new BERTopic model instance...")
    try:
        # Using a standard Sentence Transformer model. Larger models might yield better topics but are slower.
        # 'all-MiniLM-L6-v2' is a good default.
        embedding_model_name = "all-MiniLM-L6-v2"
        logger.info(f"Loading sentence transformer model: {embedding_model_name}")
        # Specify cache folder for sentence_transformers if desired, otherwise uses default ~/.cache/torch/sentence_transformers
        # embedding_model = SentenceTransformer(embedding_model_name, cache_folder=os.path.join(config.BASE_DIR, "st_cache"))
        embedding_model = SentenceTransformer(embedding_model_name)
        logger.info("Sentence transformer model loaded.")

        # nr_topics="auto" lets BERTopic determine the optimal number of topics.
        # min_topic_size: Adjust based on expected data volume. Too low = noisy topics, too high = missed topics.
        # calculate_probabilities=True is needed for topic assignment later if not using .transform
        topic_model = BERTopic(
            embedding_model=embedding_model,
            nr_topics="auto",         # Automatically determine number of topics
            min_topic_size=5,         # Minimum number of documents required for a topic
            calculate_probabilities=False, # Set to False for faster transform, True if probabilities needed
            verbose=True              # Print progress messages
        )
        is_topic_model_fitted = False # New model is not fitted yet
        logger.info("New BERTopic model instance created.")
    except Exception as e:
         logger.error(f"Error creating new BERTopic model: {e}", exc_info=True)
         topic_model = None # Ensure it's None if creation fails
         is_topic_model_fitted = False

    return topic_model

def train_topic_model(texts: list[str]):
    """
    Fits the BERTopic model on the provided texts and saves it.
    """
    global topic_model, is_topic_model_fitted
    if topic_model is None:
         logger.error("Topic model instance is not available. Cannot train.")
         return False

    if not texts or not isinstance(texts, list) or len(texts) < topic_model.min_topic_size:
        logger.warning(f"Not enough texts ({len(texts)}) provided for topic model training (min_topic_size={topic_model.min_topic_size}). Training skipped.")
        return False

    try:
        logger.info(f"Fitting BERTopic model on {len(texts)} documents...")
        # fit_transform fits the model and assigns topics to the training data.
        # This can take significant time and memory depending on data size and model.
        topics, probs = topic_model.fit_transform(texts) # Returns topics and probabilities for the input texts
        is_topic_model_fitted = True
        logger.info(f"BERTopic model fitted successfully. Found {len(topic_model.get_topic_info()) - 1} topics (excluding outliers).") # -1 for outlier topic

        # Save the fitted model for later use
        try:
            # Ensure the directory exists before saving
            os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
            topic_model.save(MODEL_SAVE_PATH, serialization="pickle") # Or 'safetensors' if preferred & deps installed
            logger.info(f"BERTopic model saved to {MODEL_SAVE_PATH}")
        except Exception as e:
            # Log error but don't necessarily fail the whole operation if saving fails
            logger.error(f"Error saving BERTopic model to {MODEL_SAVE_PATH}: {e}", exc_info=True)

        return True
    except Exception as e:
        logger.error(f"Error fitting BERTopic model: {e}", exc_info=True)
        is_topic_model_fitted = False # Reset flag if fitting fails
        return False

def get_topic(text: str) -> int:
    """
    Assigns a topic ID to a single new text document using the fitted model.
    Returns the topic ID or -1 if model not fitted/available or error occurs.
    """
    global topic_model, is_topic_model_fitted
    if topic_model is None or not is_topic_model_fitted:
        # Log only once or less frequently if this becomes noisy
        logger.debug("Topic model not available or not fitted. Cannot assign topic.")
        # Attempt to load/get model again - maybe it was fitted by another process/request
        loaded_model = get_topic_model()
        if loaded_model is None or not is_topic_model_fitted: # Re-check flag after get_topic_model call
             return -1 # Indicate outlier/unassigned topic

    if not text or not isinstance(text, str) or len(text.strip()) == 0:
        logger.warning("Invalid or empty input text for topic assignment.")
        return -1 # Outlier topic for invalid input

    try:
        # Use transform for assigning topics to new documents.
        # It expects a list of documents.
        # This is generally faster than fit_transform after the model is fitted.
        topics, _ = topic_model.transform([text]) # Pass text as a list
        return topics[0] if topics else -1 # Return the topic ID for the first (only) document
    except Exception as e:
        # Handle potential errors, e.g., if the text is very different or causes embedding issues
        logger.error(f"Error assigning topic to text snippet '{text[:100]}...': {e}", exc_info=False) # Keep log concise
        return -1 # Assign to outlier topic in case of error

def get_topic_info() -> pd.DataFrame:
    """
    Returns information about the topics found by the model as a Pandas DataFrame.
    Returns an empty DataFrame if model not fitted/available or error occurs.
    """
    global topic_model, is_topic_model_fitted
    if topic_model is None or not is_topic_model_fitted:
        logger.debug("Topic model not available or not fitted. Cannot get topic info.")
        # Attempt to load/get model again
        loaded_model = get_topic_model()
        if loaded_model is None or not is_topic_model_fitted:
            return pd.DataFrame() # Return empty DataFrame

    try:
        # Get the topic information DataFrame from BERTopic
        # This includes Topic ID, Count, Name (keywords), Representation, etc.
        # Filter out the outlier topic (-1) info if desired for cleaner display later.
        topic_info_df = topic_model.get_topic_info() #.query("Topic != -1")
        if topic_info_df is None: # Should return df, but check just in case
             logger.warning("topic_model.get_topic_info() returned None.")
             return pd.DataFrame()

        logger.info(f"Retrieved topic info for {len(topic_info_df)} topics (including outliers).")
        return topic_info_df
    except Exception as e:
        logger.error(f"Error retrieving topic info: {e}", exc_info=True)
        return pd.DataFrame() # Return empty DataFrame on error

# --- Initial Load Trigger ---
# Ensure models are loaded/checked when this module is imported.
get_topic_model() # Load/create topic model instance
# Sentiment model loading is handled in its try-except block above.