# backend/models.py
from pydantic import BaseModel, Field, HttpUrl, validator
from datetime import datetime
from typing import Optional, List

# --- Input Models (Requests to API) ---

class CollectRequest(BaseModel):
    """ Request model for triggering data collection via the /collect endpoint. """
    brand_keyword: str = Field(..., min_length=1, description="Brand name or keyword to search for")
    # Allow URLs as strings initially, validate format later if strict typing needed
    rss_urls: Optional[List[str]] = Field(None, description="Optional list of RSS feed URLs (as strings)")

    # Example of custom validator if strict HttpUrl validation is desired for RSS
    # @validator('rss_urls', each_item=True, pre=True, allow_reuse=True)
    # def check_url_format(cls, v):
    #     if isinstance(v, str):
    #         # Basic check, Pydantic's HttpUrl is more robust
    #         if not v.startswith(('http://', 'https://')):
    #              raise ValueError("RSS URL must start with http:// or https://")
    #     return v


# --- Output Models (Responses from API) ---

class AnalyzedDataResponse(BaseModel):
    """ Pydantic model for representing analyzed data sent FROM the API (e.g., to the dashboard). """
    id: int
    text: str
    source: Optional[str] = None # Make optional as DB allows NULL if not provided
    source_unique_id: str
    brand_keyword: Optional[str] = None
    timestamp: datetime # Pydantic handles string parsing from DB/JSON to datetime
    fetch_timestamp: datetime
    sentiment_label: str
    sentiment_score: float
    topic_id: int

    class Config:
        # orm_mode = True # Enable reading data directly from ORM models or Row objects
        # If using Pydantic v2, orm_mode is replaced by from_attributes = True
        from_attributes = True


class TopicInfoResponse(BaseModel):
    """ Pydantic model for topic details sent FROM the API. """
    # Define fields based on columns from BERTopic's get_topic_info() DataFrame
    Topic: int # Matches DataFrame column name
    Count: int # Matches DataFrame column name
    Name: str  # Matches DataFrame column name
    Representation: Optional[List[str]] = None # Optional field if needed
    Representative_Docs: Optional[List[str]] = None # Optional field

    class Config:
        # orm_mode = True
        from_attributes = True # Pydantic v2
        # Allow population by field name if input dict keys match DataFrame cols exactly
        # allow_population_by_field_name = True # Not needed if aliases aren't used


class StatusResponse(BaseModel):
    """ Generic response model for status messages (e.g., background task accepted). """
    message: str
    details: Optional[str] = None

class ErrorResponse(BaseModel):
    """ Generic response model for errors. """
    detail: str