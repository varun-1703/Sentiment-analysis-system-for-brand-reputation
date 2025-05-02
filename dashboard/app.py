# dashboard/app.py
import dash
from dash import dcc, html, Input, Output, State, callback, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import requests # To call the backend API
import logging
from datetime import datetime
import json # For handling store data if needed
import os
from typing import Optional, Dict, Any, List  # Add typing imports
import base64
from io import BytesIO
import re
from collections import Counter
import numpy as np

# Add wordcloud and PIL for word cloud generation
try:
    from wordcloud import WordCloud
    from PIL import Image
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False
    print("WordCloud library not available. Install with: pip install wordcloud pillow")

# --- Backend API URL ---
# Get from environment variable or use default localhost (matches backend default)
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
print(f"Dashboard configured to use backend at: {BACKEND_URL}")

# --- Configure Logging ---
# Basic config for dashboard messages
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Logger specific to the dashboard app

# --- Initialize Dash App ---
# Use Bootstrap themes for better styling and layout control.
# https://dash-bootstrap-components.opensource.faculty.ai/docs/themes/
# CYBORG, DARKLY, LUMEN, SANDSTONE are good options.
app = dash.Dash(__name__,
                external_stylesheets=[dbc.themes.DARKLY],  # Changed to DARKLY for better visuals
                suppress_callback_exceptions=True, # Needed if callbacks target components created by other callbacks
                title="Brand Reputation Monitor")

# --- Helper Functions ---

def fetch_data_from_api(endpoint: str, params: Optional[Dict] = None) -> Optional[Any]:
    """ Helper to perform GET requests to the backend API. Returns JSON data or None. """
    # Make sure endpoint starts with / but doesn't include the base URL
    if not endpoint.startswith('/'):
        endpoint = '/' + endpoint
        
    url = f"{BACKEND_URL}{endpoint}"
    try:
        logger.info(f"Fetching data from: {url} with params: {params}")
        response = requests.get(url, params=params, timeout=30) # 30-second timeout for GET
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        data = response.json()
        logger.info(f"API GET successful for {url}. Status: {response.status_code}. Received {len(data) if isinstance(data, list) else 'non-list'} data.")
        return data
    except requests.exceptions.Timeout:
        logger.error(f"API GET request timed out for {url}")
        return {"error": "Request timed out", "status_code": 408} # Simulate error structure
    except requests.exceptions.ConnectionError:
         logger.error(f"API GET request connection error for {url}. Is backend running?")
         return {"error": "Connection error - is backend running?", "status_code": 503}
    except requests.exceptions.HTTPError as e:
        # Attempt to get error detail from response JSON if possible
        error_detail = e.response.text # Default to raw text
        try:
             error_json = e.response.json()
             error_detail = error_json.get('detail', error_detail)
        except ValueError: # If response is not JSON
             pass
        logger.error(f"API GET request failed for {url}: Status {e.response.status_code}, Detail: {error_detail}")
        return {"error": error_detail, "status_code": e.response.status_code}
    except requests.exceptions.RequestException as e:
        logger.error(f"API GET request failed for {url} with unexpected error: {e}")
        return {"error": str(e), "status_code": 500} # Generic error code
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON response from {url}: {e}. Response text: {response.text[:200]}")
        return {"error": "Invalid JSON response from server", "status_code": 500}


def post_to_api(endpoint: str, json_data: Dict) -> Dict:
    """ Helper to perform POST requests to the backend API. Returns dict (response JSON or error info). """
    url = f"{BACKEND_URL}{endpoint}"
    try:
        logger.debug(f"Posting data to: {url} with payload: {json_data}")
        response = requests.post(url, json=json_data, timeout=60) # Longer timeout for POST (background tasks)

        # Check for specific success/accepted codes first
        if response.status_code in [200, 201, 202]:
             logger.info(f"API POST successful for {url}. Status: {response.status_code}")
             try:
                 # Return JSON response if available, otherwise construct success message
                 return response.json()
             except json.JSONDecodeError:
                 return {"message": response.text or f"Request accepted (Status {response.status_code})"}
        else:
            # Handle non-success status codes as errors
            response.raise_for_status() # Raise HTTPError for other non-2xx codes

    except requests.exceptions.Timeout:
        logger.error(f"API POST request timed out for {url}")
        return {"error": "Request timed out", "status_code": 408}
    except requests.exceptions.ConnectionError:
         logger.error(f"API POST request connection error for {url}. Is backend running?")
         return {"error": "Connection error - is backend running?", "status_code": 503}
    except requests.exceptions.HTTPError as e:
        error_detail = e.response.text
        try:
             error_json = e.response.json()
             error_detail = error_json.get('detail', error_detail)
        except ValueError:
             pass
        logger.error(f"API POST request failed for {url}: Status {e.response.status_code}, Detail: {error_detail}")
        return {"error": error_detail, "status_code": e.response.status_code}
    except requests.exceptions.RequestException as e:
        logger.error(f"API POST request failed for {url} with unexpected error: {e}")
        return {"error": str(e), "status_code": 500}

    # Fallback return if raise_for_status() wasn't hit but status not 2xx (shouldn't happen often)
    return {"error": f"Unexpected non-success status: {response.status_code}", "status_code": response.status_code}


# --- App Layout Definition ---
app.layout = dbc.Container([
    # -- Header --
    dbc.Row(
        dbc.Col(html.H1(app.title, className="text-center text-primary mb-4 mt-4"), width=12)
    ),

    # -- Control Panel Row --
    dbc.Card(dbc.CardBody([
        dbc.Row([
            # Brand/Keyword Input
            dbc.Col([
                dbc.Label("Brand/Keyword:", html_for="brand-keyword-input", className="fw-bold"),
                dbc.Input(id="brand-keyword-input", placeholder="Enter brand or keyword...", type="text", value="Example", className="mb-2"),
            ], md=4), # Medium screen 4 columns wide

            # RSS Feed Input
            dbc.Col([
                 dbc.Label("RSS Feed URLs (one per line):", html_for="rss-urls-input", className="fw-bold"),
                 dbc.Textarea(id="rss-urls-input", placeholder="https://example.com/feed\nhttps://another.org/rss", style={'height': '80px'}, className="mb-2"),
            ], md=4),

            # Action Buttons
            dbc.Col([
                dbc.Label("Actions:", className="fw-bold"), html.Br(),
                dbc.ButtonGroup([
                    dbc.Button("Fetch New Data", id="fetch-data-button", color="primary", className="me-2"),
                    dbc.Button("Retrain Topics", id="retrain-topics-button", color="warning", className="me-2", title="Retrain topic model using all data in DB"),
                    dbc.Button("Refresh View", id="refresh-view-button", color="info", outline=True, title="Manually reload data from backend"),
                ], className="mb-2"),
            ], md=4, className="d-flex flex-column align-items-start justify-content-end"), # Align content lower

        ], className="align-items-end"), # Align row items vertically at the bottom
    ]), className="mb-4"), # Add margin below the card

     # -- Status Message Area --
     # Use Loading component to wrap the alert for better UX during background calls
     dcc.Loading(id="loading-status", type="default", children=[
         dbc.Alert(
             id="status-alert",
             color="info",
             dismissable=True,
             is_open=False,
             duration=5000, # Auto-dismiss after 5 seconds for success messages
             className="mt-2 mb-3" # Add some spacing
         )
     ]),

    # Add a trigger for auto-loading data on page load
    html.Div(id='auto-load-trigger', style={'display': 'none'}),

    # -- Data Display Tabs --
    dbc.Tabs([
        # Overview Tab (Charts)
        dbc.Tab(label="Overview", tab_id="tab-overview", children=[
            dbc.Row([
                dbc.Col(dcc.Loading(dcc.Graph(id="sentiment-pie-chart")), lg=6, className="mt-4"), # Large screen 6 columns
                dbc.Col(dcc.Loading(dcc.Graph(id="sentiment-time-series")), lg=6, className="mt-4"),
            ], className="mt-2"),
            dbc.Row([
                 dbc.Col(dcc.Loading(dcc.Graph(id="topic-bar-chart")), lg=12, className="mt-4"),
            ], className="mt-2"),
            # New row for word cloud
            dbc.Row([
                dbc.Col(dbc.Card([
                    dbc.CardHeader("Top Terms Word Cloud", className="text-center"),
                    dbc.CardBody([
                        dcc.Loading(html.Img(id="word-cloud-image", style={'width': '100%'})),
                        html.Div(
                            id="wordcloud-missing-message",
                            children="Install wordcloud with 'pip install wordcloud pillow' to enable this visualization",
                            style={'display': 'none' if WORDCLOUD_AVAILABLE else 'block', 'textAlign': 'center', 'padding': '20px', 'color': '#aaa'}
                        )
                    ])
                ]), lg=12, className="mt-4"),
            ], className="mt-2")
        ]),

        # Data Table Tab
        dbc.Tab(label="Data Table", tab_id="tab-data", children=[
            dbc.Row(dbc.Col(
                dcc.Loading(
                    dash_table.DataTable(
                        id='data-table',
                        # Define columns dynamically in callback for flexibility or explicitly here
                        columns=[
                            {"name": "Time", "id": "timestamp_str", "type": "datetime"},
                            {"name": "Source", "id": "source", "type": "text"},
                            {"name": "Sentiment", "id": "sentiment_label", "type": "text"},
                            # {"name": "Score", "id": "sentiment_score_str", "type": "numeric"},
                            {"name": "Topic", "id": "topic_display", "type": "text"}, # Combined Topic ID + Name
                            {"name": "Keyword", "id": "brand_keyword", "type": "text"},
                            {"name": "Text Snippet", "id": "text_short", "type": "text"},
                            # Hidden columns for full data access if needed or tooltips
                            {"name": "Full Text", "id": "text", "hidden": True},
                            {"name": "Source ID", "id": "source_unique_id", "hidden": True},
                            {"name": "Topic ID Raw", "id": "topic_id", "hidden": True},
                        ],
                        data=[], # Initial empty data
                        page_size=20, # Number of rows per page
                        style_table={'overflowX': 'auto', 'minWidth': '100%'},
                        style_header={
                            'backgroundColor': 'rgb(30, 30, 30)',
                            'color': 'white',
                            'fontWeight': 'bold',
                            'border': '1px solid #444'
                        },
                        style_cell={
                            'textAlign': 'left',
                            'padding': '8px',
                            'minWidth': '100px', 'width': '150px', 'maxWidth': '400px', # Adjust widths
                            'overflow': 'hidden',
                            'textOverflow': 'ellipsis',
                            'whiteSpace': 'normal', # Allow wrapping in cells
                            'height': 'auto',      # Adjust height for wrapped text
                            'fontFamily': 'Arial, sans-serif',
                            'fontSize': '13px',
                            'backgroundColor': 'rgb(50, 50, 50)',
                            'color': 'white',
                            'border': '1px solid #444'
                        },
                        style_cell_conditional=[ # Fine-tune specific column widths
                             {'if': {'column_id': 'timestamp_str'}, 'width': '150px', 'minWidth': '130px', 'maxWidth': '160px'},
                             {'if': {'column_id': 'sentiment_label'}, 'width': '100px', 'minWidth': '90px', 'maxWidth': '120px'},
                             {'if': {'column_id': 'topic_display'}, 'width': '180px', 'minWidth': '150px', 'maxWidth': '300px'},
                             {'if': {'column_id': 'text_short'}, 'width': '400px', 'minWidth': '250px', 'maxWidth': '600px'},
                        ],
                        sort_action="native", # Enable backend sorting (if implemented) or "native" frontend sorting
                        filter_action="native", # Enable frontend filtering
                        row_selectable='multi', # Enable row selection for comparison
                        row_deletable=False,
                        tooltip_data=[], # Updated in callback
                        tooltip_duration=None, # Keep tooltip open
                        style_data_conditional=[ # Color rows by sentiment
                             {
                                 'if': {'filter_query': '{sentiment_label} = "positive"'},
                                 'backgroundColor': 'rgba(40, 167, 69, 0.2)', # Green with alpha
                                 'color': '#28a745'
                             },
                             {
                                 'if': {'filter_query': '{sentiment_label} = "negative"'},
                                 'backgroundColor': 'rgba(220, 53, 69, 0.2)', # Red with alpha
                                 'color': '#dc3545'
                             },
                             {
                                 'if': {'filter_query': '{sentiment_label} = "neutral"'},
                                 'backgroundColor': 'rgba(108, 117, 125, 0.2)', # Grey with alpha
                                 'color': '#6c757d'
                             },
                             # Highlight selected rows
                             {
                                 'if': {'state': 'selected'},
                                 'backgroundColor': 'rgba(0, 123, 255, 0.2)',
                                 'border': '1px solid #007bff',
                             }
                         ]
                    ) # End DataTable
                ), # End Loading
            width=12), className="mt-4") # End Row/Col
        ]), # End Data Table Tab

         # Topics Tab
         dbc.Tab(label="Topic Details", tab_id="tab-topics", children=[
            dbc.Row(dbc.Col(
                dcc.Loading(html.Div(id='topic-details-table')), # Wrap the output Div in Loading
            width=12), className="mt-4")
         ]),

    ]), # End Tabs

    # --- Hidden Stores for Data ---
    # Store data in browser's memory - suitable for moderate amounts of data
    # For very large data, consider server-side caching or different architecture
    dcc.Store(id='api-status-store'),    # Stores messages {message, error, status_code} from API calls
    dcc.Store(id='main-data-store'),      # Stores main analyzed data (list of dicts)
    dcc.Store(id='topic-data-store'),     # Stores topic info (list of dicts)

    # --- Interval for Periodic Refresh (Optional) ---
    # dcc.Interval(
    #     id='interval-component',
    #     interval=5*60*1000, # Refresh every 5 minutes (e.g., 300000 ms)
    #     n_intervals=0,
    #     disabled=True # Start disabled, enable if needed
    # ),

], fluid=True) # Use fluid container for full width responsiveness

# --- Callbacks ---

# Callback 1: Handle Action Buttons (Fetch Data, Retrain Topics)
@callback(
    Output('api-status-store', 'data'), # Output to status store
    # Inputs from buttons
    Input('fetch-data-button', 'n_clicks'),
    Input('retrain-topics-button', 'n_clicks'),
    # State needed for fetch action
    State('brand-keyword-input', 'value'),
    State('rss-urls-input', 'value'),
    prevent_initial_call=True # Don't run on page load
)
def handle_actions(fetch_clicks, retrain_clicks, keyword, rss_text):
    # Determine which button was clicked using dash.callback_context
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update # No button clicked

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    api_response = None # Initialize response variable

    if button_id == 'fetch-data-button':
        logger.info(f"Fetch button clicked. Keyword: '{keyword}'.")
        if not keyword or not keyword.strip():
            return {"error": "Please enter a Brand/Keyword before fetching.", "status_code": 400}

        # Process RSS URLs: split by newline, filter empty, strip whitespace
        rss_urls = []
        if rss_text:
            rss_urls = [url.strip() for url in rss_text.split('\n') if url.strip()]
            logger.info(f"Processed RSS URLs for fetch: {rss_urls}")

        payload = {"brand_keyword": keyword.strip(), "rss_urls": rss_urls}
        api_response = post_to_api("/collect", json_data=payload)

    elif button_id == 'retrain-topics-button':
        logger.info("Retrain topics button clicked.")
        api_response = post_to_api("/retrain-topic-model", json_data={}) # Empty payload

    else: # Should not happen with current inputs
        return dash.no_update

    # Return the response (success message or error dict) to the status store
    logger.debug(f"API response from action '{button_id}': {api_response}")
    return api_response


# Callback 2: Display Status Messages from API calls
@callback(
    Output('status-alert', 'children'),
    Output('status-alert', 'color'),
    Output('status-alert', 'is_open'),
    Output('status-alert', 'duration'), # Control auto-dismiss duration
    Input('api-status-store', 'data'), # Triggered when status store changes
    prevent_initial_call=True
)
def display_status_message(status_data):
    if not status_data or not isinstance(status_data, dict):
        return "", "info", False, None # Hide alert if no valid data

    message = status_data.get("message")
    error = status_data.get("error")
    status_code = status_data.get("status_code")
    details = status_data.get("details") # Optional extra info from API

    if error:
        color = "danger"
        msg_text = f"Error ({status_code or 'N/A'}): {error}"
        duration = None # Keep error messages open until dismissed
        logger.error(f"Dashboard displaying error: {msg_text}")
        return msg_text, color, True, duration
    elif message:
        color = "success" if status_code in [200, 201, 202] else "warning"
        msg_text = message
        if details:
            msg_text += f" ({details})"
        # Auto-dismiss success/info messages, keep warnings open longer/indefinitely?
        duration = 8000 if color == "success" else None
        logger.info(f"Dashboard displaying status: {msg_text}")
        return msg_text, color, True, duration
    else:
        # No message or error, hide the alert
        return "", "info", False, None


# Callback 3: Load/Refresh Data from Backend API
@callback(
     Output('main-data-store', 'data'),
     Output('topic-data-store', 'data'),
     Output('api-status-store', 'data', allow_duplicate=True), # Update status on load error
     Input('refresh-view-button', 'n_clicks'), # Manual refresh trigger
     # Trigger after fetch/retrain actions are acknowledged (indirectly via status update)
     # We listen to the *result* of the action in the status store.
     # A successful 'collect' or 'retrain' message could trigger a refresh.
     # Or use an Interval component. Let's use manual refresh for now.
     # Input('api-status-store', 'data') # Could cause loop if status updates frequently
     # Input('interval-component', 'n_intervals') # Triggered by timer if enabled
     prevent_initial_call=True # Changed to True to fix the duplicate callback error
)
def load_data(refresh_clicks):#, status_data, n_intervals):
    """ Loads main data and topic data from backend API. Triggered on load and manual refresh. """
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else 'initial load'
    logger.info(f"load_data triggered by: {trigger_id}")

    # Fetch main analyzed data - load a decent amount for dashboard view
    # Use helper which returns error dict on failure
    data_response = fetch_data_from_api("/data", params={"limit": 1000})
    # Fetch topic info
    topics_response = fetch_data_from_api("/topics")

    error_msg = None
    data_result = []
    topics_result = []

    # Check for errors from API calls
    if isinstance(data_response, dict) and data_response.get("error"):
         error_msg = data_response # Use the error dict from helper
         logger.error(f"Failed to load main data: {error_msg}")
    elif isinstance(data_response, list):
        data_result = data_response # Store successful data fetch
        logger.info(f"Successfully loaded {len(data_result)} data records from backend")
    else: # Unexpected response type
         error_msg = {"error": "Unexpected response format for main data", "status_code": 500}
         logger.error(error_msg["error"])


    if isinstance(topics_response, dict) and topics_response.get("error"):
        # Combine errors if both failed, otherwise use topic error
        if error_msg:
            error_msg["error"] += f" | Failed to load topics: {topics_response['error']}"
        else:
             error_msg = topics_response
        logger.error(f"Failed to load topics: {topics_response}")
    elif isinstance(topics_response, list):
        topics_result = topics_response
        logger.info(f"Successfully loaded {len(topics_result)} topics from backend")
    elif error_msg is None: # Only set topic error if main data load succeeded
         error_msg = {"error": "Unexpected response format for topic data", "status_code": 500}
         logger.error(error_msg["error"])


    # Store data as dicts (which they already are from JSON) for dcc.Store
    # Return error message to status store if any error occurred
    return data_result, topics_result, error_msg


# Callback 4: Update All Visualizations based on stored data
@callback(
    # Outputs to the Graph and Table components
    Output('sentiment-pie-chart', 'figure'),
    Output('sentiment-time-series', 'figure'),
    Output('topic-bar-chart', 'figure'),
    Output('data-table', 'data'),
    Output('data-table', 'tooltip_data'),
    Output('topic-details-table', 'children'), # Output to the Div containing the topic table
    Output('word-cloud-image', 'src'), # Add output for word cloud image
    # Inputs from the data stores
    Input('main-data-store', 'data'),
    Input('topic-data-store', 'data')
)
def update_visualizations(main_data_store, topic_data_store):
    """ Updates all graphs and tables when the data in stores changes. """

    # --- Handle Empty/Invalid Data ---
    if not main_data_store or not isinstance(main_data_store, list):
        logger.warning("No main data available in store to update visualizations.")
        # Return empty figures/tables with informative titles/messages
        empty_fig = go.Figure().update_layout(
            title="No Data Available", xaxis={'visible': False}, yaxis={'visible': False},
             annotations=[{"text": "Fetch data using the controls above.", "xref": "paper", "yref": "paper", "showarrow": False, "font": {"size": 14}}]
             )
        empty_topics_table = dbc.Alert("No topic data available.", color="info")
        return empty_fig, empty_fig, empty_fig, [], [], empty_topics_table, None

    # --- Prepare DataFrames ---
    try:
        df = pd.DataFrame(main_data_store)
        # Convert timestamp strings from JSON/store back to datetime objects
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        # Create formatted strings for table display
        df['timestamp_str'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
        df['sentiment_score_str'] = df['sentiment_score'].round(3).astype(str)
        # Create shortened text for table display (handle potential None)
        df['text_short'] = df['text'].fillna('').str.slice(0, 150) + '...'

    except Exception as e:
        logger.error(f"Error processing main data for visualization: {e}", exc_info=True)
        # Return empty state again if DataFrame processing fails
        empty_fig = go.Figure().update_layout(title="Error Processing Data")
        empty_topics_table = dbc.Alert("Error processing topic data.", color="danger")
        return empty_fig, empty_fig, empty_fig, [], [], empty_topics_table, None


    # --- Prepare Topic Data (if available) ---
    topics_df = pd.DataFrame()
    topic_map = {} # Dictionary to map topic_id to name
    if topic_data_store and isinstance(topic_data_store, list):
         try:
             topics_df = pd.DataFrame(topic_data_store)
             # Create a mapping from topic ID to its generated name for easier lookup
             if not topics_df.empty and 'Topic' in topics_df.columns and 'Name' in topics_df.columns:
                  topic_map = topics_df.set_index('Topic')['Name'].to_dict()
         except Exception as e:
              logger.error(f"Error processing topic data for visualization: {e}", exc_info=True)
              # Continue without topic names if processing fails


    # --- Create Topic Display Column in Main DataFrame ---
    # Use the map to get topic names, provide default if not found or if topic is -1
    df['topic_display'] = df['topic_id'].apply(
        lambda tid: f"{tid}: {topic_map.get(tid, 'Unknown Name')}" if tid != -1 else f"{tid}: Outlier/Unassigned"
    )


    # --- Generate Figures ---
    try:
        # 1. Sentiment Pie Chart
        sentiment_counts = df['sentiment_label'].value_counts()
        pie_fig = px.pie(
            sentiment_counts,
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            title='Overall Sentiment Distribution',
            color=sentiment_counts.index,
            color_discrete_map={'positive':'#28a745', 'negative':'#dc3545', 'neutral':'#6c757d', 'error': '#ffc107'}, # Bootstrap colors
            hole=0.3 # Make it a donut chart
        )
        pie_fig.update_layout(
            legend_title_text='Sentiment', 
            margin=dict(t=50, b=0, l=0, r=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            hoverlabel=dict(font_size=16, font_family="Arial")
        )


        # 2. Sentiment Time Series
        # Resample data by day (or hour 'H', week 'W') and count sentiments
        # Ensure index is datetime before resampling
        df_time_indexed = df.set_index('timestamp')
        df_resampled = df_time_indexed.resample('D')['sentiment_label'].value_counts().unstack(fill_value=0)
        time_fig = px.area( # Use area chart for better visualization of trends over time
            df_resampled,
            x=df_resampled.index,
            y=df_resampled.columns, # Plot each sentiment
            title='Sentiment Over Time (Daily Count)',
            labels={'value': 'Count', 'timestamp': 'Date', 'sentiment_label': 'Sentiment'},
            color_discrete_map={'positive':'#28a745', 'negative':'#dc3545', 'neutral':'#6c757d', 'error': '#ffc107'}
        )
        time_fig.update_layout(
            legend_title_text='Sentiment', 
            hovermode="x unified",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(20,20,20,0.8)',
            font=dict(color='white'),
            xaxis=dict(gridcolor='rgba(80,80,80,0.3)'),
            yaxis=dict(gridcolor='rgba(80,80,80,0.3)')
        )


        # 3. Topic Bar Chart
        # Use the 'topic_display' column we created
        topic_counts = df['topic_display'].value_counts().nlargest(20).sort_values(ascending=True) # Show top 20, sort for horizontal bar
        topic_fig = px.bar(
            topic_counts,
            x=topic_counts.values,
            y=topic_counts.index, # Use index (topic names) for y-axis
            orientation='h', # Horizontal bar chart
            title='Top 20 Topics by Mention Count',
            labels={'x': 'Number of Mentions', 'y': 'Topic'},
            color=topic_counts.values,  # Color bars by count
            color_continuous_scale=px.colors.sequential.Viridis  # Use Viridis color scale
        )
        topic_fig.update_layout(
            yaxis_title=None, 
            xaxis_title="Number of Mentions", 
            margin=dict(l=100),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(20,20,20,0.8)',
            font=dict(color='white'),
            xaxis=dict(gridcolor='rgba(80,80,80,0.3)'),
            yaxis=dict(
                categoryorder='total ascending',
                title=None, 
                gridcolor='rgba(80,80,80,0.3)'
            )
        )
        # Add interactive tooltips
        topic_fig.update_traces(hovertemplate='<b>%{y}</b><br>Count: %{x}<extra></extra>')


    except Exception as e:
         logger.error(f"Error generating figures: {e}", exc_info=True)
         # Return error state if figure generation fails
         error_fig = go.Figure().update_layout(title="Error Generating Chart")
         empty_topics_table = dbc.Alert("Error processing topic data.", color="danger")
         return error_fig, error_fig, error_fig, [], [], empty_topics_table, None


    # --- Prepare Data Table Output ---
    # Sort by timestamp descending for display
    table_data = df.sort_values('timestamp', ascending=False).to_dict('records')
    # Prepare tooltip data - show full text on hover over the snippet cell
    # Generate tooltips only for the 'text_short' column referencing the 'text' column data
    tooltip_data = [
        {
            'text_short': {'value': row['text'], 'type': 'markdown'} if row.get('text') else {'value': '', 'type': 'markdown'}
        } for row in table_data
    ]


    # --- Prepare Topic Details Table ---
    if not topics_df.empty:
         # Select and potentially rename columns for display
         display_cols = ['Topic', 'Count', 'Name'] # Add 'Representation' if needed
         cols_to_display_in_table = [col for col in display_cols if col in topics_df.columns]
         topic_details_table_component = dbc.Table.from_dataframe(
             topics_df[cols_to_display_in_table],
             striped=True,
             bordered=True,
             hover=True,
             responsive=True, # Make table scroll horizontally on small screens
             className="mt-3",
             color="dark",  # Match dark theme
         )
    else:
         topic_details_table_component = dbc.Alert("No topic information available. Train or retrain the topic model.", color="info")

    
    # --- Generate Word Cloud ---
    wordcloud_img = None
    if WORDCLOUD_AVAILABLE and not df.empty:
        try:
            # Combine all text data
            all_text = ' '.join(df['text'].fillna('').astype(str).tolist())
            
            # Clean and prepare text
            # Remove URLs, mentions, special chars, and common words
            all_text = re.sub(r'https?://\S+|www\.\S+', '', all_text)  # Remove URLs
            all_text = re.sub(r'@\w+', '', all_text)  # Remove mentions
            all_text = re.sub(r'[^\w\s]', '', all_text)  # Remove special chars
            
            # Basic stopwords list - should be expanded for better results
            stopwords = set(['the', 'and', 'to', 'of', 'a', 'in', 'is', 'it', 'that', 'for', 'was', 'on',
                             'with', 'as', 'are', 'at', 'be', 'this', 'by', 'from', 'an', 'but', 'not',
                             'what', 'all', 'were', 'when', 'we', 'you', 'they', 'there', 'has', 'have',
                             'had', 'one', 'so', 'or', 'if', 'any', 'would', 'no', 'which', 'their', 'them'])
            
            # Count frequencies
            words = re.findall(r'\b\w+\b', all_text.lower())
            word_freq = Counter(word for word in words if word not in stopwords and len(word) > 2)
            
            # Generate word cloud
            wordcloud = WordCloud(
                width=1000, 
                height=500, 
                background_color='#1e1e1e',  # Dark background
                colormap='viridis',
                max_words=100,
                min_font_size=10,
                max_font_size=80,
                prefer_horizontal=0.9
            ).generate_from_frequencies(word_freq)
            
            # Convert to image
            img = wordcloud.to_image()
            buf = BytesIO()
            img.save(buf, format='PNG')
            wordcloud_img = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
            
        except Exception as e:
            logger.error(f"Error generating word cloud: {e}", exc_info=True)
            wordcloud_img = None
    else:
        # Create a placeholder image for when wordcloud isn't available
        wordcloud_img = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="

    # --- Return all updated components ---
    return pie_fig, time_fig, topic_fig, table_data, tooltip_data, topic_details_table_component, wordcloud_img

# Auto-load data on page refresh/load
@callback(
    Output('refresh-view-button', 'n_clicks'),
    Input('auto-load-trigger', 'children'),
    prevent_initial_call=False  # Run on page load
)
def auto_load_data(_):
    """Automatically trigger the refresh-view-button on page load to get data"""
    logger.info("Auto-loading data on page load")
    # Return n_clicks=1 to simulate a click on the refresh-view button
    return 1

# --- Run the App ---
if __name__ == '__main__':
    # Make sure backend is running first! Check environment variable or default.
    logger.info(f"Attempting to connect to backend API at: {BACKEND_URL}")
    
    # Test API connection before starting the dashboard
    try:
        test_response = requests.get(f"{BACKEND_URL}/", timeout=5)
        if test_response.status_code == 200:
            logger.info(f"Successfully connected to backend API. Status: {test_response.status_code}")
        else:
            logger.warning(f"Backend API responded with status code: {test_response.status_code}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to connect to backend API: {e}")
        logger.warning("Dashboard will start but may not function correctly without backend API")
     
    logger.info("Ensure the backend FastAPI server is running before starting the dashboard.")
    logger.info("Dashboard starting...")
    # Set debug=False for production deployment
    # host='0.0.0.0' makes it accessible on your network
    app.run(debug=True, host='0.0.0.0', port=8050) # Changed from run_server to run