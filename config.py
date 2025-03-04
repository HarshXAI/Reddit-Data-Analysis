import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# File paths and directories
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DEMO_DATA_PATH = os.path.join(DATA_DIR, "demo_reddit_data.jsonl")

# App styling configs
APP_TITLE = "Reddit Data Analysis Dashboard"
APP_ICON = "📊"
APP_DESCRIPTION = """
This dashboard analyzes Reddit posts to extract insights, trends, and topics.
Upload your Reddit JSONL data to begin analysis.
"""

# Analysis settings
DEFAULT_TOPICS = 5
MAX_TOPICS = 10
TRUSTED_DOMAINS = {
    'wikipedia.org', 'nytimes.com', 'reuters.com', 'bbc.com', 
    'theguardian.com', 'washingtonpost.com', 'nih.gov',
    'edu', 'gov', 'nature.com', 'science.org', 'who.int'
}
UNTRUSTED_DOMAINS = {
    'example-fake-news.com', 'totallyrealnews.com'
}

# Create data directory if it doesn't exist
os.makedirs(DATA_DIR, exist_ok=True)
