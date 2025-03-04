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

# Application Configuration

# Application details
APP_TITLE = "Reddit Data Analyzer"
APP_ICON = "📊"
APP_DESCRIPTION = """
Analyze Reddit communities through post data visualization and deep content analysis. 
Upload a JSONL file with Reddit post data to begin.
"""

# Demo data path (for when no file is uploaded but demo mode is selected)
DEMO_DATA_PATH = "data/demo_reddit_data.jsonl"

# Topic modeling configurations
DEFAULT_TOPICS = 5
MAX_TOPICS = 10

# Trusted and untrusted domains for credibility analysis
TRUSTED_DOMAINS = [
    "nature.com", "science.org", "nih.gov", "nasa.gov", "edu", 
    "bbc.com", "reuters.com", "apnews.com", "who.int", "cdc.gov",
    "nytimes.com", "washingtonpost.com", "theguardian.com", 
    "scientificamerican.com", "smithsonianmag.com"
]

UNTRUSTED_DOMAINS = [
    "infowars.com", "naturalnews.com", "breitbart.com", 
    "dailywire.com", "beforeitsnews.com", "bitchute.com",
    "rumble.com", "parler.com", "gab.com", "gettr.com",
    "4chan.org", "thedcpatriot.com", "thegatewaypundit.com"
]

# Create data directory if it doesn't exist
os.makedirs(DATA_DIR, exist_ok=True)
