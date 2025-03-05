import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DEMO_DATA_PATH = os.path.join(DATA_DIR, "demo_reddit_data.jsonl")
APP_TITLE = "Reddit Data Analyzer"
APP_ICON = "📊"
APP_DESCRIPTION = """
Analyze Reddit communities through post data visualization and deep content analysis. 
Upload a JSONL file with Reddit post data to begin.
"""

DEMO_DATA_PATH = "data/data.jsonl"

DEFAULT_TOPICS = 5
MAX_TOPICS = 10

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

os.makedirs(DATA_DIR, exist_ok=True)
