import streamlit as st
import os
import sys
import config
import data_processing
from modules.stats_analysis import StatsAgent
from modules.summary_agent import SummaryAgent
from modules.advanced_analysis import AdvancedAnalysisAgent
from modules.topic_modeling import TopicModelAgent
from modules.ai_summary import GeminiSummaryAgent
import logging
import time
from typing import Tuple, Optional
import pandas as pd
import nltk
from nltk_setup import setup_nltk
import importlib.util

# Setup NLTK resources early
setup_nltk()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure current directory is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
    logger.info(f"Added current directory to Python path: {current_dir}")

# Import page modules with robust error handling
def import_page_module(module_name):
    """Import a page module with detailed error handling"""
    try:
        # First try the standard import
        return importlib.import_module(f"pages.{module_name}")
    except ImportError as e:
        logger.error(f"Failed to import {module_name}: {str(e)}")
        
        # Try direct path import as fallback
        try:
            module_path = os.path.join(current_dir, "pages", f"{module_name}.py")
            if not os.path.exists(module_path):
                logger.error(f"Module file not found: {module_path}")
                raise ImportError(f"Module file not found: {module_path}")
                
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            if spec is None:
                logger.error(f"Failed to create spec for {module_path}")
                raise ImportError(f"Failed to create spec for {module_path}")
                
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        except Exception as e2:
            logger.error(f"Failed to import {module_name} from path: {str(e2)}")
            
            # Create a stub module as last resort
            class StubModule:
                @staticmethod
                def render(*args, **kwargs):
                    st.error(f"Failed to load {module_name} module")
                    st.warning(f"Error details: {str(e)}")
            
            logger.warning(f"Created stub module for {module_name}")
            return StubModule()

# Import all page modules
page_modules = {}
for module_name in ["overview", "time_series", "text_analysis", "advanced_topics", "credibility", "ai_insights"]:
    page_modules[module_name] = import_page_module(module_name)
    logger.info(f"Loaded module: {module_name}")

# Page config
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon=config.APP_ICON,
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Hide the sidebar
st.markdown("""
<style>
    [data-testid="collapsedControl"] {display: none;}
    section[data-testid="stSidebar"] {display: none !important;}
</style>
""", unsafe_allow_html=True)

def main():
    st.title(config.APP_TITLE)
    st.markdown(config.APP_DESCRIPTION)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader("Upload JSONL Data", type=["jsonl", "json"])
    
    with col2:
        st.write("")
        st.write("")
        use_demo_data = st.checkbox("Use Demo Data", value=uploaded_file is None)
    
    if uploaded_file is not None or use_demo_data:
        with st.spinner("Processing data..."):
            df, stats_agent = data_processing.load_data(uploaded_file, use_demo_data)
            if df is None:
                st.error("Error loading data. Please check your file format.")
                return
                
            advanced_agent = AdvancedAnalysisAgent(df)
            topic_agent = TopicModelAgent(df)
            gemini_agent = GeminiSummaryAgent()
            summary_agent = SummaryAgent(df, stats_agent, topic_agent)
            
            st.success(f"Loaded {len(df):,} Reddit posts from {'demo data' if use_demo_data else 'uploaded file'}")
            
            overview_tab, time_series_tab, text_analysis_tab, advanced_topics_tab, credibility_tab, ai_insights_tab = st.tabs([
                "📈 Overview & Stats", 
                "⏱️ Time Series Analysis", 
                "🔤 Text Analysis",
                "🔍 Advanced Topics",
                "🛡️ Credibility Analysis",
                "🤖 AI Insights"
            ])
            
            # Use the modules from our page_modules dictionary
            with overview_tab:
                page_modules["overview"].render(df, stats_agent, advanced_agent)
                
            with time_series_tab:
                page_modules["time_series"].render(df, stats_agent)
                
            with text_analysis_tab:
                page_modules["text_analysis"].render(df, stats_agent)
                
            with advanced_topics_tab:
                page_modules["advanced_topics"].render(df, advanced_agent, gemini_agent)
                
            with credibility_tab:
                page_modules["credibility"].render(df, advanced_agent)
                
            with ai_insights_tab:
                page_modules["ai_insights"].render(df, stats_agent, advanced_agent, gemini_agent, summary_agent)
    else:
        st.info("Please upload a JSONL file containing Reddit data to begin analysis.")
        st.markdown("""
        ## Expected Data Format
        
        The JSONL file should contain Reddit posts with one JSON object per line:
        ```
        {"kind": "t3", "data": {"subreddit": "...", "title": "...", "selftext": "...", "score": 123, ...}}
        {"kind": "t3", "data": {"subreddit": "...", "title": "...", "selftext": "...", "score": 123, ...}}
        ```
        """)

if __name__ == "__main__":
    main()