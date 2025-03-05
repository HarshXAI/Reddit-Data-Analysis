import streamlit as st
import os
import config
import data_processing
from modules.stats_analysis import StatsAgent
from modules.summary_agent import SummaryAgent
from modules.advanced_analysis import AdvancedAnalysisAgent
from modules.topic_modeling import TopicModelAgent
from modules.ai_summary import GeminiSummaryAgent
from visualization_helpers import render_metric_card, render_insight_box

from pages import overview, time_series, text_analysis, advanced_topics, credibility, ai_insights

st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon=config.APP_ICON,
    layout="wide",
    initial_sidebar_state="collapsed"
)

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
            
            with overview_tab:
                overview.render(df, stats_agent, advanced_agent)
                
            with time_series_tab:
                time_series.render(df, stats_agent)
                
            with text_analysis_tab:
                text_analysis.render(df, stats_agent)
                
            with advanced_topics_tab:
                advanced_topics.render(df, advanced_agent, gemini_agent)
                
            with credibility_tab:
                credibility.render(df, advanced_agent)
                
            with ai_insights_tab:
                ai_insights.render(df, stats_agent, advanced_agent, gemini_agent, summary_agent)
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