import streamlit as st
import pandas as pd
from modules.stats_analysis import StatsAgent
from visualization_helpers import render_custom_insight_box

def render_ai_summary_box(title, summary):
    st.markdown(f"**🤖 {title}**")
    st.markdown(summary)

def render(df, stats_agent, advanced_agent, gemini_agent, summary_agent):
    st.header("AI-Generated Insights")
    
    if gemini_agent.has_valid_key:
        st.success("✅ Connected to Google Gemini API")
    else:
        st.warning("""
        ⚠️ No valid Gemini API key found. Add your API key to the .env file:
        ```
        """)
        
        # Add diagnostics option when API connection fails
        if st.button("Run Gemini API Diagnostics"):
            with st.spinner("Running diagnostics..."):
                diagnostic_results = gemini_agent.run_model_diagnostics()
                st.code(diagnostic_results)
                
            st.info("""
            **Common fixes:**
            1. Make sure your API key is correctly set in the .env file
            2. Verify your API key is valid and has access to Gemini models
            3. Check if you're using the correct model name (e.g., 'gemini-pro' not 'gemini-1.0-pro')
            """)
    
    # Display the original summary
    st.subheader("Basic Summary")
    basic_summary = summary_agent.generate_summary()
    st.markdown(f"### Key Findings\n{basic_summary}")
    
    # Create tabs for AI insights
    ai_tab1, ai_tab2, ai_tab3 = st.tabs([
        "Time Series Insights", 
        "Topic Insights", 
        "Credibility Insights"
    ])
    
    with ai_tab1:
        time_data = stats_agent.get_posts_over_time()['day']
        
        subreddits = ["All Subreddits"] + list(df['subreddit'].value_counts().head(10).index)
        selected_subreddit = st.selectbox(
            "Select a subreddit for focused analysis:",
            subreddits,
            index=0,
            key="time_series_subreddit_selector"
        )
        
        subreddit = None if selected_subreddit == "All Subreddits" else selected_subreddit
        
        if subreddit:
            subreddit_df = df[df['subreddit'] == subreddit]
            if len(subreddit_df) > 0:
                subreddit_stats = StatsAgent(subreddit_df)
                time_data = subreddit_stats.get_posts_over_time()['day']
        
        with st.spinner("Generating AI summary..."):
            summary = gemini_agent.generate_time_series_summary(time_data, subreddit)
            render_custom_insight_box(summary, title="AI Time Series Analysis", icon="🤖")
    
    with ai_tab2:
        topic_data = advanced_agent.generate_topics(n_topics=5)
        
        subreddits = ["All Subreddits"] + list(df['subreddit'].value_counts().head(10).index)
        selected_subreddit = st.selectbox(
            "Select a subreddit for focused analysis:",
            subreddits,
            index=0,
            key="ai_topic_insights_subreddit_selector"
        )
        
        subreddit = None if selected_subreddit == "All Subreddits" else selected_subreddit
        
        if subreddit:
            subreddit_df = df[df['subreddit'] == subreddit]
            if len(subreddit_df) >= 20:  # Ensure we have enough data
                subreddit_agent = advanced_agent.__class__(subreddit_df)
                topic_data = subreddit_agent.generate_topics(n_topics=3)
        
        with st.spinner("Generating topic insights..."):
            summary = gemini_agent.generate_topic_summary(topic_data, subreddit)
            render_custom_insight_box(summary, title="AI Topic Analysis", icon="🤖")
    
    with ai_tab3:
        credibility_df = advanced_agent.score_credibility()
        
        subreddits = ["All Subreddits"] + list(df['subreddit'].value_counts().head(10).index)
        selected_subreddit = st.selectbox(
            "Select a subreddit for focused analysis:",
            subreddits,
            index=0,
            key="credibility_subreddit_selector"
        )
        
        subreddit = None if selected_subreddit == "All Subreddits" else selected_subreddit
        
        if subreddit:
            filtered_cred_df = credibility_df[credibility_df['subreddit'] == subreddit]
        else:
            filtered_cred_df = credibility_df
        
        with st.spinner("Generating credibility insights..."):
            summary = gemini_agent.generate_misinformation_summary(filtered_cred_df)
            render_custom_insight_box(summary, title="AI Credibility Analysis", icon="🤖")