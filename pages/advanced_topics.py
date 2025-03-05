import streamlit as st
import plotly.express as px
import pandas as pd
from visualization_helpers import render_custom_insight_box

def render(df, advanced_agent, gemini_agent):
    st.header("Advanced Topic Analysis")
    
    topic_data = advanced_agent._generate_simple_topics(n_topics=5)
    
    subreddits = ["All Subreddits"] + list(df['subreddit'].value_counts().head(10).index)
    selected_subreddit = st.selectbox(
        "Select a subreddit for focused analysis:",
        subreddits,
        index=0,
        key="advanced_topic_subreddit_selector"
    )
    
    subreddit = None if selected_subreddit == "All Subreddits" else selected_subreddit
    
    if subreddit:
        subreddit_df = df[df['subreddit'] == subreddit]
        if len(subreddit_df) >= 20:  # Ensure we have enough data
            subreddit_agent = advanced_agent.__class__(subreddit_df)
            topic_data = subreddit_agent._generate_simple_topics(n_topics=3)
    
    with st.spinner("Generating topic insights..."):
        summary = gemini_agent.generate_topic_summary(topic_data, subreddit)
        render_custom_insight_box(summary, title="AI Topic Analysis", icon="🤖")
    
    st.subheader("Keyword Trend Detection")
    
    time_window = st.selectbox(
        "Time Window",
        ["Day (D)", "Week (W)", "Month (M)"],
        index=1
    )
    
    window_code = time_window[time_window.find("(")+1:time_window.find(")")]
    
    min_count = st.slider("Minimum Keyword Count", 3, 20, 5)
    
    with st.spinner("Detecting trends..."):
        trend_data = advanced_agent.detect_trends(time_window=window_code, min_count=min_count)
    
    if "error" in trend_data:
        st.error(trend_data["error"])
    else:
        trending_df = trend_data.get("trending_keywords", pd.DataFrame())
        trend_timeseries = trend_data.get("trend_timeseries", pd.DataFrame())
        top_keywords = trend_data.get("top_keywords", [])
        
        if trending_df.empty:
            st.info("No significant trends detected in this time period. Try adjusting the time window or minimum count.")
        else:
            st.subheader("Top Trending Keywords")
            
            trending_display = trending_df.copy()
            if 'increase_ratio' in trending_display.columns:
                trending_display['increase'] = trending_display['increase_ratio'].apply(
                    lambda x: f"{x:.1f}x" if x != float('inf') else "New"
                )
            
            st.dataframe(
                trending_display[['word', 'period', 'count', 'increase']].head(10),
                use_container_width=True,
                hide_index=True
            )
        
        if not trend_timeseries.empty and top_keywords:
            st.subheader("Keyword Trends Over Time")
            
            show_keywords = st.multiselect(
                "Select keywords to visualize:",
                options=top_keywords,
                default=top_keywords[:min(5, len(top_keywords))]
            )
            
            if show_keywords:
                filtered_trends = trend_timeseries[trend_timeseries['keyword'].isin(show_keywords)]
                
                fig = px.line(
                    filtered_trends,
                    x="period",
                    y="count",
                    color="keyword",
                    markers=True,
                    title="Keyword Frequency Over Time"
                )
                fig.update_layout(
                    hovermode="x unified"
                )
                st.plotly_chart(fig, use_container_width=True)
