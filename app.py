import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import nltk
import os
import sys
from datetime import datetime, timedelta
import numpy as np
from wordcloud import WordCloud
import config


# Import custom modules
from modules.data_ingestion import DataIngestionAgent
from modules.stats_analysis import StatsAgent
from modules.topic_modeling import TopicModelAgent
from modules.summary_agent import SummaryAgent
from modules.advanced_analysis import AdvancedAnalysisAgent
from modules.ai_summary import GeminiSummaryAgent

# Page config
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon=config.APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

def render_metric_card(label, value, delta=None, unit=""):
    """Render a styled metric card."""
    delta_html = f"<span style='font-size: 0.8rem; color: {'#2e7d32' if delta and delta > 0 else '#c62828' if delta and delta < 0 else '#6c757d'};'>{delta:+.1f}% {unit}</span>" if delta is not None else ""
    
    st.markdown(f"""
    <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 8px; text-align: center;">
        <div style="font-size: 0.9rem; color: #6c757d; margin-bottom: 0.2rem;">{label}</div>
        <div style="font-size: 1.8rem; font-weight: 700; color: #1e88e5;">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

def render_ai_summary(summary):
    """Render a styled AI summary box."""
    st.markdown(f"""
    <div style="background-color: #f1f8ff; border-left: 4px solid #1e88e5; padding: 1rem; margin: 1rem 0; border-radius: 4px;">
        <h3>🤖 AI Summary</h3>
        <p>{summary}</p>
    </div>
    """, unsafe_allow_html=True)

def render_credibility_meter(score, factors):
    """Render a credibility score meter."""
    # Determine color based on score
    if score >= 70:
        color = "#4caf50"  # Green for high credibility
    elif score >= 40:
        color = "#ff9800"  # Orange/Amber for medium credibility
    else:
        color = "#f44336"  # Red for low credibility
        
    st.markdown(f"""
    <div style="margin-bottom: 1rem;">
        <div style="display: flex; justify-content: space-between;">
            <span>Low Credibility</span>
            <span>High Credibility</span>
        </div>
        <div style="height: 8px; width: 100%; background-color: #e9ecef; border-radius: 4px; overflow: hidden; margin: 4px 0;">
            <div style="height: 100%; width: {score}%; background-color: {color}; border-radius: 4px;"></div>
        </div>
        <div style="text-align: center; margin-top: 0.5rem;">
            <span style="font-weight: 600; color: {color};">Score: {score}/100</span>
        </div>
        <div style="font-size: 0.85rem; margin-top: 0.5rem;">
            <b>Factors:</b> {factors if factors else "No specific factors detected"}
        </div>
    </div>
    """, unsafe_allow_html=True)

def load_data(uploaded_file=None, use_demo_data=False):
    """
    Load data from either an uploaded file or generate demo data.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        use_demo_data: Flag to use demo data if no file is uploaded
        
    Returns:
        df: Processed DataFrame with Reddit data
        ingestion_agent: DataIngestionAgent instance
    """
    if uploaded_file is not None:
        # Process uploaded file
        ingestion_agent = DataIngestionAgent(uploaded_file)
        return ingestion_agent.get_dataframe(), ingestion_agent
    
    elif use_demo_data:
        # Try to load demo data or create synthetic data
        if os.path.exists(config.DEMO_DATA_PATH):
            ingestion_agent = DataIngestionAgent(config.DEMO_DATA_PATH)
            return ingestion_agent.get_dataframe(), ingestion_agent
        else:
            # Generate synthetic data
            synthetic_data = []
            base_date = datetime.now() - timedelta(days=30)
            
            subreddits = ["WorldNews", "Technology", "Science", "Gaming", "Politics"]
            topics = ["AI", "Climate", "Elections", "Space", "Economy"]
            
            for i in range(100):  # Generate 100 synthetic posts
                created_date = base_date + timedelta(
                    days=np.random.randint(0, 30),
                    hours=np.random.randint(0, 24)
                )
                
                subreddit = np.random.choice(subreddits)
                topic = np.random.choice(topics)
                score = int(np.random.normal(50, 30))
                
                synthetic_data.append({
                    "subreddit": subreddit,
                    "title": f"Discussion about {topic} impact on society",
                    "selftext": f"This is a synthetic post about {topic} for demo purposes.",
                    "author": f"demo_user_{np.random.randint(1, 11)}",
                    "score": score,
                    "created_utc": created_date.timestamp()
                })
            
            # Create a DataFrame with synthetic data
            df = pd.DataFrame(synthetic_data)
            
            # Add date columns
            df['created_date'] = pd.to_datetime(df['created_utc'], unit='s')
            df['date'] = df['created_date'].dt.date
            df['year'] = df['created_date'].dt.year
            df['month'] = df['created_date'].dt.month
            df['day'] = df['created_date'].dt.day
            df['day_of_week'] = df['created_date'].dt.dayofweek
            df['hour'] = df['created_date'].dt.hour
            
            # Create a DataIngestionAgent but skip loading
            ingestion_agent = DataIngestionAgent.__new__(DataIngestionAgent)
            ingestion_agent.df = df
            return df, ingestion_agent
    
    return None, None

def main():
    # App title and description
    st.title(config.APP_TITLE)
    st.markdown(config.APP_DESCRIPTION)
    
    # File uploader for JSONL data
    uploaded_file = st.sidebar.file_uploader("Upload Reddit JSONL Data", type=["jsonl", "json"])
    
    # Add demo data option
    use_demo_data = st.sidebar.checkbox("Use Demo Data", value=uploaded_file is None)
    
    if uploaded_file is not None or use_demo_data:
        # Process data
        with st.spinner("Processing data..."):
            df, ingestion_agent = load_data(uploaded_file, use_demo_data)
            if df is None:
                st.error("Error loading data. Please check your file format.")
                return
                
            # Initialize agents
            stats_agent = StatsAgent(df)
            advanced_agent = AdvancedAnalysisAgent(df)
            
            # Show basic info about the dataset
            st.sidebar.success(f"Loaded {len(df):,} Reddit posts")
            
            # Create tabs for different analyses
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "📈 Overview & Stats", 
                "⏱️ Time Series Analysis", 
                "🔤 Text Analysis",
                "🔍 Advanced Topics",
                "🛡️ Credibility Analysis",
                "🤖 AI Insights"
            ])
            
            with tab1:
                st.header("Dataset Overview")
                
                # Display general stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    render_metric_card("Total Posts", f"{len(df):,}")
                with col2:
                    render_metric_card("Total Subreddits", f"{stats_agent.get_unique_subreddit_count():,}")
                with col3:
                    render_metric_card("Avg. Score", f"{stats_agent.get_average_score():.1f}")
                
                # Top subreddits
                st.subheader("Top Subreddits")
                subreddit_counts = stats_agent.get_subreddit_distribution()
                fig = px.bar(
                    subreddit_counts.head(10),
                    x="count",
                    y="subreddit",
                    orientation="h",
                    title="Top 10 Subreddits by Post Count",
                    color="count",
                    color_continuous_scale="Blues"
                )
                fig.update_layout(
                    yaxis=dict(autorange="reversed"),
                    coloraxis_showscale=False
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Score distribution
                st.subheader("Score Distribution")
                fig = px.histogram(
                    df,
                    x="score",
                    nbins=50,
                    title="Distribution of Post Scores",
                    color_discrete_sequence=["#1e88e5"]
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Network graph section (in overview tab)
                st.subheader("Community Network Analysis")
                
                # Check if we should show the network graph
                show_network = st.checkbox("Show Author-Subreddit Network Graph", value=False)
                
                if show_network:
                    with st.spinner("Generating network graph..."):
                        network_results = advanced_agent.generate_network_graph()
                        
                        if "error" in network_results:
                            st.warning(network_results["error"])
                            st.info("Network analysis requires data with author and subreddit information, with multiple posts by the same authors.")
                        else:
                            # Get the network data
                            network_data = network_results["network_data"]
                            graph_stats = network_results["graph_stats"]
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Subreddits", f"{graph_stats['num_subreddits']}")
                            with col2:
                                st.metric("Authors", f"{graph_stats['num_authors']}")
                            with col3:
                                st.metric("Connections", f"{graph_stats['num_edges']}")
                            
                            # Create network visualization
                            fig = go.Figure()
                            
                            # Add edges
                            fig.add_trace(go.Scatter(
                                x=network_data["edge_x"],
                                y=network_data["edge_y"],
                                line=dict(width=0.5, color="#cccccc"),
                                hoverinfo="none",
                                mode="lines"
                            ))
                            
                            # Add nodes
                            fig.add_trace(go.Scatter(
                                x=network_data["node_x"],
                                y=network_data["node_y"],
                                mode="markers",
                                marker=dict(
                                    size=network_data["node_size"],
                                    color=network_data["node_color"],
                                    line=dict(width=1, color="#888888")
                                ),
                                text=network_data["node_text"],
                                hovertemplate="%{text}<extra></extra>"
                            ))
                            
                            fig.update_layout(
                                showlegend=False,
                                title="Author-Subreddit Network",
                                hovermode="closest",
                                margin=dict(b=0, l=0, r=0, t=50),
                                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                height=600
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            st.markdown("""
                            **Network Legend**: 
                            - **Red nodes**: Subreddits (size shows post count)
                            - **Blue nodes**: Authors
                            - **Lines**: Author posted in subreddit
                            """)
            
            with tab2:
                st.header("Time Series Analysis")
                
                # Posts over time
                posts_over_time = stats_agent.get_posts_over_time()
                
                # Time aggregation selector
                time_agg = st.selectbox(
                    "Time Aggregation",
                    ["Day", "Week", "Month"],
                    index=0
                )
                
                if time_agg == "Day":
                    time_data = posts_over_time['day']
                elif time_agg == "Week":
                    time_data = posts_over_time['week']
                else:
                    time_data = posts_over_time['month']
                
                # Show peak statistics if we have data
                if not time_data.empty:
                    max_date = time_data.loc[time_data['count'].idxmax(), 'date']
                    max_count = time_data['count'].max()
                    
                    st.metric(
                        "Peak Activity", 
                        f"{max_count} posts", 
                        f"on {max_date.strftime('%Y-%m-%d') if isinstance(max_date, pd.Timestamp) else max_date}"
                    )
                
                # Line chart with improved styling
                fig = px.line(
                    time_data,
                    x="date",
                    y="count",
                    title=f"Posts by {time_agg}",
                    markers=True
                )
                fig.update_traces(
                    line=dict(color="#1e88e5", width=2),
                    marker=dict(size=6, color="#1e88e5")
                )
                fig.update_layout(
                    hovermode="x unified"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Posts by day of week and hour
                col1, col2 = st.columns(2)
                
                with col1:
                    dow_data = stats_agent.get_posts_by_day_of_week()
                    fig = px.bar(
                        dow_data,
                        x="day_name",
                        y="count",
                        title="Posts by Day of Week",
                        color="count",
                        color_continuous_scale="Blues"
                    )
                    fig.update_layout(coloraxis_showscale=False)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    hour_data = stats_agent.get_posts_by_hour()
                    fig = px.line(
                        hour_data,
                        x="hour",
                        y="count",
                        title="Posts by Hour of Day",
                        markers=True
                    )
                    fig.update_traces(
                        line=dict(color="#1e88e5", width=2),
                        marker=dict(size=6)
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                st.header("Text Analysis")
                
                # Word cloud
                st.subheader("Word Cloud of Post Titles")
                wordcloud = stats_agent.generate_title_wordcloud()
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis("off")
                st.pyplot(fig)
                
                # Top keywords
                st.subheader("Top Keywords in Titles")
                keywords = stats_agent.get_top_keywords_in_titles(n=20)
                fig = px.bar(
                    keywords,
                    x="count",
                    y="word",
                    orientation="h",
                    title="Top 20 Keywords in Post Titles",
                    color="count",
                    color_continuous_scale="Blues"
                )
                fig.update_layout(coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)
                
                # Keyword search
                st.subheader("Keyword Search")
                search_term = st.text_input("Enter keyword to search in posts:")
                if search_term:
                    filtered_posts = stats_agent.search_posts(search_term)
                    st.write(f"Found {len(filtered_posts)} posts containing '{search_term}'")
                    st.dataframe(filtered_posts[['title', 'subreddit', 'score', 'created_date']])
                
                # Original topic modeling
                st.subheader("Basic Topic Modeling")
                
                n_topics = st.slider("Number of Topics", min_value=2, max_value=config.MAX_TOPICS, value=config.DEFAULT_TOPICS)
                
                with st.spinner("Generating topic model..."):
                    topic_agent = TopicModelAgent(df)
                    topics = topic_agent.generate_topics(n_topics=n_topics)
                
                for i, (topic_id, words, docs) in enumerate(topics):
                    with st.expander(f"Topic {i+1}: {', '.join(words[:3])}"):
                        st.write(f"**Keywords**: {', '.join(words)}")
                        st.write("**Example posts:**")
                        for j, doc in enumerate(docs[:3]):
                            st.markdown(f"- {doc}")
                            if j >= 2:  # Show only top 3 examples
                                break
            
            with tab4:
                st.header("Advanced Topic Analysis")
                
                # BERTopic option
                st.subheader("BERTopic Model")
                use_bert = st.checkbox("Use BERTopic (advanced but slower)", value=False)
                
                if use_bert:
                    st.info("BERTopic uses advanced NLP for more accurate topics but is slower to process.")
                    n_bert_topics = st.slider("Number of BERTopic Topics", min_value=2, max_value=config.MAX_TOPICS, value=config.DEFAULT_TOPICS, key="bert_slider")
                    
                    with st.spinner("Generating BERTopic model..."):
                        topic_data = advanced_agent.generate_bert_topics(n_topics=n_bert_topics)
                        
                        # Check if we have an error and need to use fallback
                        if "error" in topic_data and "fallback" in topic_data:
                            st.warning(f"BERTopic error: {topic_data['error']}. Using fallback method.")
                            topic_data = topic_data["fallback"]
                        elif "error" in topic_data:
                            st.error(topic_data["error"])
                            topic_data = {"topic_terms": {}, "topic_docs": {}}
                    
                    # Display topics
                    topic_terms = topic_data.get("topic_terms", {})
                    topic_docs = topic_data.get("topic_docs", {})
                    
                    if not topic_terms:
                        st.warning("No topics were found in the data.")
                    else:
                        # Create a word cloud of the top topics combined
                        st.subheader("Topic Word Cloud")
                        all_topic_words = []
                        for words in topic_terms.values():
                            all_topic_words.extend(words[:5])  # Top 5 words from each topic
                        
                        if all_topic_words:
                            # Count word frequencies
                            word_freqs = {}
                            for word in all_topic_words:
                                if word in word_freqs:
                                    word_freqs[word] += 1
                                else:
                                    word_freqs[word] = 1
                            
                            wordcloud = WordCloud(
                                width=800, height=400,
                                background_color='white',
                                colormap='viridis',
                                max_words=100
                            ).generate_from_frequencies(word_freqs)
                            
                            fig, ax = plt.subplots(figsize=(10, 6))
                            ax.imshow(wordcloud, interpolation='bilinear')
                            ax.axis("off")
                            st.pyplot(fig)
                        
                        # Topic listing
                        st.subheader("Discovered Topics")
                        
                        for topic_id, words in topic_terms.items():
                            docs = topic_docs.get(topic_id, [])
                            
                            with st.expander(f"Topic {topic_id+1}: {', '.join(words[:3])}"):
                                st.markdown(f"**Keywords**: {', '.join(words)}")
                                st.markdown("**Example posts:**")
                                for doc in docs[:3]:
                                    st.markdown(f"- {doc}")
                else:
                    st.info("Enable BERTopic above for advanced topic modeling, or use the basic topic modeling in the Text Analysis tab.")
                
                # Trend detection section
                st.subheader("Keyword Trend Detection")
                
                # Time window selection
                time_window = st.selectbox(
                    "Time Window",
                    ["Day (D)", "Week (W)", "Month (M)"],
                    index=1
                )
                
                # Map selection to time window code
                window_code = time_window[time_window.find("(")+1:time_window.find(")")]
                
                # Minimum count selection
                min_count = st.slider("Minimum Keyword Count", 3, 20, 5)
                
                # Generate trends
                with st.spinner("Detecting trends..."):
                    trend_data = advanced_agent.detect_trends(time_window=window_code, min_count=min_count)
                
                if "error" in trend_data:
                    st.error(trend_data["error"])
                else:
                    # Get trending keywords and timeseries
                    trending_df = trend_data.get("trending_keywords", pd.DataFrame())
                    trend_timeseries = trend_data.get("trend_timeseries", pd.DataFrame())
                    top_keywords = trend_data.get("top_keywords", [])
                    
                    if trending_df.empty:
                        st.info("No significant trends detected in this time period. Try adjusting the time window or minimum count.")
                    else:
                        # Display top trending keywords
                        st.subheader("Top Trending Keywords")
                        
                        # Format the trending keywords table
                        trending_display = trending_df.copy()
                        if 'increase_ratio' in trending_display.columns:
                            trending_display['increase'] = trending_display['increase_ratio'].apply(
                                lambda x: f"{x:.1f}x" if x != float('inf') else "New"
                            )
                        
                        # Show the trending keywords
                        st.dataframe(
                            trending_display[['word', 'period', 'count', 'increase']].head(10),
                            use_container_width=True,
                            hide_index=True
                        )
                    
                    # Display trend visualization
                    if not trend_timeseries.empty and top_keywords:
                        st.subheader("Keyword Trends Over Time")
                        
                        # Filter to top keywords for cleaner visualization
                        show_keywords = st.multiselect(
                            "Select keywords to visualize:",
                            options=top_keywords,
                            default=top_keywords[:min(5, len(top_keywords))]
                        )
                        
                        if show_keywords:
                            # Filter the data
                            filtered_trends = trend_timeseries[trend_timeseries['keyword'].isin(show_keywords)]
                            
                            # Create a trend visualization
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
            
            with tab5:
                st.header("Content Credibility Analysis")
                
                with st.spinner("Analyzing content credibility..."):
                    # Generate credibility scores
                    credibility_df = advanced_agent.score_credibility()
                    
                    if 'error' in credibility_df.columns:
                        st.error(f"Error in credibility analysis: {credibility_df['error'].iloc[0]}")
                    else:
                        # Overall stats
                        col1, col2, col3 = st.columns(3)
                        
                        avg_score = credibility_df['credibility_score'].mean()
                        with col1:
                            render_metric_card("Average Credibility Score", f"{avg_score:.1f}/100")
                        
                        high_cred = (credibility_df['credibility_score'] >= 70).sum()
                        high_percent = (high_cred / len(credibility_df)) * 100
                        with col2:
                            render_metric_card("High Credibility Posts", f"{high_cred} ({high_percent:.1f}%)")
                        
                        low_cred = (credibility_df['credibility_score'] < 40).sum()
                        low_percent = (low_cred / len(credibility_df)) * 100
                        with col3:
                            render_metric_card("Low Credibility Posts", f"{low_cred} ({low_percent:.1f}%)")
                        
                        # Display credibility distribution
                        st.subheader("Credibility Score Distribution")
                        
                        fig = px.histogram(
                            credibility_df,
                            x="credibility_score",
                            nbins=20,
                            color_discrete_sequence=["#1e88e5"]
                        )
                        fig.update_layout(
                            xaxis_title="Credibility Score",
                            yaxis_title="Number of Posts",
                            xaxis=dict(range=[0, 100]),
                        )
                        # Add vertical lines for low/medium/high credibility boundaries
                        fig.add_shape(
                            type="line", line=dict(dash="dash", color="red"),
                            x0=40, x1=40, y0=0, y1=1, yref="paper"
                        )
                        fig.add_shape(
                            type="line", line=dict(dash="dash", color="green"),
                            x0=70, x1=70, y0=0, y1=1, yref="paper"
                        )
                        # Add annotations
                        fig.add_annotation(x=20, y=0.95, yref="paper", text="Low Credibility", showarrow=False)
                        fig.add_annotation(x=55, y=0.95, yref="paper", text="Medium Credibility", showarrow=False)
                        fig.add_annotation(x=85, y=0.95, yref="paper", text="High Credibility", showarrow=False)
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Posts with lowest credibility scores
                        st.subheader("Posts with Low Credibility Scores")
                        
                        low_cred_posts = credibility_df.sort_values('credibility_score').head(5)
                        
                        for _, post in low_cred_posts.iterrows():
                            with st.expander(f"{post['title']}"):
                                st.markdown(f"**Subreddit**: r/{post['subreddit']}")
                                st.markdown(f"**Credibility Score**: {post['credibility_score']}/100")
                                
                                # Render credibility meter
                                render_credibility_meter(
                                    int(post['credibility_score']),
                                    post['credibility_factors']
                                )
                        
                        # Explanation of the credibility scoring
                        with st.expander("How credibility scores are calculated"):
                            st.markdown("""
                            ### Credibility Scoring Methodology
                            
                            Posts are analyzed based on several factors:
                            
                            1. **Source Credibility**: Links to reputable sources increase score, while suspicious domains lower it
                            2. **Content Analysis**: Posts with misinformation patterns or conspiracy terms receive lower scores
                            3. **Presentation Style**: Excessive capitalization, punctuation, or alarmist language lowers credibility
                            4. **User History**: Author's historical credibility impacts score (when available)
                            
                            Scores range from 0-100, with higher scores indicating higher credibility.
                            """)
            
            with tab6:
                st.header("AI-Generated Insights")
                
                # Initialize the summary agent with Gemini API
                gemini_agent = GeminiSummaryAgent()
                
                if gemini_agent.has_valid_key:
                    st.success("✅ Connected to Google Gemini API")
                else:
                    st.warning("""
                    ⚠️ No valid Gemini API key found. Add your API key to the .env file:
                    ```
                    GEMINI_API_KEY=your_key_here
                    ```
                    Using mock summaries for demonstration.
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
                summary_agent = SummaryAgent(df, stats_agent, topic_agent)
                basic_summary = summary_agent.generate_summary()
                st.markdown(f"### Key Findings\n{basic_summary}")
                
                # Create tabs for AI insights
                ai_tab1, ai_tab2, ai_tab3 = st.tabs([
                    "Time Series Insights", 
                    "Topic Insights", 
                    "Credibility Insights"
                ])
                
                with ai_tab1:
                    # Get time series data for AI analysis
                    time_data = stats_agent.get_posts_over_time()['day']
                    
                    # Select a subreddit for focused analysis
                    subreddits = ["All Subreddits"] + list(df['subreddit'].value_counts().head(10).index)
                    selected_subreddit = st.selectbox(
                        "Select a subreddit for focused analysis:",
                        subreddits,
                        index=0,
                        key="time_series_subreddit_selector"  # Unique key for time series tab
                    )
                    
                    subreddit = None if selected_subreddit == "All Subreddits" else selected_subreddit
                    
                    # Get time series for the selected subreddit
                    if subreddit:
                        subreddit_df = df[df['subreddit'] == subreddit]
                        if len(subreddit_df) > 0:
                            subreddit_stats = StatsAgent(subreddit_df)
                            time_data = subreddit_stats.get_posts_over_time()['day']
                    
                    # Generate the AI summary
                    with st.spinner("Generating AI summary..."):
                        summary = gemini_agent.generate_time_series_summary(time_data, subreddit)
                        st.markdown(f"""
                        <div style="background-color: #f1f8ff; border-left: 4px solid #1e88e5; padding: 1rem; margin: 1rem 0; border-radius: 4px;">
                            <h4 style="color: #1e88e5; margin-top: 0; margin-bottom: 0.75rem; font-weight: 600;">🤖 AI Time Series Analysis</h4>
                            <p style="color: #333333; font-size: 16px; line-height: 1.6; margin-bottom: 0; font-family: 'Segoe UI', system-ui, sans-serif;">{summary}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                with ai_tab2:
                    # Generate topics for AI summary
                    advanced_agent = AdvancedAnalysisAgent(df)
                    topic_data = advanced_agent._generate_simple_topics(n_topics=5)
                    
                    # Select a subreddit for focused analysis
                    subreddits = ["All Subreddits"] + list(df['subreddit'].value_counts().head(10).index)
                    selected_subreddit = st.selectbox(
                        "Select a subreddit for focused analysis:",
                        subreddits,
                        index=0,
                        key="topic_insights_subreddit_selector"  # Updated unique key
                    )
                    
                    subreddit = None if selected_subreddit == "All Subreddits" else selected_subreddit
                    
                    # Get topics for the selected subreddit if specified
                    if subreddit:
                        subreddit_df = df[df['subreddit'] == subreddit]
                        if len(subreddit_df) >= 20:  # Ensure we have enough data
                            subreddit_agent = AdvancedAnalysisAgent(subreddit_df)
                            topic_data = subreddit_agent._generate_simple_topics(n_topics=3)
                    
                    # Generate the topic summary
                    with st.spinner("Generating topic insights..."):
                        summary = gemini_agent.generate_topic_summary(topic_data, subreddit)
                        st.markdown(f"""
                        <div style="background-color: #f1f8ff; border-left: 4px solid #1e88e5; padding: 1rem; margin: 1rem 0; border-radius: 4px;">
                            <h4 style="color: #1e88e5; margin-top: 0; margin-bottom: 0.75rem; font-weight: 600;">🤖 AI Time Series Analysis</h4>
                            <p style="color: #333333; font-size: 16px; line-height: 1.6; margin-bottom: 0; font-family: 'Segoe UI', system-ui, sans-serif;">{summary}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                with ai_tab3:
                    # Get credibility scores for AI analysis
                    credibility_df = advanced_agent.score_credibility()
                    
                    # Select a subreddit for focused analysis
                    subreddits = ["All Subreddits"] + list(df['subreddit'].value_counts().head(10).index)
                    selected_subreddit = st.selectbox(
                        "Select a subreddit for focused analysis:",
                        subreddits,
                        index=0,
                        key="credibility_subreddit_selector"  # Updated unique key
                    )
                    
                    subreddit = None if selected_subreddit == "All Subreddits" else selected_subreddit
                    
                    # Filter credibility data by subreddit if specified
                    if subreddit:
                        filtered_cred_df = credibility_df[credibility_df['subreddit'] == subreddit]
                    else:
                        filtered_cred_df = credibility_df
                    
                    # Generate the credibility summary
                    with st.spinner("Generating credibility insights..."):
                        summary = gemini_agent.generate_misinformation_summary(filtered_cred_df)
                        st.markdown(f"""
                        <div style="background-color: #f1f8ff; border-left: 4px solid #1e88e5; padding: 1rem; margin: 1rem 0; border-radius: 4px;">
                            <h4 style="color: #1e88e5; margin-top: 0; margin-bottom: 0.75rem; font-weight: 600;">🤖 AI Time Series Analysis</h4>
                            <p style="color: #333333; font-size: 16px; line-height: 1.6; margin-bottom: 0; font-family: 'Segoe UI', system-ui, sans-serif;">{summary}</p>
                        </div>
                        """, unsafe_allow_html=True)
    else:
        # Show sample data and instructions if no file uploaded
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
