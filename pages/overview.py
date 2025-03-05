import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from visualization_helpers import render_insight_box, render_metric_card

def render(df, stats_agent, advanced_agent):
    st.header("Dataset Overview")
    
    
    col1, col2, col3 = st.columns(3)
    with col1:
        render_metric_card("Total Posts", f"{len(df):,}")
    with col2:
        render_metric_card("Total Subreddits", f"{stats_agent.get_unique_subreddit_count():,}")
    with col3:
        render_metric_card("Avg. Score", f"{stats_agent.get_average_score():.1f}")
    

    st.subheader("Top Subreddits")
    subreddit_counts = stats_agent.get_subreddit_distribution()
    top_subreddit = subreddit_counts.iloc[0]['subreddit']
    top_count = subreddit_counts.iloc[0]['count']
    top_percent = (top_count / len(df)) * 100
    
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
    
    # Add dynamic explanation for top subreddits
    insight_text = f"""
    **Interpretation:** This graph shows the distribution of posts across the most active subreddits. 
    **r/{top_subreddit}** is the dominant community with **{top_count:,} posts** ({top_percent:.1f}% of total). 
    The distribution pattern suggests {'a diverse set of communities' if top_percent < 20 else 'concentration in a few key subreddits'}, which {'indicates broad discussion across multiple topics' if top_percent < 20 else 'shows focused interest in specific topics'}.
    """
    render_insight_box(insight_text)
    
    
    st.subheader("Score Distribution")
    avg_score = df['score'].mean()
    median_score = df['score'].median()
    max_score = df['score'].max()
    
    fig = px.histogram(
        df,
        x="score",
        nbins=50,
        title="Distribution of Post Scores",
        color_discrete_sequence=["#1e88e5"]
    )
    st.plotly_chart(fig, use_container_width=True)
    

    skew_description = "right-skewed (most posts have lower scores with a few highly upvoted outliers)" if avg_score < median_score else "relatively balanced"
    insight_text = f"""
    **Interpretation:** This histogram displays the distribution of post scores, which is {skew_description}. 
    The average score is **{avg_score:.1f}** with a maximum of **{max_score}**. 
    {'This pattern is typical of social media platforms where a small percentage of content receives significant engagement.' if avg_score < median_score else 'This suggests more uniform engagement across posts than typically seen on social platforms.'}
    """
    render_insight_box(insight_text)
    
    
    st.subheader("Community Network Analysis")
    
    show_network = st.checkbox("Show Author-Subreddit Network Graph", value=False)
    
    if show_network:
        with st.spinner("Generating network graph..."):
            network_results = advanced_agent.generate_network_graph()
            
            if "error" in network_results:
                st.warning(network_results["error"])
                st.info("Network analysis requires data with author and subreddit information, with multiple posts by the same authors.")
            else:
                network_data = network_results["network_data"]
                graph_stats = network_results["graph_stats"]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Subreddits", f"{graph_stats['num_subreddits']}")
                with col2:
                    st.metric("Authors", f"{graph_stats['num_authors']}")
                with col3:
                    st.metric("Connections", f"{graph_stats['num_edges']}")
                
                
                fig = go.Figure()
                
                
                fig.add_trace(go.Scatter(
                    x=network_data["edge_x"],
                    y=network_data["edge_y"],
                    line=dict(width=0.5, color="#cccccc"),
                    hoverinfo="none",
                    mode="lines"
                ))
                
                
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
