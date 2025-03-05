import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import pandas as pd
from visualization_helpers import render_insight_box, generate_category_insight

def render(df, stats_agent):
    st.header("Text Analysis")
     
    st.subheader("Word Cloud of Post Titles")
    wordcloud = stats_agent.generate_title_wordcloud()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)
    
    top_keywords = stats_agent.get_top_keywords_in_titles(n=5)
    top_words = ", ".join([f"**{word}**" for word in top_keywords['word'].tolist()])
    
    insight_text = f"""
    **Interpretation:** This word cloud visualizes the most frequent words in post titles, with larger words appearing more frequently.
    The dominant terms ({top_words}) highlight the main topics of discussion in this dataset.
    This provides a quick visual summary of what content is most discussed across the analyzed posts.
    """
    render_insight_box(insight_text)
    
    st.subheader("Top Keywords in Titles")
    keywords = stats_agent.get_top_keywords_in_titles(n=20)
    top_word = keywords.iloc[0]['word']
    top_count = keywords.iloc[0]['count']
    total_words = keywords['count'].sum()
    top_percent = (top_count / total_words) * 100
    
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
    
    word_diversity = "diverse vocabulary with no single dominant term" if top_percent < 10 else "several recurring themes" if top_percent < 20 else "conversation dominated by a few key terms"
    
    insight_text = f"""
    **Interpretation:** This chart ranks the most common words in post titles. 
    The word "**{top_word}**" appears **{top_count} times** ({top_percent:.1f}% of all key terms).
    The distribution shows a {word_diversity}, indicating {'a wide range of topics being discussed' if top_percent < 10 else 'focused discussion on specific themes' if top_percent < 20 else 'concentrated attention on particular subjects'}.
    """
    render_insight_box(insight_text)
    
    st.subheader("Keyword Search")
    search_term = st.text_input("Enter keyword to search in posts:")
    if search_term:
        filtered_posts = stats_agent.search_posts(search_term)
        st.write(f"Found {len(filtered_posts)} posts containing '{search_term}'")
        st.dataframe(filtered_posts[['title', 'subreddit', 'score', 'created_date']])
    
    st.subheader("Basic Topic Modeling")
    
    import config
    from modules.topic_modeling import TopicModelAgent
    
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
