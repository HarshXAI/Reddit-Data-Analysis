import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import networkx as nx
from bertopic import BERTopic
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import nltk
import requests
from datetime import datetime, timedelta
import re
import os
import logging
import traceback
from typing import Dict, List, Tuple, Union, Optional
from urllib.parse import urlparse
from modules.credibility_analyzer import CredibilityAnalyzer

# Set up logging
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

class AdvancedAnalysisAgent:
    """Agent responsible for advanced AI/ML analysis on Reddit data."""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the AdvancedAnalysisAgent with processed Reddit data.
        
        Args:
            df: DataFrame containing Reddit post data
        """
        self.df = df
        self._prepare_text_data()
        self.credibility_analyzer = CredibilityAnalyzer()
        
    def _prepare_text_data(self) -> None:
        """Prepare text data for advanced analysis."""
        # Combine title and selftext if available for comprehensive text analysis
        if 'selftext' in self.df.columns:
            self.df['combined_text'] = self.df['title'] + ' ' + self.df['selftext'].fillna('')
        else:
            self.df['combined_text'] = self.df['title']
            
        # Ensure text fields are strings
        self.df['combined_text'] = self.df['combined_text'].astype(str)
        self.df['title'] = self.df['title'].astype(str)
        
        # Extract all URLs from text for misinformation analysis
        self.df['urls'] = self.df['combined_text'].apply(self._extract_urls)
    
    def _extract_urls(self, text: str) -> List[str]:
        """Extract all URLs from text content."""
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        return re.findall(url_pattern, text)
    
    def generate_bert_topics(self, n_topics: int = 10) -> Dict:
        """
        Generate topics using BERTopic.
        
        Args:
            n_topics: Number of topics to generate
            
        Returns:
            Dictionary containing topic model, topic info, topic terms, representative documents, and topic evolution visualization data.
        """
        if len(self.df) < 10:
            return {"error": "Not enough data for topic modeling"}
            
        try:
            # Import UMAP and BERTopic
            from umap import UMAP
            import bertopic
            logger.info(f"Using BERTopic version: {bertopic.__version__}")
            
            # Create a UMAP model for dimensionality reduction
            umap_model = UMAP(
                n_components=5,
                n_neighbors=15,
                min_dist=0.1,
                metric='cosine',
                random_state=42
            )
            
            # Initialize BERTopic model with the UMAP model
            model = BERTopic(
                umap_model=umap_model,
                min_topic_size=5,
                nr_topics=n_topics,
                calculate_probabilities=False,
                verbose=False
            )
            
            # Ensure combined_text is a list of strings
            texts = self.df['combined_text'].dropna().astype(str).tolist()
            topics, probs = model.fit_transform(texts)
            
            # Get topic information
            topic_info = model.get_topic_info()
            
            # Retrieve topics from the model as a dictionary
            topics_dict = model.get_topics()
            topic_terms = {}
            for topic, words in topics_dict.items():
                if topic != -1 and words is not None:
                    # words is a list of tuples (word, score)
                    topic_terms[topic] = [word for word, _ in words]
            
            # Get representative document titles for each topic
            topic_docs = {}
            for topic_id in topic_terms.keys():
                indices = [i for i, t in enumerate(topics) if t == topic_id]
                if indices:
                    topic_docs[topic_id] = self.df.iloc[indices]['title'].head(3).tolist()
                else:
                    topic_docs[topic_id] = []
            
            # Assign topics to the DataFrame
            self.df['topic_id'] = topics
            
            # Generate topic evolution visualization data if created_date is available
            topic_evolution_df = pd.DataFrame()
            if 'created_date' in self.df.columns:
                self.df['date'] = self.df['created_date'].dt.date
                topics_over_time = model.topics_over_time(
                    texts,
                    self.df['topic_id'].tolist(),
                    self.df['created_date'].tolist()
                )
                
                topic_evolution = []
                for topic, _ in topics_dict.items():
                    if topic != -1:
                        topic_data = [d for d in topics_over_time if d[0] == topic]
                        if topic_data:
                            for entry in topic_data:
                                topic_evolution.append({
                                    'topic': f"Topic {topic}",
                                    'date': entry[1],
                                    'weight': entry[2]
                                })
                
                topic_evolution_df = pd.DataFrame(topic_evolution)
            
            return {
                "model": model,
                "topic_info": topic_info,
                "topic_terms": topic_terms,
                "topic_docs": topic_docs,
                "topic_evolution": topic_evolution_df
            }
            
        except Exception as e:
            logger.error(f"Error in BERTopic: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                "error": f"Error in topic modeling: {str(e)}",
                "fallback": self._generate_simple_topics(n_topics)
            }
    
    def _generate_simple_topics(self, n_topics: int = 5) -> Dict:
        """
        Generate simple topics using TF-IDF as fallback method.
        
        Args:
            n_topics: Number of topics to generate
            
        Returns:
            Dictionary containing topics and terms
        """
        from sklearn.decomposition import NMF
        
        # Create TF-IDF matrix
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            max_df=0.7,
            min_df=10
        )
        
        try:
            tfidf = vectorizer.fit_transform(self.df['combined_text'])
            feature_names = vectorizer.get_feature_names_out()
            
            # Apply NMF for topic modeling
            nmf = NMF(n_components=n_topics, random_state=42)
            nmf_results = nmf.fit_transform(tfidf)
            
            # Get top words for each topic
            topic_terms = {}
            for topic_idx, topic in enumerate(nmf.components_):
                top_words_idx = topic.argsort()[:-11:-1]
                top_words = [feature_names[i] for i in top_words_idx]
                topic_terms[topic_idx] = top_words
            
            # Assign topics to documents
            self.df['topic_id'] = nmf_results.argmax(axis=1)
            
            # Get example docs for each topic
            topic_docs = {}
            for topic_id in range(n_topics):
                docs = self.df[self.df['topic_id'] == topic_id]['title'].head(3).tolist()
                topic_docs[topic_id] = docs
            
            return {
                "topic_terms": topic_terms,
                "topic_docs": topic_docs
            }
        
        except Exception as e:
            return {"error": f"Topic modeling failed: {str(e)}"}
    
    def detect_trends(self, time_window: str = 'D', min_count: int = 5) -> Dict:
        """
        Detect trending keywords over time.
        
        Args:
            time_window: Time window for grouping ('D'=day, 'W'=week, 'M'=month)
            min_count: Minimum count threshold for a keyword to be considered
            
        Returns:
            Dictionary containing trending keywords and visualization data
        """
        if 'created_date' not in self.df.columns:
            return {"error": "Timestamp data not available for trend detection"}
        
        try:
            # Extract and tokenize important words from titles
            from nltk.tokenize import word_tokenize
            from nltk.corpus import stopwords
            stop_words = set(stopwords.words('english'))
            
            def extract_keywords(text):
                """Extract keywords from text, excluding stopwords"""
                tokens = word_tokenize(text.lower())
                # Filter out short words, non-alphabetic tokens, and stopwords
                keywords = [word for word in tokens 
                          if len(word) > 3 
                          and word.isalpha()
                          and word not in stop_words]
                return keywords
            
            # Apply keyword extraction to all titles
            self.df['keywords'] = self.df['title'].apply(extract_keywords)
            
            # Group by date period
            self.df['period'] = self.df['created_date'].dt.to_period(time_window)
            
            # Count keywords by period
            keyword_trends = {}
            periods = sorted(self.df['period'].unique())
            
            for period in periods:
                period_df = self.df[self.df['period'] == period]
                all_keywords = [word for keywords in period_df['keywords'] for word in keywords]
                keyword_counts = Counter(all_keywords)
                keyword_trends[period] = keyword_counts
            
            # Find trending keywords (significant increase between periods)
            trending_words = []
            for i in range(1, len(periods)):
                prev_period = periods[i-1]
                curr_period = periods[i]
                prev_counts = keyword_trends[prev_period]
                curr_counts = keyword_trends[curr_period]
                for word, count in curr_counts.items():
                    if count >= min_count:
                        prev_count = prev_counts.get(word, 0)
                        if prev_count > 0:
                            increase_ratio = count / prev_count
                            if increase_ratio > 1.5:  # 50% increase threshold
                                trending_words.append({
                                    'word': word,
                                    'period': str(curr_period),
                                    'count': count,
                                    'increase_ratio': increase_ratio
                                })
                        else:
                            trending_words.append({
                                'word': word,
                                'period': str(curr_period),
                                'count': count,
                                'increase_ratio': float('inf')
                            })
            
            trending_df = pd.DataFrame(trending_words)
            if not trending_df.empty:
                trending_df = trending_df.sort_values('increase_ratio', ascending=False)
            
            # Create time series data for visualization
            trend_viz_data = []
            top_keywords = set()
            if not trending_df.empty:
                top_keywords = set(trending_df.head(10)['word'])
            if len(top_keywords) < 5:
                all_keywords = [word for keywords in self.df['keywords'] for word in keywords]
                top_overall = [word for word, _ in Counter(all_keywords).most_common(10)]
                top_keywords.update(top_overall)
                top_keywords = set(list(top_keywords)[:10])
            
            for period in periods:
                period_str = str(period)
                counts = keyword_trends[period]
                for keyword in top_keywords:
                    trend_viz_data.append({
                        'period': period_str,
                        'keyword': keyword,
                        'count': counts.get(keyword, 0)
                    })
            
            trend_timeseries = pd.DataFrame(trend_viz_data)
            
            return {
                "trending_keywords": trending_df,
                "trend_timeseries": trend_timeseries,
                "top_keywords": list(top_keywords)
            }
            
        except Exception as e:
            return {"error": f"Error in trend detection: {str(e)}"}
    
    def score_credibility(self) -> pd.DataFrame:
        """
        Score the credibility of posts using the CredibilityAnalyzer.
        
        Returns:
            DataFrame with original columns plus credibility scores and factors
        """
        try:
            logger.info("Analyzing content credibility...")
            result_df = self.credibility_analyzer.batch_analyze_posts(self.df)
            score_distribution = result_df['credibility_score'].describe()
            logger.info(f"Credibility score distribution: {score_distribution}")
            return result_df
            
        except Exception as e:
            logger.error(f"Error in credibility analysis: {str(e)}")
            logger.error(traceback.format_exc())
            return pd.DataFrame({
                'error': [f"Failed to analyze credibility: {str(e)}"]
            })
    
    def generate_network_graph(self) -> Dict:
        """
        Generate network graph visualization for community analysis.
        
        Returns:
            Dictionary containing network data and visualization
        """
        try:
            if 'author' not in self.df.columns or 'subreddit' not in self.df.columns:
                return {"error": "Author and subreddit data required for network analysis"}
            
            filtered_df = self.df[self.df['author'] != '[deleted]'].copy()
            if len(filtered_df) < 5:
                return {"error": "Not enough author data for network analysis"}
            
            G = nx.Graph()
            author_subreddit_count = {}
            
            for _, post in filtered_df.iterrows():
                author = post['author']
                subreddit = post['subreddit']
                key = (author, subreddit)
                author_subreddit_count[key] = author_subreddit_count.get(key, 0) + 1
            
            authors = set()
            subreddits = set()
            for (author, subreddit), count in author_subreddit_count.items():
                if count >= 1:
                    if author not in authors:
                        G.add_node(author, type='author', size=5)
                        authors.add(author)
                    if subreddit not in subreddits:
                        G.add_node(subreddit, type='subreddit', size=10)
                        subreddits.add(subreddit)
                    G.add_edge(author, subreddit, weight=count)
            
            if len(G.nodes) > 100:
                subreddit_counts = filtered_df['subreddit'].value_counts()
                top_subreddits = set(subreddit_counts.head(30).index)
                nodes_to_remove = [node for node in G.nodes() if G.nodes[node]['type'] == 'subreddit' and node not in top_subreddits]
                G.remove_nodes_from(nodes_to_remove)
                
                nodes_to_remove = [node for node in G.nodes() if G.nodes[node]['type'] == 'author' and G.degree[node] < 2]
                G.remove_nodes_from(nodes_to_remove)
            
            pos = nx.spring_layout(G, seed=42)
            node_x, node_y, node_text, node_size, node_color = [], [], [], [], []
            
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_text.append(node)
                if G.nodes[node]['type'] == 'subreddit':
                    node_color.append('red')
                    count = filtered_df[filtered_df['subreddit'] == node].shape[0]
                    node_size.append(min(20 + count, 50))
                else:
                    node_color.append('blue')
                    node_size.append(10)
            
            edge_x, edge_y, edge_width = [], [], []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                weight = G.edges[edge]['weight']
                edge_width.append(max(1, min(weight, 5)))
            
            network_data = {
                "node_x": node_x,
                "node_y": node_y,
                "node_text": node_text,
                "node_size": node_size,
                "node_color": node_color,
                "edge_x": edge_x,
                "edge_y": edge_y,
                "edge_width": edge_width
            }
            
            return {
                "network_data": network_data,
                "graph_stats": {
                    "num_nodes": len(G.nodes),
                    "num_edges": len(G.edges),
                    "num_authors": len([n for n in G.nodes if G.nodes[n]['type'] == 'author']),
                    "num_subreddits": len([n for n in G.nodes if G.nodes[n]['type'] == 'subreddit'])
                }
            }
            
        except Exception as e:
            return {"error": f"Error in network graph generation: {str(e)}"}
    
    def generate_topics(self, n_topics: int = 10) -> Dict:
        """
        Generate topics using NMF (Non-negative Matrix Factorization).
        
        Args:
            n_topics: Number of topics to generate
            
        Returns:
            Dictionary containing topic info, topic terms, and representative documents.
        """
        if len(self.df) < 10:
            return {"error": "Not enough data for topic modeling"}
            
        try:
            from sklearn.decomposition import NMF
            
            # Create TF-IDF matrix
            vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                max_df=0.7,
                min_df=10
            )
            
            tfidf = vectorizer.fit_transform(self.df['combined_text'])
            feature_names = vectorizer.get_feature_names_out()
            
            # Apply NMF for topic modeling
            nmf = NMF(n_components=n_topics, random_state=42)
            nmf_results = nmf.fit_transform(tfidf)
            
            # Get top words for each topic
            topic_terms = {}
            for topic_idx, topic in enumerate(nmf.components_):
                top_words_idx = topic.argsort()[:-11:-1]
                top_words = [feature_names[i] for i in top_words_idx]
                topic_terms[topic_idx] = top_words
            
            # Assign topics to documents
            self.df['topic_id'] = nmf_results.argmax(axis=1)
            
            # Get example docs for each topic
            topic_docs = {}
            for topic_id in range(n_topics):
                docs = self.df[self.df['topic_id'] == topic_id]['title'].head(3).tolist()
                topic_docs[topic_id] = docs
            
            # Create topic evolution data if timestamps are available
            topic_evolution_df = pd.DataFrame()
            if 'created_date' in self.df.columns:
                self.df['date'] = self.df['created_date'].dt.date
                evolution_data = []
                
                for date in sorted(self.df['date'].unique()):
                    date_mask = self.df['date'] == date
                    topic_dist = self.df[date_mask]['topic_id'].value_counts()
                    total = topic_dist.sum()
                    
                    for topic_id in range(n_topics):
                        count = topic_dist.get(topic_id, 0)
                        evolution_data.append({
                            'topic': f"Topic {topic_id}",
                            'date': date,
                            'weight': count / total if total > 0 else 0
                        })
                
                topic_evolution_df = pd.DataFrame(evolution_data)
            
            return {
                "topic_terms": topic_terms,
                "topic_docs": topic_docs,
                "topic_evolution": topic_evolution_df
            }
            
        except Exception as e:
            logger.error(f"Error in topic modeling: {str(e)}")
            logger.error(traceback.format_exc())
            return {"error": f"Topic modeling failed: {str(e)}"}