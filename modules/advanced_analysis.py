
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
from typing import Dict, List, Tuple, Union, Optional
from urllib.parse import urlparse

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
            Dictionary containing topic model, topics, and visualizations
        """
        # Check if we have enough data
        if len(self.df) < 10:
            return {"error": "Not enough data for topic modeling"}
            
        try:
            # Initialize BERTopic model with reduced dimensionality for speed
            model = BERTopic(
                n_components=5,
                min_topic_size=5,
                nr_topics=n_topics,
                calculate_probabilities=False,
                verbose=False
            )
            
            # Fit the model on our combined text
            topics, probs = model.fit_transform(self.df['combined_text'])
            
            # Get topic info
            topic_info = model.get_topic_info()
            
            # Get topic terms for each topic
            topic_terms = {}
            for topic in range(len(model.get_topics())):
                if topic != -1:  # Skip outlier topic (-1)
                    topic_terms[topic] = [word for word, _ in model.get_topic(topic)]
            
            # Get representative documents for each topic
            topic_docs = {}
            for topic_id in topic_terms.keys():
                indices = [i for i, t in enumerate(topics) if t == topic_id]
                if indices:
                    topic_docs[topic_id] = self.df.iloc[indices]['title'].head(3).tolist()
                else:
                    topic_docs[topic_id] = []
            
            # Assign topics to the DataFrame for further analysis
            self.df['topic_id'] = topics
            
            # Generate visualization data
            if 'created_date' in self.df.columns:
                # Prepare data for topic over time visualization
                self.df['date'] = self.df['created_date'].dt.date
                topics_over_time = model.topics_over_time(
                    self.df['combined_text'],
                    self.df['topic_id'],
                    self.df['created_date']
                )
                
                # Format for Plotly visualization
                topic_evolution = []
                for topic_id in topic_terms.keys():
                    if topic_id != -1:  # Skip outlier topic
                        topic_data = [d for d in topics_over_time if d[0] == topic_id]
                        if topic_data:
                            for entry in topic_data:
                                topic_evolution.append({
                                    'topic': f"Topic {topic_id}",
                                    'date': entry[1],
                                    'weight': entry[2]
                                })
                
                topic_evolution_df = pd.DataFrame(topic_evolution)
            else:
                topic_evolution_df = pd.DataFrame()
            
            return {
                "model": model,
                "topic_info": topic_info,
                "topic_terms": topic_terms,
                "topic_docs": topic_docs,
                "topic_evolution": topic_evolution_df
            }
            
        except Exception as e:
            # Fallback to a simpler representation if BERTopic fails
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
            
            def extract_keywords(text):
                """Extract keywords from text, excluding stopwords"""
                tokens = word_tokenize(text.lower())
                # Filter out short words and common stopwords
                keywords = [word for word in tokens 
                          if len(word) > 3 
                          and word.isalpha()
                          and word not in nltk.corpus.stopwords.words('english')]
                return keywords
            
            # Apply keyword extraction to all titles
            self.df['keywords'] = self.df['title'].apply(extract_keywords)
            
            # Group by date
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
                            # New word that wasn't in previous period
                            trending_words.append({
                                'word': word,
                                'period': str(curr_period),
                                'count': count,
                                'increase_ratio': float('inf')  # Infinity for new words
                            })
            
            # Sort by increase ratio (most trending first)
            trending_df = pd.DataFrame(trending_words)
            if not trending_df.empty:
                trending_df = trending_df.sort_values('increase_ratio', ascending=False)
            
            # Create time series data for visualization
            trend_viz_data = []
            
            # Get top 10 trending keywords
            top_keywords = set()
            if not trending_df.empty:
                top_keywords = set(trending_df.head(10)['word'])
            
            # If we don't have enough trending keywords, add some top overall keywords
            if len(top_keywords) < 5:
                all_keywords = [word for keywords in self.df['keywords'] for word in keywords]
                top_overall = [word for word, _ in Counter(all_keywords).most_common(10)]
                top_keywords.update(top_overall)
                top_keywords = set(list(top_keywords)[:10])  # Limit to 10
            
            # Create time series for each keyword
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
        Assign credibility scores to posts based on various factors.
        
        Returns:
            DataFrame with credibility scores
        """
        # Create a copy to avoid modifying the original dataframe
        credibility_df = self.df.copy()
        
        # Initialize credibility score (50 = neutral)
        credibility_df['credibility_score'] = 50
        credibility_df['credibility_factors'] = ''
        
        try:
            # Factor 1: Check for presence of external links
            # Posts with reputable external sources tend to be more credible
            credibility_df['has_links'] = credibility_df['urls'].apply(lambda x: len(x) > 0)
            
            # Apply link bonus/penalty based on domain reputation
            def score_links(urls):
                if not urls:
                    return 0, []
                
                # Simple list of trusted and untrusted domains
                trusted_domains = {
                    'wikipedia.org', 'nytimes.com', 'reuters.com', 'bbc.com', 
                    'theguardian.com', 'washingtonpost.com', 'nih.gov',
                    'edu', 'gov', 'nature.com', 'science.org', 'who.int'
                }
                
                untrusted_domains = {
                    'example-fake-news.com', 'totallyrealnews.com'
                }
                
                score = 0
                factors = []
                
                for url in urls:
                    try:
                        domain = urlparse(url).netloc.lower()
                        if any(trusted in domain for trusted in trusted_domains):
                            score += 10
                            factors.append(f"Trusted source: {domain}")
                        elif any(untrusted in domain for untrusted in untrusted_domains):
                            score -= 20
                            factors.append(f"Potentially unreliable source: {domain}")
                    except:
                        pass
                
                return score, factors
            
            # Apply link scoring
            link_scores = credibility_df['urls'].apply(score_links)
            credibility_df['link_score'] = [score[0] for score in link_scores]
            link_factors = [score[1] for score in link_scores]
            
            # Factor 2: Text analysis for misinformation patterns
            # Use TF-IDF to detect language patterns common in misleading content
            misinformation_patterns = [
                "they don't want you to know", "wake up", "mainstream media won't tell you",
                "secret", "conspiracy", "they are hiding", "truth revealed",
                "what they don't tell you", "government doesn't want you to know"
            ]
            
            def check_misinformation_patterns(text):
                text = text.lower()
                matched_patterns = [pattern for pattern in misinformation_patterns if pattern in text]
                score = -5 * len(matched_patterns)
                return score, matched_patterns
            
            misinfo_results = credibility_df['combined_text'].apply(check_misinformation_patterns)
            credibility_df['pattern_score'] = [result[0] for result in misinfo_results]
            misinfo_factors = [result[1] for result in misinfo_results]
            
            # Factor 3: Extreme sentiment or all-caps
            def check_style_red_flags(text):
                score = 0
                factors = []
                
                # Check for excessive capitalization (shouting)
                if len(text) > 20:
                    capitals = sum(1 for c in text if c.isupper())
                    cap_ratio = capitals / len(text)
                    if cap_ratio > 0.7:
                        score -= 10
                        factors.append("Excessive capitalization")
                
                # Check for excessive punctuation
                exclamations = text.count('!')
                if exclamations > 3:
                    score -= min(exclamations, 10)
                    factors.append("Excessive punctuation")
                
                # Check for multiple question marks (questioning reality)
                questions = text.count('?')
                if questions > 3:
                    score -= min(questions, 5)
                    factors.append("Multiple rhetorical questions")
                
                return score, factors
            
            style_results = credibility_df['title'].apply(check_style_red_flags)
            credibility_df['style_score'] = [result[0] for result in style_results]
            style_factors = [result[1] for result in style_results]
            
            # Factor 4: User karma/reputation (if available)
            if 'author_karma' in credibility_df.columns:
                # Normalize author karma to a score between -10 and +10
                karma_scores = credibility_df['author_karma'].fillna(0)
                max_karma = max(1000, karma_scores.max())  # Cap at 1000 or actual max
                credibility_df['karma_score'] = (karma_scores / max_karma * 10).clip(-10, 10)
            else:
                credibility_df['karma_score'] = 0
            
            # Combine all factors
            credibility_df['credibility_score'] = (
                50 +  # Neutral starting point
                credibility_df['link_score'] +
                credibility_df['pattern_score'] + 
                credibility_df['style_score'] +
                credibility_df['karma_score']
            ).clip(0, 100)  # Keep score between 0 and 100
            
            # Combine explanation factors
            for i, row in credibility_df.iterrows():
                factors = []
                if len(link_factors[i]) > 0:
                    factors.extend(link_factors[i])
                if len(misinfo_factors[i]) > 0:
                    factors.append(f"Contains potential misinformation patterns: {', '.join(misinfo_factors[i])}")
                if len(style_factors[i]) > 0:
                    factors.extend(style_factors[i])
                
                credibility_df.at[i, 'credibility_factors'] = "; ".join(factors)
            
            # Create probability of misinformation column
            credibility_df['misinfo_probability'] = (100 - credibility_df['credibility_score']) / 100
            
            return credibility_df[['title', 'subreddit', 'credibility_score', 'misinfo_probability', 'credibility_factors']]
            
        except Exception as e:
            credibility_df['error'] = str(e)
            return credibility_df[['title', 'credibility_score', 'error']]

    def generate_network_graph(self) -> Dict:
        """
        Generate network graph visualization for community analysis.
        
        Returns:
            Dictionary containing network data and visualization
        """
        try:
            # Check if we have author and subreddit data
            if 'author' not in self.df.columns or 'subreddit' not in self.df.columns:
                return {"error": "Author and subreddit data required for network analysis"}
            
            # Filter out deleted/removed authors
            filtered_df = self.df[self.df['author'] != '[deleted]'].copy()
            if len(filtered_df) < 5:
                return {"error": "Not enough author data for network analysis"}
            
            # Create a graph
            G = nx.Graph()
            
            # Track subreddit activity per author
            author_subreddit_count = {}
            
            # Add author-subreddit connections
            for _, post in filtered_df.iterrows():
                author = post['author']
                subreddit = post['subreddit']
                
                key = (author, subreddit)
                if key in author_subreddit_count:
                    author_subreddit_count[key] += 1
                else:
                    author_subreddit_count[key] = 1
            
            # Add nodes and edges with weight based on frequency
            authors = set()
            subreddits = set()
            
            for (author, subreddit), count in author_subreddit_count.items():
                if count >= 1:  # Only include if author posted multiple times
                    if author not in authors:
                        G.add_node(author, type='author', size=5)
                        authors.add(author)
                    
                    if subreddit not in subreddits:
                        G.add_node(subreddit, type='subreddit', size=10)
                        subreddits.add(subreddit)
                    
                    G.add_edge(author, subreddit, weight=count)
            
            # Limit graph size for visualization
            if len(G.nodes) > 100:
                # Keep top subreddits by post count
                subreddit_counts = filtered_df['subreddit'].value_counts()
                top_subreddits = set(subreddit_counts.head(30).index)
                
                # Filter graph to only include top subreddits
                nodes_to_remove = []
                for node in G.nodes:
                    if G.nodes[node]['type'] == 'subreddit' and node not in top_subreddits:
                        nodes_to_remove.append(node)
                
                G.remove_nodes_from(nodes_to_remove)
                
                # Further reduce by removing low-degree author nodes
                nodes_to_remove = []
                for node in G.nodes:
                    if G.nodes[node]['type'] == 'author' and G.degree[node] < 2:
                        nodes_to_remove.append(node)
                
                G.remove_nodes_from(nodes_to_remove)
            
            # Prepare data for Plotly network visualization
            pos = nx.spring_layout(G, seed=42)
            
            node_x = []
            node_y = []
            node_text = []
            node_size = []
            node_color = []
            
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_text.append(node)
                
                if G.nodes[node]['type'] == 'subreddit':
                    node_color.append('red')
                    count = filtered_df[filtered_df['subreddit'] == node].shape[0]
                    node_size.append(min(20 + count, 50))  # Scale by post count, max size 50
                else:
                    node_color.append('blue')
                    node_size.append(10)  # Default author size
            
            # Create edges
            edge_x = []
            edge_y = []
            edge_width = []
            
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                
                # Edge width based on weight
                weight = G.edges[edge]['weight']
                edge_width.append(max(1, min(weight, 5)))  # Between 1-5 based on weight
            
            # Prepare network data for visualization
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
