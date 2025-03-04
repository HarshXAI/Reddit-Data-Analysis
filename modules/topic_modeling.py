import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
import numpy as np
from typing import List, Tuple, Dict, Any
import nltk
from nltk.corpus import stopwords

# Download NLTK resources if needed
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

class TopicModelAgent:
    """Agent responsible for topic modeling on Reddit posts."""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the TopicModelAgent with a DataFrame.
        
        Args:
            df: DataFrame containing Reddit post data
        """
        self.df = df
        self._prepare_text_data()
        # Use list of stopwords instead of set for compatibility with CountVectorizer
        self.stop_words = list(stopwords.words('english'))
        # Add Reddit-specific stopwords
        reddit_stopwords = ['amp', 'x200b', 'https', 'http', 'www', 'com', 
                           'reddit', 'like', 'just', 'post', 'get', 'would']
        self.stop_words.extend(reddit_stopwords)
    
    def _prepare_text_data(self):
        """Prepare text data for topic modeling."""
        # Combine title and selftext if available
        if 'selftext' in self.df.columns:
            self.df['combined_text'] = self.df['title'] + ' ' + self.df['selftext'].fillna('')
        else:
            self.df['combined_text'] = self.df['title']
        
        # Clean the text
        self.df['combined_text'] = self.df['combined_text'].astype(str)
    
    def generate_topics(self, n_topics: int = 5, method: str = 'lda') -> List[Tuple[int, List[str], List[str]]]:
        """
        Generate topics from post text.
        
        Args:
            n_topics: Number of topics to generate
            method: Method to use ('lda' or 'nmf')
            
        Returns:
            List of tuples (topic_id, list_of_keywords, example_docs)
        """
        # Create document-term matrix - use 'english' instead of set of stopwords
        vectorizer = CountVectorizer(
            stop_words='english',  # Use built-in English stopwords
            max_df=0.95,
            min_df=2,
            max_features=10000
        )
        
        # Rest of the method remains the same
        X = vectorizer.fit_transform(self.df['combined_text'])
        feature_names = vectorizer.get_feature_names_out()
        
        # Apply topic modeling
        if method == 'nmf':
            model = NMF(n_components=n_topics, random_state=42)
        else:
            # Default to LDA
            model = LatentDirichletAllocation(
                n_components=n_topics,
                max_iter=10,
                learning_method='online',
                random_state=42
            )
        
        # Fit the model
        topic_word_matrix = model.fit_transform(X)
        
        # Get topic keywords
        def get_topic_words(topic_idx, top_n=10):
            topic = model.components_[topic_idx]
            top_word_indices = topic.argsort()[:-top_n-1:-1]
            return [feature_names[i] for i in top_word_indices]
        
        # Get document-topic distributions
        doc_topic_matrix = model.transform(X)
        
        # For each topic, get the top documents
        topics = []
        for topic_idx in range(n_topics):
            # Get top words for this topic
            topic_words = get_topic_words(topic_idx)
            
            # Get top documents for this topic
            doc_indices = doc_topic_matrix[:, topic_idx].argsort()[::-1][:5]
            example_docs = self.df.iloc[doc_indices]['title'].tolist()
            
            topics.append((topic_idx, topic_words, example_docs))
        
        return topics
