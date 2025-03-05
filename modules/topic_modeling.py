import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
import numpy as np
from typing import List, Tuple, Dict, Any
import nltk
from nltk.corpus import stopwords

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

class TopicModelAgent:
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self._prepare_text_data()
        self.stop_words = list(stopwords.words('english'))
        reddit_stopwords = ['amp', 'x200b', 'https', 'http', 'www', 'com', 
                           'reddit', 'like', 'just', 'post', 'get', 'would']
        self.stop_words.extend(reddit_stopwords)
    
    def _prepare_text_data(self):
        if 'selftext' in self.df.columns:
            self.df['combined_text'] = self.df['title'] + ' ' + self.df['selftext'].fillna('')
        else:
            self.df['combined_text'] = self.df['title']
        
        self.df['combined_text'] = self.df['combined_text'].astype(str)
    
    def generate_topics(self, n_topics: int = 5, method: str = 'lda') -> List[Tuple[int, List[str], List[str]]]:

        vectorizer = CountVectorizer(
            stop_words='english',
            max_df=0.95,
            min_df=2,
            max_features=10000
        )
        
        X = vectorizer.fit_transform(self.df['combined_text'])
        feature_names = vectorizer.get_feature_names_out()
        if method == 'nmf':
            model = NMF(n_components=n_topics, random_state=42)
        else:
            model = LatentDirichletAllocation(
                n_components=n_topics,
                max_iter=10,
                learning_method='online',
                random_state=42
            )
        
        topic_word_matrix = model.fit_transform(X)
        
        def get_topic_words(topic_idx, top_n=10):
            topic = model.components_[topic_idx]
            top_word_indices = topic.argsort()[:-top_n-1:-1]
            return [feature_names[i] for i in top_word_indices]
        
        doc_topic_matrix = model.transform(X)
        
        topics = []
        for topic_idx in range(n_topics):
            topic_words = get_topic_words(topic_idx)
            doc_indices = doc_topic_matrix[:, topic_idx].argsort()[::-1][:5]
            example_docs = self.df.iloc[doc_indices]['title'].tolist()
            
            topics.append((topic_idx, topic_words, example_docs))
        
        return topics