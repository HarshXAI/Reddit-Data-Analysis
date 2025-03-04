import pandas as pd
import numpy as np
from wordcloud import WordCloud
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords

# Download NLTK resources
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

class StatsAgent:
    """Agent responsible for statistical analysis of Reddit data."""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the StatsAgent with a DataFrame.
        
        Args:
            df: DataFrame containing Reddit post data
        """
        self.df = df
        self._initialize_stopwords()
    
    def _initialize_stopwords(self):
        """Initialize stopwords for text analysis."""
        self.stop_words = set(stopwords.words('english'))
        # Add Reddit-specific stopwords
        reddit_stopwords = {'amp', 'x200b', 'https', 'http', 'www', 'com', 
                           'reddit', 'like', 'just', 'post', 'get', 'would'}
        self.stop_words.update(reddit_stopwords)
    
    def get_unique_subreddit_count(self) -> int:
        """Return the number of unique subreddits."""
        return self.df['subreddit'].nunique()
    
    def get_average_score(self) -> float:
        """Return the average score of posts."""
        return self.df['score'].mean()
    
    def get_subreddit_distribution(self) -> pd.DataFrame:
        """Return the distribution of posts by subreddit."""
        subreddit_counts = self.df['subreddit'].value_counts().reset_index()
        subreddit_counts.columns = ['subreddit', 'count']
        return subreddit_counts
    
    def get_posts_over_time(self) -> dict:
        """Return the number of posts over different time periods."""
        # Posts by day
        posts_by_day = self.df.groupby('date').size().reset_index(name='count')
        
        # Posts by week
        self.df['week'] = self.df['created_date'].dt.isocalendar().week
        self.df['year_week'] = self.df['created_date'].dt.strftime('%Y-%U')
        posts_by_week = self.df.groupby('year_week').size().reset_index(name='count')
        posts_by_week['date'] = pd.to_datetime(posts_by_week['year_week'] + '-0', format='%Y-%U-%w')
        
        # Posts by month
        self.df['year_month'] = self.df['created_date'].dt.strftime('%Y-%m')
        posts_by_month = self.df.groupby('year_month').size().reset_index(name='count')
        posts_by_month['date'] = pd.to_datetime(posts_by_month['year_month'] + '-01', format='%Y-%m-%d')
        
        return {
            'day': posts_by_day,
            'week': posts_by_week[['date', 'count']],
            'month': posts_by_month[['date', 'count']]
        }
    
    def get_posts_by_day_of_week(self) -> pd.DataFrame:
        """Return the distribution of posts by day of week."""
        day_names = {
            0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday',
            4: 'Friday', 5: 'Saturday', 6: 'Sunday'
        }
        
        posts_by_dow = self.df.groupby('day_of_week').size().reset_index(name='count')
        posts_by_dow['day_name'] = posts_by_dow['day_of_week'].map(day_names)
        posts_by_dow = posts_by_dow.sort_values('day_of_week')
        
        return posts_by_dow
    
    def get_posts_by_hour(self) -> pd.DataFrame:
        """Return the distribution of posts by hour of day."""
        posts_by_hour = self.df.groupby('hour').size().reset_index(name='count')
        return posts_by_hour
    
    def _clean_text(self, text: str) -> str:
        """Clean text by removing special characters and stopwords."""
        if not isinstance(text, str):
            return ""
            
        # Remove URLs, special characters, and convert to lowercase
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = text.lower()
        
        # Remove stopwords
        words = text.split()
        words = [word for word in words if word not in self.stop_words and len(word) > 2]
        
        return ' '.join(words)
    
    def generate_title_wordcloud(self) -> WordCloud:
        """Generate a word cloud from post titles."""
        # Clean titles
        cleaned_titles = self.df['title'].astype(str).apply(self._clean_text)
        all_text = ' '.join(cleaned_titles)
        
        # Generate word cloud
        wordcloud = WordCloud(
            width=800, height=400,
            background_color='white',
            max_words=200,
            contour_width=3,
            colormap='viridis'
        ).generate(all_text)
        
        return wordcloud
    
    def get_top_keywords_in_titles(self, n: int = 20) -> pd.DataFrame:
        """Return the top n keywords in post titles."""
        # Clean titles
        cleaned_titles = self.df['title'].astype(str).apply(self._clean_text)
        all_words = ' '.join(cleaned_titles).split()
        
        # Count word frequencies
        word_counts = Counter(all_words)
        top_words = word_counts.most_common(n)
        
        # Convert to DataFrame
        keywords_df = pd.DataFrame(top_words, columns=['word', 'count'])
        
        return keywords_df
    
    def search_posts(self, keyword: str) -> pd.DataFrame:
        """Search for posts containing a specific keyword."""
        keyword = keyword.lower()
        
        # Search in titles and selftext
        title_mask = self.df['title'].astype(str).str.lower().str.contains(keyword)
        
        if 'selftext' in self.df.columns:
            selftext_mask = self.df['selftext'].astype(str).str.lower().str.contains(keyword)
            mask = title_mask | selftext_mask
        else:
            mask = title_mask
        
        return self.df[mask]
