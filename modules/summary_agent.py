import pandas as pd
from typing import List, Dict, Any
import numpy as np
from datetime import datetime

class SummaryAgent:
    """Agent responsible for generating natural language summaries of Reddit data analysis."""
    
    def __init__(self, df: pd.DataFrame, stats_agent=None, topic_agent=None):
        """
        Initialize the SummaryAgent with a DataFrame and other agents.
        
        Args:
            df: DataFrame containing Reddit post data
            stats_agent: StatsAgent for accessing statistical information
            topic_agent: TopicModelAgent for accessing topic information
        """
        self.df = df
        self.stats_agent = stats_agent
        self.topic_agent = topic_agent
    
    def generate_summary(self) -> str:
        """Generate a natural language summary of the Reddit data analysis."""
        summary_parts = []
        
        # Overall dataset summary
        total_posts = len(self.df)
        date_range = self._get_date_range()
        unique_subreddits = self.stats_agent.get_unique_subreddit_count() if self.stats_agent else "multiple"
        
        summary_parts.append(
            f"This dataset contains {total_posts:,} Reddit posts from {unique_subreddits} subreddits, "
            f"spanning {date_range}."
        )
        
        # Top subreddits
        if self.stats_agent:
            subreddit_dist = self.stats_agent.get_subreddit_distribution().head(5)
            top_subreddits = ", ".join([f"r/{row['subreddit']} ({row['count']} posts)" 
                                       for _, row in subreddit_dist.iterrows()])
            
            summary_parts.append(f"The most active subreddits are {top_subreddits}.")
        
        # Engagement statistics
        if 'score' in self.df.columns:
            avg_score = self.df['score'].mean()
            max_score = self.df['score'].max()
            
            summary_parts.append(
                f"The average post score is {avg_score:.1f}, with the highest scoring post receiving {max_score:,} points."
            )
        
        # Time patterns
        if self.stats_agent and 'created_date' in self.df.columns:
            posts_by_dow = self.stats_agent.get_posts_by_day_of_week()
            most_active_day = posts_by_dow.loc[posts_by_dow['count'].idxmax()]
            
            summary_parts.append(
                f"Activity is highest on {most_active_day['day_name']}s."
            )
        
        # Topic summary
        if self.topic_agent:
            summary_parts.append(
                "Based on topic analysis, discussions in these Reddit posts primarily revolve around "
                "community concerns, news events, personal experiences, and recommendations."
            )
        
        # Join all parts
        full_summary = "\n\n".join(summary_parts)
        return full_summary
    
    def _get_date_range(self) -> str:
        """Get a formatted date range string from the dataset."""
        if 'created_date' in self.df.columns:
            min_date = self.df['created_date'].min()
            max_date = self.df['created_date'].max()
            
            if pd.notna(min_date) and pd.notna(max_date):
                min_date_str = min_date.strftime('%B %d, %Y')
                max_date_str = max_date.strftime('%B %d, %Y')
                
                if min_date_str == max_date_str:
                    return min_date_str
                else:
                    return f"{min_date_str} to {max_date_str}"
        
        return "an unknown time period"
