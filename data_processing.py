
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from modules.data_ingestion import DataIngestionAgent
import config

def load_data(uploaded_file=None, use_demo_data=False):
    """
    Load data from either an uploaded file or generate demo data.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        use_demo_data: Flag to use demo data if no file is uploaded
        
    Returns:
        df: Processed DataFrame with Reddit data
        stats_agent: Initialized StatsAgent
    """
    from modules.stats_analysis import StatsAgent
    
    if uploaded_file is not None:
        # Process uploaded file
        ingestion_agent = DataIngestionAgent(uploaded_file)
        df = ingestion_agent.get_dataframe()
    
    elif use_demo_data:
        # Try to load demo data or create synthetic data
        if os.path.exists(config.DEMO_DATA_PATH):
            ingestion_agent = DataIngestionAgent(config.DEMO_DATA_PATH)
            df = ingestion_agent.get_dataframe()
        else:
            # Generate synthetic data
            df = generate_synthetic_data()
    else:
        # No data provided
        return None, None
    
    # Initialize stats agent
    stats_agent = StatsAgent(df)
    
    return df, stats_agent

def generate_synthetic_data():
    """
    Generate synthetic Reddit data for demo purposes.
    
    Returns:
        DataFrame with synthetic Reddit data
    """
    # Generate synthetic data
    synthetic_data = []
    base_date = datetime.now() - timedelta(days=30)
    
    subreddits = ["WorldNews", "Technology", "Science", "Gaming", "Politics"]
    topics = ["AI", "Climate", "Elections", "Space", "Economy"]
    
    # Generate 100 synthetic posts
    for i in range(100):
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
    
    return df
