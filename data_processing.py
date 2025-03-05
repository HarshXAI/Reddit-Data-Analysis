
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from modules.data_ingestion import DataIngestionAgent
import config

def load_data(uploaded_file=None, use_demo_data=False):
    from modules.stats_analysis import StatsAgent
    
    if uploaded_file is not None:
        ingestion_agent = DataIngestionAgent(uploaded_file)
        df = ingestion_agent.get_dataframe()
    
    elif use_demo_data:
        if os.path.exists(config.DEMO_DATA_PATH):
            ingestion_agent = DataIngestionAgent(config.DEMO_DATA_PATH)
            df = ingestion_agent.get_dataframe()
        else:
            df = generate_synthetic_data()
    else:
        return None, None
    stats_agent = StatsAgent(df)
    
    return df, stats_agent

def generate_synthetic_data():
    synthetic_data = []
    base_date = datetime.now() - timedelta(days=30)
    
    subreddits = ["WorldNews", "Technology", "Science", "Gaming", "Politics"]
    topics = ["AI", "Climate", "Elections", "Space", "Economy"]
    
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
    df = pd.DataFrame(synthetic_data)
    
    df['created_date'] = pd.to_datetime(df['created_utc'], unit='s')
    df['date'] = df['created_date'].dt.date
    df['year'] = df['created_date'].dt.year
    df['month'] = df['created_date'].dt.month
    df['day'] = df['created_date'].dt.day
    df['day_of_week'] = df['created_date'].dt.dayofweek
    df['hour'] = df['created_date'].dt.hour
    
    return df
