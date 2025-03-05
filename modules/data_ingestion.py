import pandas as pd
import json
from datetime import datetime
import re
from typing import Union, List, Dict, Any
import io

class DataIngestionAgent:

    def __init__(self, file_path: Union[str, io.BytesIO]):

        self.df = self._load_data(file_path)
        self._preprocess_data()
    
    def _load_data(self, file_path: Union[str, io.BytesIO]) -> pd.DataFrame:

        try:
            if isinstance(file_path, str):

                posts = []
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            try:
                                item = json.loads(line)
                                if isinstance(item, dict) and 'data' in item:
                                    posts.append(item['data'])
                            except json.JSONDecodeError:
                                continue
            else:
                posts = []
                content = file_path.getvalue().decode('utf-8')
                for line in content.splitlines():
                    if line.strip():
                        try:
                            item = json.loads(line)
                            if isinstance(item, dict) and 'data' in item:
                                posts.append(item['data'])
                        except json.JSONDecodeError:
                            continue
            
            if not posts:
                raise ValueError("No valid Reddit posts found in the file")
                
            df = pd.DataFrame(posts)
            return df
            
        except Exception as e:
            raise ValueError(f"Error loading JSONL data: {str(e)}")
    
    def _preprocess_data(self):
        if 'created_utc' in self.df.columns:
            self.df['created_date'] = pd.to_datetime(self.df['created_utc'], unit='s')
            
            self.df['date'] = self.df['created_date'].dt.date
            self.df['year'] = self.df['created_date'].dt.year
            self.df['month'] = self.df['created_date'].dt.month
            self.df['day'] = self.df['created_date'].dt.day
            self.df['day_of_week'] = self.df['created_date'].dt.dayofweek
            self.df['hour'] = self.df['created_date'].dt.hour
        
        if 'selftext' in self.df.columns:
            self.df['selftext'] = self.df['selftext'].fillna('')
        
        if 'score' in self.df.columns:
            self.df['score'] = pd.to_numeric(self.df['score'], errors='coerce').fillna(0).astype(int)
    
    def get_dataframe(self) -> pd.DataFrame:
        return self.df
