import pandas as pd
import numpy as np
import re
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import requests
from urllib.parse import urlparse
from typing import Dict, List, Any, Tuple, Set
import logging
import config

logger = logging.getLogger(__name__)

class CredibilityAnalyzer:
    
    CREDIBILITY_MARKERS = {
        'evidence': [
            'according to', 'study finds', 'evidence shows', 'data indicates', 
            'researchers found', 'analysis shows', 'statistics reveal',
            'experts say', 'survey indicates', 'sources confirm'
        ],
        'balanced_language': [
            'on the other hand', 'however', 'alternatively', 'in contrast',
            'different perspective', 'opposing view', 'some argue', 
            'critics say', 'debate', 'discussion'
        ],
        'precision': [
            'specifically', 'precisely', 'exactly', 'approximately',
            'estimated', 'about', 'around', 'measured', 'calculated'
        ]
    }
    
    CREDIBILITY_DETRACTORS = {
        'conspiracy': [
            'conspiracy', 'coverup', 'cover-up', 'hoax', 'illuminati', 
            'nwo', 'deep state', 'they don\'t want you to know', 
            'what they\'re hiding', 'secret agenda', 'mind control'
        ],
        'sensationalism': [
            'shocking', 'bombshell', 'unbelievable', 'mind-blowing',
            'you won\'t believe', 'jaw-dropping', 'explosive', 
            'scandalous', 'outrageous', 'banned'
        ],
        'hedging': [
            'maybe', 'perhaps', 'possibly', 'allegedly', 'reportedly',
            'supposedly', 'claimed', 'rumored', 'anonymous sources'
        ],
        'urgency': [
            'urgent', 'breaking', 'alert', 'emergency', 'crisis', 
            'act now', 'limited time', 'warning', 'danger'
        ]
    }
    
    def __init__(self):
        try:
            nltk.data.find('vader_lexicon')
        except LookupError:
            nltk.download('vader_lexicon', quiet=True)
        
        self.sia = SentimentIntensityAnalyzer()
        self.trusted_domains = config.TRUSTED_DOMAINS
        self.untrusted_domains = config.UNTRUSTED_DOMAINS
    
    def analyze_post(self, title: str, text: str, score: int = 0, author: str = '') -> Tuple[int, List[str]]:

        factors = []
        base_score = 50  
        title = str(title or '')
        text = str(text or '')
        combined_text = f"{title} {text}".lower()
        
        score_adjustments = []
        
        if re.search(r'[A-Z]{5,}', title):
            score_adjustments.append((-15, "Uses excessive capitalization"))
        
        if re.search(r'[!?]{2,}', title):
            score_adjustments.append((-10, "Uses excessive punctuation"))
        
        trusted_domains_found = []
        for domain in self.trusted_domains:
            if domain in combined_text:
                trusted_domains_found.append(domain)
        
        if trusted_domains_found:
            score_adjustments.append((15, f"References trusted source(s): {', '.join(trusted_domains_found)}"))
        
        untrusted_domains_found = []
        for domain in self.untrusted_domains:
            if domain in combined_text:
                untrusted_domains_found.append(domain)
        
        if untrusted_domains_found:
            score_adjustments.append((-20, f"References untrusted source(s): {', '.join(untrusted_domains_found)}"))
        
        sentiment = self.sia.polarity_scores(combined_text)
        compound = sentiment['compound']
        
        if abs(compound) > 0.8:  # Very extreme sentiment
            score_adjustments.append((-10, "Contains extremely emotional language"))
        
        if len(combined_text) < 20:
            score_adjustments.append((-5, "Very short content"))
        elif len(combined_text) > 500:
            score_adjustments.append((5, "Detailed explanation"))
        
        for category, markers in self.CREDIBILITY_MARKERS.items():
            matching_markers = [marker for marker in markers if marker in combined_text]
            if matching_markers:
                score_adjustments.append((10, f"Uses credible language: {category}"))
                break  
        for category, detractors in self.CREDIBILITY_DETRACTORS.items():
            matching_detractors = [detractor for detractor in detractors if detractor in combined_text]
            if matching_detractors:
                score_adjustments.append((-15, f"Uses questionable language: {category}"))
                break  # Count only once per category
      
        if score > 100:
            score_adjustments.append((10, "Highly upvoted by community"))
        elif score < 0:
            score_adjustments.append((-5, "Downvoted by community"))
        
        urls = re.findall(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', combined_text)
        if urls:
            url_domains = [urlparse(url).netloc for url in urls]
            trusted_url_count = sum(1 for domain in url_domains if any(trusted in domain for trusted in self.trusted_domains))
            untrusted_url_count = sum(1 for domain in url_domains if any(untrusted in domain for untrusted in self.untrusted_domains))
            
            if trusted_url_count > 0:
                score_adjustments.append((trusted_url_count * 5, f"Contains {trusted_url_count} link(s) to reputable sources"))
            if untrusted_url_count > 0:
                score_adjustments.append((untrusted_url_count * -10, f"Contains {untrusted_url_count} link(s) to questionable sources"))
    
        random_factor = np.random.randint(-3, 4)
        score_adjustments.append((random_factor, None)) 
        for adjustment, factor in score_adjustments:
            base_score += adjustment
            if factor:  # Don't add None factors (like randomization)
                factors.append(factor)
        
        final_score = max(0, min(100, base_score))
        
        return final_score, factors
    
    def batch_analyze_posts(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Batch analyzing {len(df)} posts for credibility")
        result_df = df.copy()
        
        result_df['credibility_score'] = 0
        result_df['credibility_factors'] = ''
        
        for idx, row in result_df.iterrows():
            title = row.get('title', '')
            text = row.get('selftext', '')
            score = row.get('score', 0)
            author = row.get('author', '')
            
            cred_score, cred_factors = self.analyze_post(title, text, score, author)
            
            result_df.at[idx, 'credibility_score'] = cred_score
            result_df.at[idx, 'credibility_factors'] = ", ".join(cred_factors) if cred_factors else "No specific factors detected"
        
        score_stats = result_df['credibility_score'].describe()
        logger.info(f"Credibility score distribution: min={score_stats['min']:.1f}, max={score_stats['max']:.1f}, mean={score_stats['mean']:.1f}")
        
        return result_df
