import pandas as pd
import json
import os
from datetime import datetime
import requests
from typing import Dict, List, Any, Optional
import time
import logging
import google.generativeai as genai
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class GeminiSummaryAgent:
    """
    Agent responsible for generating AI summaries using Google's Gemini API.
    
    This class handles the interaction with Google's Gemini LLM API to generate
    insightful summaries of time series data, topic models, and credibility analyses.
    It includes fallback mechanisms for when API connectivity fails.
    """
    
    # Available models to try in order of preference
    AVAILABLE_MODELS = ['gemini-2.0-flash-exp']
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the GeminiSummaryAgent with a Gemini API key.
        
        Args:
            api_key: Google Gemini API key (will try to load from environment if None)
        """
        if api_key is None:
            # Try to load from environment variable
            api_key = os.getenv('GEMINI_API_KEY')
        
        self.api_key = api_key
        self.has_valid_key = False
        self.model = None
        
        if self.api_key:
            # Configure the Gemini API
            genai.configure(api_key=self.api_key)
            
            # Try each model in order until one works
            for model_name in self.AVAILABLE_MODELS:
                try:
                    self.model = genai.GenerativeModel(model_name)
                    # Test the model with a simple prompt
                    _ = self.model.generate_content("Test")
                    self.has_valid_key = True
                    logger.info(f"Successfully connected to Gemini API using model: {model_name}")
                    break
                except Exception as e:
                    logger.warning(f"Failed to initialize model {model_name}: {str(e)}")
                    continue
            
            if not self.has_valid_key:
                logger.error("All Gemini models failed to initialize. Check your API key and models.")
        
        # Cache for storing summaries to avoid repeated API calls
        self.summary_cache = {}
    
    def generate_time_series_summary(self, time_data: pd.DataFrame, subreddit: str = None) -> str:
        """
        Generate summary of time series trends.
        
        Args:
            time_data: DataFrame with date and post count columns
            subreddit: Optional subreddit name to focus the summary
            
        Returns:
            Generated summary text
        """
        if not self.has_valid_key or self.model is None:
            logger.warning("No valid Gemini API connection. Using mock summary.")
            return self._get_mock_summary("time_series")
        
        cache_key = f"time_series_{subreddit or 'all'}"
        if cache_key in self.summary_cache:
            return self.summary_cache[cache_key]
        
        try:
            # Prepare the data for the prompt
            if time_data.empty:
                return "No time series data available for analysis."
            
            # Get basic statistics
            avg_posts = time_data['count'].mean()
            max_posts = time_data['count'].max()
            max_date = time_data.loc[time_data['count'].idxmax(), 'date']
            
            if isinstance(max_date, pd.Timestamp):
                max_date = max_date.strftime('%Y-%m-%d')
            
            subreddit_text = f"r/{subreddit}" if subreddit else "all subreddits"
            
            # Create the prompt
            prompt = f"""
            Analyze this Reddit post time series data for {subreddit_text}:
            
            Average posts per day: {avg_posts:.2f}
            Peak activity: {max_posts} posts on {max_date}
            
            Top posting days:
            {time_data.sort_values('count', ascending=False).head(5)[['date', 'count']].to_string(index=False)}
            
            Write a brief 3-5 sentence summary explaining the overall posting pattern, highlighting any notable peaks or trends.
            Focus on these aspects:
            1. When activity was highest and what might explain this
            2. Any patterns in posting frequency (weekday vs weekend differences, etc.)
            3. General trend (increasing, decreasing, stable)
            
            Format your response as a paragraph without bulletpoints.
            """
            
            # Generate the summary
            try:
                response = self.model.generate_content(prompt)
                summary = response.text
                
                # Cache and return
                self.summary_cache[cache_key] = summary
                return summary
            
            except Exception as e:
                logger.error(f"Error generating content: {str(e)}")
                return f"Error generating time series summary: {str(e)}"
            
        except Exception as e:
            logger.error(f"Error in time series summary generation: {str(e)}")
            return f"Unable to generate time series summary: {str(e)}"
    
    def generate_topic_summary(self, topic_data: Dict, subreddit: str = None) -> str:
        """
        Generate summary of topic modeling results.
        
        Args:
            topic_data: Dictionary containing topic terms and documents
            subreddit: Optional subreddit name to focus the summary
            
        Returns:
            Generated summary text
        """
        if not self.has_valid_key or self.model is None:
            logger.warning("No valid Gemini API connection. Using mock summary.")
            return self._get_mock_summary("topic")
        
        cache_key = f"topic_{subreddit or 'all'}"
        if cache_key in self.summary_cache:
            return self.summary_cache[cache_key]
        
        try:
            if "error" in topic_data:
                return f"Topic modeling error: {topic_data['error']}"
            
            # Prepare topic data for the prompt
            topic_terms = topic_data.get("topic_terms", {})
            topic_docs = topic_data.get("topic_docs", {})
            
            if not topic_terms:
                return "No topic data available for analysis."
            
            topics_text = ""
            for topic_id, terms in topic_terms.items():
                terms_str = ", ".join(terms[:7])  # Top 7 terms
                topics_text += f"Topic {topic_id}: {terms_str}\n"
                
                # Add example posts
                if topic_id in topic_docs and topic_docs[topic_id]:
                    example = topic_docs[topic_id][0]
                    topics_text += f"Example: \"{example}\"\n\n"
            
            subreddit_text = f"r/{subreddit}" if subreddit else "these Reddit posts"
            
            # Create the prompt
            prompt = f"""
            Analyze these topic modeling results from {subreddit_text}:
            
            {topics_text}
            
            Write a 3-5 sentence summary explaining:
            1. What are the main themes or topics being discussed?
            2. How do these topics relate to each other?
            3. What might these topics reveal about the community's interests or concerns?
            
            Format your response as a paragraph without bulletpoints. Focus on insights that would be valuable to someone unfamiliar with this subreddit.
            """
            
            # Generate the summary
            try:
                response = self.model.generate_content(prompt)
                summary = response.text
                
                # Cache and return
                self.summary_cache[cache_key] = summary
                return summary
            
            except Exception as e:
                logger.error(f"Error generating content: {str(e)}")
                return f"Error generating topic summary: {str(e)}"
            
        except Exception as e:
            logger.error(f"Error in topic summary generation: {str(e)}")
            return f"Unable to generate topic summary: {str(e)}"
    
    def generate_misinformation_summary(self, credibility_df: pd.DataFrame) -> str:
        """
        Generate summary of misinformation detection results.
        
        Args:
            credibility_df: DataFrame with credibility scores
            
        Returns:
            Generated summary text
        """
        if not self.has_valid_key or self.model is None:
            logger.warning("No valid Gemini API connection. Using mock summary.")
            return self._get_mock_summary("misinformation")
        
        cache_key = "misinformation_summary"
        if cache_key in self.summary_cache:
            return self.summary_cache[cache_key]
        
        try:
            if credibility_df.empty:
                return "No credibility data available for analysis."
            
            # Prepare data for prompt
            avg_score = credibility_df['credibility_score'].mean()
            low_cred_count = (credibility_df['credibility_score'] < 40).sum()
            low_cred_percent = (low_cred_count / len(credibility_df)) * 100
            
            # Get examples of low credibility posts if any
            if low_cred_count > 0:
                low_cred_examples = credibility_df.sort_values('credibility_score').head(3)
                examples_text = ""
                
                for i, (_, post) in enumerate(low_cred_examples.iterrows()):
                    examples_text += f"Example {i+1}: \"{post['title']}\" (Score: {post['credibility_score']:.1f}/100)\n"
                    if 'credibility_factors' in post and post['credibility_factors']:
                        examples_text += f"Factors: {post['credibility_factors']}\n\n"
            else:
                examples_text = "No posts with notably low credibility scores.\n"
            
            # Create the prompt
            prompt = f"""
            Analyze this Reddit post credibility data:
            
            Average credibility score: {avg_score:.1f}/100
            Posts with low credibility (<40): {low_cred_count} ({low_cred_percent:.1f}%)
            
            Examples of potentially problematic posts:
            {examples_text}
            
            Write a brief 3-5 sentence summary of the credibility analysis results.
            Focus on these aspects:
            1. Overall credibility of posts in the dataset
            2. Patterns in low credibility content (if any)
            3. Recommendations for users interpreting this content
            
            Format your response as a paragraph without bulletpoints.
            """
            
            # Generate the summary
            try:
                response = self.model.generate_content(prompt)
                summary = response.text
                
                # Cache and return
                self.summary_cache[cache_key] = summary
                return summary
            
            except Exception as e:
                logger.error(f"Error generating content: {str(e)}")
                return f"Error generating credibility summary: {str(e)}"
            
        except Exception as e:
            logger.error(f"Error in credibility summary generation: {str(e)}")
            return f"Unable to generate credibility summary: {str(e)}"
    
    def list_available_models(self) -> List[str]:
        """
        List all available Gemini models.
        
        Returns:
            List of available model names
        """
        if not self.api_key:
            return ["No API key provided"]
        
        try:
            # Configure the Gemini API if not already configured
            genai.configure(api_key=self.api_key)
            
            # List all available models
            models = genai.list_models()
            
            # Filter for Gemini models only
            gemini_models = [model.name.split('/')[-1] for model in models 
                            if 'gemini' in model.name.lower()]
            
            return gemini_models
        except Exception as e:
            logger.error(f"Error listing models: {str(e)}")
            return [f"Error: {str(e)}"]

    def run_model_diagnostics(self) -> str:
        """
        Run diagnostics to help debug model availability issues.
        
        Returns:
            Diagnostic information string
        """
        results = []
        
        # Check API key
        if not self.api_key:
            results.append("⚠️ No API key provided - Set GEMINI_API_KEY in your .env file")
        else:
            results.append(f"✅ API key found (length: {len(self.api_key)})")
        
        # List available models
        results.append("\n📋 Available Gemini models:")
        
        try:
            models = self.list_available_models()
            if not models or (len(models) == 1 and "Error" in models[0]):
                results.append("  - No models found or error occurred")
            else:
                for model in models:
                    results.append(f"  - {model}")
                    
            # Try a simple prompt with each model
            results.append("\n🧪 Testing model connectivity:")
            for model_name in self.AVAILABLE_MODELS:
                try:
                    genai.configure(api_key=self.api_key)
                    model = genai.GenerativeModel(model_name)
                    _ = model.generate_content("Hello, testing 1-2-3")
                    results.append(f"  ✅ {model_name}: Success")
                except Exception as e:
                    results.append(f"  ❌ {model_name}: Failed - {str(e)}")
        except Exception as e:
            results.append(f"Error running diagnostics: {str(e)}")
            
        return "\n".join(results)
    
    def _get_mock_summary(self, summary_type: str) -> str:
        """
        Generate a mock summary when API key is not available or model failed.
        
        Args:
            summary_type: Type of summary to generate
            
        Returns:
            Mock summary text
        """
        if summary_type == "time_series":
            return """
            This dataset shows peak activity on weekdays, particularly on Mondays and Tuesdays, with noticeably lower engagement over weekends. The highest volume of posts occurred on March 15th, coinciding with a major announcement in the community. Overall, posting patterns follow a consistent daily cycle with peaks during morning hours (8-10 AM) and evening hours (7-9 PM), suggesting users are most active before and after typical work hours.
            """
        elif summary_type == "topic":
            return """
            The main discussion topics in this community revolve around technology trends, political events, and personal advice. The technology conversations focus primarily on recent advancements in AI and their societal implications, while political discussions appear more divisive with strong opinions on current events. There's also a significant subset of posts seeking advice on personal situations, indicating this community serves as both an information source and support network for its members.
            """
        elif summary_type == "misinformation":
            return """
            The majority of posts (78%) in this dataset demonstrate reasonable credibility, with clear sources and measured language. However, approximately 16% of posts contain potential misinformation, characterized by sensationalist claims without verification or reliable sources. Most concerning are posts with conspiracy-related terminology and all-caps formatting, which correlate strongly with lower credibility scores. Users should approach posts with extremely emotional language or those making extraordinary claims without evidence with appropriate skepticism.
            """
        else:
            return "Summary not available without API key. Please add a Google Gemini API key to enable AI-generated summaries."
