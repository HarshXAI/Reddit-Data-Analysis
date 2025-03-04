# Reddit Data Analysis Dashboard

This project provides an interactive dashboard for analyzing Reddit data, including time series analysis, topic modeling, AI-generated summaries, and credibility assessment.

![Dashboard Screenshot](dashboard_screenshot.png)

## Features

- **Data Ingestion**: Load and process Reddit JSONL data
- **Statistical Analysis**: Post distribution, scoring patterns, temporal analysis
- **Text Analysis**: Word clouds, keyword frequency, search functionality
- **Topic Modeling**: Discover main discussion topics using LDA or BERTopic
- **AI Summaries**: Get natural language explanations of data insights using Google's Gemini API
- **Credibility Analysis**: Assess posts for potential misinformation
- **Network Analysis**: Visualize relationships between authors and subreddits

## Project Structure

```
SimPPL project/
├── app.py                  # Original Streamlit application
├── app_enhanced.py         # Enhanced UI version with advanced features
├── modules/
│   ├── data_ingestion.py   # Data loading and preprocessing
│   ├── stats_analysis.py   # Statistical analysis functions
│   ├── topic_modeling.py   # Basic topic modeling implementation
│   ├── summary_agent.py    # Basic summary generation
│   ├── advanced_analysis.py # Advanced ML analysis (topics, trends, credibility)
│   └── ai_summary.py       # AI-powered summaries with Gemini API
├── requirements.txt        # Project dependencies
└── README.md               # Documentation
```

## Setup and Installation

1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Set up your Google Gemini API key:
   - Get a free API key from [Google AI Studio](https://ai.google.dev/)
   - Create a `.env` file in the project root with: `GEMINI_API_KEY=your_key_here`

4. Run the Streamlit app:
```bash
# Run the original app
streamlit run app.py

# Or run the enhanced version with advanced features
streamlit run app_enhanced.py
```

## Data Format

The application expects Reddit data in JSONL format (JSON Lines), with each line containing a separate JSON record for a Reddit post:

```
{"kind": "t3", "data": {"subreddit": "Subreddit name", "title": "Post title", "selftext": "Post content", "author": "Username", "score": 123, "created_utc": 1739858460.0, ...}}
{"kind": "t3", "data": {"subreddit": "Another subreddit", "title": "Another post", "selftext": "More content", "author": "Another user", "score": 456, "created_utc": 1739858470.0, ...}}
```

Each JSON object should have a `kind` field with value "t3" and a `data` field containing the post data.

## Key Features Comparison

| Feature                     | app.py (Original) | app_enhanced.py (Enhanced) |
|----------------------------|-------------------|----------------------------|
| Data Loading               | ✅               | ✅                         |
| Basic Statistics           | ✅               | ✅                         |
| Time Series Analysis       | ✅               | ✅                         |
| Word Clouds                | ✅               | ✅                         |
| Basic Topic Modeling       | ✅               | ✅                         |
| Advanced Topic Modeling    | ❌               | ✅ (BERTopic)              |
| Trend Detection            | ❌               | ✅                         |
| Credibility Analysis       | ❌               | ✅                         |
| Network Graph Analysis     | ❌               | ✅                         |
| AI-Generated Summaries     | Basic            | ✅ (Gemini API)            |
| Enhanced UI/UX             | ❌               | ✅                         |
| Light/Dark Mode            | ❌               | ✅                         |

## Deployment

### Deploying to Streamlit Cloud

1. Push this code to a GitHub repository
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Create a new app and point it to your repository
4. Select either `app.py` or `app_enhanced.py` as the main file
5. Add your Gemini API key as a secret with name `GEMINI_API_KEY`
6. Deploy!

### Deploying to Hugging Face Spaces

1. Create a new Space on [Hugging Face Spaces](https://huggingface.co/spaces)
2. Select "Streamlit" as the SDK
3. Upload this project files
4. Set the environment variable `GEMINI_API_KEY`
5. The app will automatically deploy

## Usage

1. Open the app in your browser
2. Upload your Reddit JSONL data using the file uploader in the sidebar
3. Explore different analyses using the tabs or navigation menu
4. Use filters to narrow down analysis to specific subreddits or time periods
5. View AI-generated summaries for key insights

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
