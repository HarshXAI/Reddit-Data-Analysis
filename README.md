# Reddit Data Analysis Dashboard

A comprehensive Streamlit dashboard for analyzing Reddit data with advanced analytics, AI-powered insights, and interactive visualizations.

## 🚀 Features

- **📊 Interactive Data Analysis**
  - Real-time data processing and visualization
  - Time series analysis with customizable aggregation
  - Advanced topic modeling with BERTopic
  - Network analysis of author-subreddit relationships
  - Credibility scoring and misinformation detection

- **🤖 AI-Powered Insights**
  - Google Gemini API integration for intelligent summaries
  - Automated trend detection and analysis
  - Topic modeling with advanced NLP
  - Credibility assessment with machine learning

- **📈 Visualization Components**
  - Interactive time series plots
  - Dynamic network graphs
  - Word clouds and topic visualization
  - Credibility score distribution

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/HarshXAI/reddit-analysis-dashboard.git
cd reddit-analysis-dashboard
```

2. Create and activate a virtual environment:
```bash
python -m venv simppl
source simppl/bin/activate  # On Windows: simppl\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your Google Gemini API key
```

## 🚀 Usage

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Upload your Reddit data in JSONL format or use the demo data
3. Navigate through different analysis tabs
4. Generate AI-powered insights using the Gemini API

## 📁 Project Structure

```
/SimPPL project/
├── app.py              # Main Streamlit application
├── config.py           # Configuration settings
├── data/              # Data directory for demo files
├── modules/           # Core functionality modules
│   ├── __init__.py
│   ├── data_ingestion.py
│   ├── stats_analysis.py
│   ├── topic_modeling.py
│   ├── advanced_analysis.py
│   ├── ai_summary.py
│   └── summary_agent.py
├── requirements.txt    # Project dependencies
└── .env               # Environment variables (create from .env.example)
```

## 🔧 Configuration

- Set `GEMINI_API_KEY` in `.env` for AI features
- Adjust analysis parameters in `config.py`
- Customize visualization settings in the UI

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Google Gemini API for AI capabilities
- Streamlit for the web interface
- BERTopic for advanced topic modeling
- NetworkX for graph analysis

## 🐛 Troubleshooting

### Common Issues

1. **Gemini API Connection**
   - Verify your API key in `.env`
   - Check for model availability
   - Run diagnostics from the AI Insights tab

2. **Data Loading**
   - Ensure JSONL format is correct
   - Check file permissions
   - Use demo data to verify functionality

3. **Topic Modeling**
   - BERTopic requires sufficient RAM
   - Falls back to simpler LDA if needed
   - Adjust topic numbers for better results

### Performance Tips

- Use appropriate time aggregation for large datasets
- Enable caching for repeated analyses
- Adjust chunk size for large file processing
