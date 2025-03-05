# SimPPL - Social Insights & Media Post Pattern Library

## 📝 Description
SimPPL is a powerful Streamlit-based analytics dashboard for analyzing Reddit posts and extracting meaningful insights. This tool helps you understand posting patterns, content credibility, and topic trends across different subreddits.

## 🔗 Project Resources
- [View Demo Video](https://drive.google.com/file/d/1pl_s2kArg87zchSNf3Aex0XVxOK7GxQ0/view?usp=sharing)
- [Live Demo](https://harshxai-reddit-data-analysis-app-1cseoy.streamlit.app)

## ✨ Features
- 📊 Interactive data visualization
- ⏱️ Time series analysis of posting patterns
- 🔍 Advanced topic modeling
- 🛡️ Content credibility scoring
- 🤖 AI-powered insights using Google's Gemini
- 📱 Responsive design for all devices

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.9+
pip (Python package manager)
Virtual environment (recommended)
```

### Installation
1. Clone the repository:
```bash
git clone https://github.com/HarshXAI/AI-powered-Phishing-Detection-System.git
cd SimPPL
```

2. Create and activate virtual environment:
```bash
python -m venv simppl
source simppl/bin/activate  # On Windows: simppl\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env
# Edit .env file with your API keys
```

### 🎯 Usage

1. Start the application locally:
```bash
streamlit run app.py
```

2. Access the application:
   - Local: `http://localhost:8501`
   - Deployed version: [https://harshxai-reddit-data-analysis-app-1cseoy.streamlit.app](https://harshxai-reddit-data-analysis-app-1cseoy.streamlit.app)

3. Upload your JSONL data file or use the demo data option


## 🛠️ Configuration
Key configuration options in `config.py`:
- `MAX_TOPICS`: Maximum number of topics for analysis
- `DEFAULT_TOPICS`: Default number of topics to display
- `APP_TITLE`: Application title
- `APP_DESCRIPTION`: Application description

## 📈 Features in Detail

### Time Series Analysis
- Daily/Weekly/Monthly post frequency
- Peak activity detection
- Trend analysis
- Day-of-week patterns

### Text Analysis
- Word clouds
- Keyword frequency
- Topic modeling
- Content search

### Credibility Analysis
- Content credibility scoring
- Source verification
- Engagement patterns
- Misinformation detection

### AI Insights
- Pattern recognition
- Trend prediction
- Content summarization
- Anomaly detection

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙋‍♂️ Support
For support:
- Create an issue in the repository
- Contact: YOUR_EMAIL_HERE
- Project Link: YOUR_PROJECT_LINK_HERE

## 🌟 Acknowledgments
- [Streamlit](https://streamlit.io/)
- [Google Gemini](https://deepmind.google/technologies/gemini/)
- Other libraries and contributors

---
Made with ❤️ by Harsh Kanani
