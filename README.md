# 🔍 Brand Reputation Monitor

![Version](https://img.shields.io/badge/Version-1.0-blue)
![License](https://img.shields.io/badge/License-MIT-green)

An advanced AI-powered platform for monitoring and analyzing brand reputation across multiple online sources in real-time.

## ✨ Features

- **Multi-Source Data Collection** - Aggregate brand mentions from Reddit, News APIs, and RSS feeds
- **AI-Powered Analysis** - Advanced sentiment analysis and topic modeling
- **Interactive Dashboard** - Real-time visualization of brand reputation metrics
- **Comprehensive Insights** - Track sentiment trends, topics, and sources over time
- **Scalable Architecture** - Modular design for easy extension to new data sources

## 🧠 AI & NLP Capabilities

The heart of this platform leverages cutting-edge NLP models and techniques:

- **Sentiment Analysis** using **RoBERTa** from Hugging Face Transformers
  - Detecting positive, negative, and neutral sentiments with confidence scores
  - Specialized for social media and news content

- **Topic Modeling** with **BERTopic**
  - Automatically discover topics being discussed about your brand
  - Powered by Sentence Transformers embeddings (`all-MiniLM-L6-v2`)
  - Dimensionality reduction and clustering to identify meaningful topics

- **Transformer Models** for text understanding
  - Integration with Hugging Face's ecosystem of pre-trained language models
  - Contextual understanding of brand mentions for deeper insights

## 🔧 Technical Stack

- **Backend**: FastAPI for high-performance API endpoints
- **Dashboard**: Dash with Plotly for interactive visualizations
- **Data Collection**: Reddit API (PRAW), NewsAPI, RSS feed parsing
- **Data Storage**: SQLite database for tracking mentions and analytics
- **NLP**: Transformer models, BERTopic, Sentence-Transformers
- **Machine Learning**: UMAP, HDBSCAN for clustering and topic discovery

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- API keys for NewsAPI and Reddit (optional)

### Installation

1. Clone this repository
```bash
git clone https://github.com/yourusername/brand-reputation-monitor.git
cd brand-reputation-monitor
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your API keys (see sample below)
```
NEWS_API_KEY=your_news_api_key
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
REDDIT_USER_AGENT=python:brand-monitor:v1.0 (by /u/yourusername)
REDDIT_USERNAME=your_reddit_username
REDDIT_PASSWORD=your_reddit_password
```

### Running the Application

1. Start the FastAPI backend
```bash
uvicorn backend.main:app --reload
```

2. Launch the Dashboard
```bash
python -m dashboard.app
```

3. Access the dashboard at http://127.0.0.1:8050

## 📊 Screenshots

*Coming soon*

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [BERTopic](https://github.com/MaartenGr/BERTopic)
- [Sentence-Transformers](https://github.com/UKPLab/sentence-transformers)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Dash](https://dash.plotly.com/)
