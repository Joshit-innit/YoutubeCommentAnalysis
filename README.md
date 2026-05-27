# AI-Powered YouTube Audience Intelligence Platform

Real-time NLP analytics dashboard for analyzing YouTube comments, audience sentiment, toxicity, spam patterns, engagement behavior, and discussion trends.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python" alt="Python 3.11">
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-red?style=for-the-badge&logo=streamlit" alt="Streamlit">
  <img src="https://img.shields.io/badge/Machine%20Learning-Scikit--learn-green?style=for-the-badge" alt="Machine Learning">
  <img src="https://img.shields.io/badge/NLP-Text%20Analytics-orange?style=for-the-badge" alt="NLP">
  <img src="https://img.shields.io/badge/Status-Active-success?style=for-the-badge" alt="Status">
</p>

## Overview

This project is an AI-powered YouTube analytics platform that fetches comments from a YouTube video and transforms them into an interactive audience intelligence dashboard. It combines natural language processing, machine learning, and data visualization to help understand how viewers react, what topics they discuss, and whether the comment section contains toxic or spam-like behavior.

The application follows a production-style machine learning pattern:

```text
Offline model training + online real-time inference
```

The toxicity model is trained separately, saved as a serialized model, and loaded by the Streamlit application during analysis. This avoids retraining during user sessions and keeps dashboard inference fast.

## Key Features

- Fetch comments from any supported YouTube video URL using the YouTube Data API v3
- Analyze audience sentiment as positive, negative, or neutral
- Detect toxic comments using a trained machine learning model
- Identify repeated or suspicious spam-like comments
- Calculate audience satisfaction, controversy, toxicity, and spam metrics
- Visualize sentiment, toxicity, comment length, likes, keywords, and word clouds
- Display the most liked comments and longest comments
- Export the analyzed comment dataset as a CSV report
- Generate automated audience insights from the analysis results

## Dashboard Analytics

| Area | Description |
| --- | --- |
| Sentiment analysis | Measures positive, negative, and neutral audience reactions |
| Toxicity detection | Flags comments predicted as harmful or toxic |
| Spam detection | Identifies repeated comments and possible coordinated activity |
| Engagement analytics | Analyzes likes, comment lengths, and high-performing comments |
| Topic discovery | Surfaces common keywords and repeated discussion themes |
| Audience satisfaction | Estimates overall viewer satisfaction from sentiment balance |
| Controversy score | Measures how divided the audience appears to be |
| Data export | Allows downloading analyzed comments as a CSV file |

## System Architecture

```text
YouTube video URL
        |
        v
Extract video ID
        |
        v
Fetch comments from YouTube Data API
        |
        v
Clean and preprocess comment text
        |
        +------------------------+
        |                        |
        v                        v
Toxicity prediction       Sentiment analysis
        |                        |
        +-----------+------------+
                    |
                    v
        Metrics, charts, spam detection
                    |
                    v
        Streamlit audience intelligence dashboard
```

## Machine Learning Pipeline

### Training Phase

The toxicity classifier is trained offline using a labeled toxic comment dataset.

```text
Raw toxic comment dataset
        |
        v
Text cleaning and preprocessing
        |
        v
TF-IDF vectorization
        |
        v
Logistic Regression training
        |
        v
Save toxicity_model.pkl and tfidf_vectorizer.pkl
```

### Inference Phase

When a user analyzes a YouTube video, the saved model and vectorizer are loaded to classify new comments in real time.

```text
YouTube comments
        |
        v
Preprocess comment text
        |
        v
Load saved TF-IDF vectorizer
        |
        v
Load saved toxicity model
        |
        v
Predict toxic or safe
        |
        v
Generate dashboard metrics and insights
```

## Project Structure

```text
YoutubeToxicAnalysis/
├── app/
│   ├── streamlit_app.py
│   ├── styles/
│   │   └── custom.css
│   └── components/
│       ├── charts.py
│       ├── fetch.py
│       ├── insights.py
│       ├── metrics.py
│       ├── preprocessing.py
│       ├── sentiment.py
│       ├── spam_detection.py
│       ├── topic_analysis.py
│       └── toxicity.py
├── data/
│   ├── raw/
│   │   ├── comments.csv
│   │   └── train.csv
│   └── processed/
│       ├── cleaned_comments.csv
│       ├── final_comments.csv
│       └── predicted_comments.csv
├── models/
│   ├── tfidf_vectorizer.pkl
│   └── toxicity_model.pkl
├── notebooks/
│   ├── eda.ipynb
│   └── visualization.ipynb
├── src/
│   ├── download_dataset.py
│   ├── fetch_comments.py
│   ├── predict.py
│   ├── preprocess.py
│   ├── sentiment_analysis.py
│   ├── train_model.py
│   └── utils.py
├── main.py
├── requirements.txt
└── README.md
```

## Technology Stack

| Category | Tools |
| --- | --- |
| Application | Streamlit |
| Data processing | Pandas, NumPy |
| Machine learning | Scikit-learn, Logistic Regression, TF-IDF |
| NLP | NLTK, VADER Sentiment Analyzer, regex preprocessing |
| Visualization | Plotly, Matplotlib, WordCloud |
| API integration | YouTube Data API v3 |
| Configuration | python-dotenv |
| Model persistence | Joblib |

## Installation

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd YoutubeToxicAnalysis
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Virtual Environment

macOS/Linux:

```bash
source venv/bin/activate
```

Windows:

```bash
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

If Plotly is not already installed in your environment, install it with:

```bash
pip install plotly
```

## YouTube API Setup

1. Open the [Google Cloud Console](https://console.cloud.google.com/).
2. Create or select a Google Cloud project.
3. Enable **YouTube Data API v3**.
4. Create an API key.
5. Copy `.env.example` to `.env` and add your API key.

```env
YOUTUBE_API_KEY=your_youtube_api_key_here
```

## Running the Application

```bash
python -m streamlit run app/streamlit_app.py
```

Then open the local Streamlit URL shown in the terminal.

## How to Use

1. Paste a YouTube video URL into the sidebar.
2. Select the maximum number of comments to analyze.
3. Optionally enable the toxic-comments-only filter.
4. Click **Analyze Video**.
5. Review the generated dashboard tabs:
   - Sentiment
   - Toxicity
   - Spam
   - Top Comments
   - Engagement
   - Raw Data
6. Download the CSV report if needed.

## Model Training

The project includes a training script for the toxicity classifier:

```bash
python src/train_model.py
```

The script:

- Loads the labeled training dataset from `data/raw/train.csv`
- Cleans and preprocesses comment text
- Converts text into TF-IDF features
- Trains a Logistic Regression classifier
- Saves the trained model to `models/toxicity_model.pkl`
- Saves the vectorizer to `models/tfidf_vectorizer.pkl`

## Outputs

The dashboard can produce:

- Sentiment distribution chart
- Toxic vs safe comment chart
- Word cloud of frequent terms
- Comment length distribution
- Likes distribution
- Top liked comments table
- Potential spam comments table
- Full analyzed dataset
- Downloadable CSV report
- Automated audience insight messages

## Learning Outcomes

This project demonstrates practical implementation of:

- End-to-end NLP application development
- Offline model training and online inference
- Text preprocessing and feature extraction
- TF-IDF vectorization
- Toxicity classification with Scikit-learn
- VADER-based sentiment analysis
- YouTube API integration
- Streamlit dashboard development
- Data visualization and CSV reporting
- Modular Python project organization

## Future Enhancements

- Add transformer-based toxicity classification using BERT or similar models
- Add emotion detection beyond positive, negative, and neutral sentiment
- Add topic clustering for deeper audience segmentation
- Add sarcasm and irony detection
- Add bot-like behavior and coordinated activity detection
- Add historical comparison between multiple videos
- Add creator-facing recommendation summaries
- Improve deployment support for Streamlit Cloud, Render, or Railway

## Author

**Joshit Tammana**

Machine Learning, NLP, AI Engineering, Data Analytics, and Intelligent Systems enthusiast.
