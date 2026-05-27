import pandas as pd

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load comments
df = pd.read_csv("data/processed/predicted_comments.csv")

# Initialize analyzer
analyzer = SentimentIntensityAnalyzer()

# Sentiment function
def get_sentiment(text):

    score = analyzer.polarity_scores(str(text))

    compound = score["compound"]

    if compound >= 0.05:
        return "Positive"

    elif compound <= -0.05:
        return "Negative"

    else:
        return "Neutral"

# Apply sentiment analysis
df["sentiment"] = df["comment"].apply(get_sentiment)

# Save results
df.to_csv(
    "data/processed/final_comments.csv",
    index=False
)

# Sentiment counts
sentiment_counts = df["sentiment"].value_counts()

print("\n===== SENTIMENT ANALYSIS =====\n")

print(sentiment_counts)

# Calculate percentages
total = len(df)

positive = (
    sentiment_counts.get("Positive", 0)
    / total
) * 100

negative = (
    sentiment_counts.get("Negative", 0)
    / total
) * 100

neutral = (
    sentiment_counts.get("Neutral", 0)
    / total
) * 100

print(f"\nPositive: {positive:.2f}%")
print(f"Negative: {negative:.2f}%")
print(f"Neutral: {neutral:.2f}%")