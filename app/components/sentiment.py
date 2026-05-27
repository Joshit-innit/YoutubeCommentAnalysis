from vaderSentiment.vaderSentiment import (
    SentimentIntensityAnalyzer
)

analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text):

    score = analyzer.polarity_scores(
        str(text)
    )

    compound = score["compound"]

    if compound >= 0.05:
        return "Positive"

    elif compound <= -0.05:
        return "Negative"

    else:
        return "Neutral"