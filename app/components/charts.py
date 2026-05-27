import plotly.express as px

from wordcloud import WordCloud

import matplotlib.pyplot as plt

# =========================================
# SENTIMENT PIE CHART
# =========================================

def sentiment_chart(df):

    sentiment_counts = (
        df["sentiment"]
        .value_counts()
        .reset_index()
    )

    sentiment_counts.columns = [
        "Sentiment",
        "Count"
    ]

    fig = px.pie(
        sentiment_counts,
        names="Sentiment",
        values="Count",
        title="Audience Sentiment"
    )

    return fig

# =========================================
# TOXICITY BAR CHART
# =========================================

def toxicity_chart(df):

    toxicity_counts = (
        df["toxicity_prediction"]
        .value_counts()
        .reset_index()
    )

    toxicity_counts.columns = [
        "Type",
        "Count"
    ]

    toxicity_counts["Type"] = [
        "Safe",
        "Toxic"
    ]

    fig = px.bar(
        toxicity_counts,
        x="Type",
        y="Count",
        title="Toxic vs Safe Comments"
    )

    return fig

# =========================================
# COMMENT LENGTH HISTOGRAM
# =========================================

def comment_length_chart(df):

    fig = px.histogram(
        df,
        x="comment_length",
        nbins=30,
        title="Comment Length Distribution"
    )

    return fig

# =========================================
# LIKES DISTRIBUTION
# =========================================

def likes_chart(df):

    fig = px.histogram(
        df,
        x="likes",
        nbins=30,
        title="Likes Distribution"
    )

    return fig

# =========================================
# WORD CLOUD
# =========================================

def generate_wordcloud(df):

    all_words = " ".join(
        df["comment"]
        .astype(str)
    )

    wordcloud = WordCloud(
        width=1200,
        height=600,
        background_color="white"
    ).generate(all_words)

    fig, ax = plt.subplots(
        figsize=(12,6)
    )

    ax.imshow(wordcloud)

    ax.axis("off")

    return fig