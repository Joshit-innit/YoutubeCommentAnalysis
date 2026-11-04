"""Runs the comment pipeline and returns everything the dashboard displays."""

import pandas as pd

from components.clustering import cluster_comments
from components.fetch import fetch_comments, fetch_video_metadata
from components.insights import generate_insights
from components.metrics import calculate_metrics
from components.preprocessing import clean_text
from components.sentiment import get_sentiment
from components.spam_detection import detect_spam
from components.toxicity import predict_toxicity


def run_analysis(video_id, api_key, max_comments):
    """Return the analysis for a video, or None when it has no comments."""
    video_meta = fetch_video_metadata(video_id, api_key)
    df = fetch_comments(video_id, api_key).head(max_comments)

    if df.empty:
        return None

    df["clean_comment"] = df["comment"].apply(clean_text)
    df["toxicity_prediction"] = predict_toxicity(df["clean_comment"])
    df["sentiment"] = df["comment"].apply(get_sentiment)

    clusters, cluster_keywords = cluster_comments(
        df["clean_comment"],
        n_clusters=min(4, len(df)),
    )
    df["cluster"] = clusters

    df["comment_length"] = df["comment"].astype(str).apply(len)
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
    df["emotion_intensity"] = (
        df["comment_length"] * 0.35
        + df["likes"] * 1.8
        + df["toxicity_prediction"] * 28
    )

    # KMeans cluster ids carry no meaning, so topics are named by their own
    # top keywords and numbered from largest to smallest.
    topics = []
    for rank, cluster_id in enumerate(df["cluster"].value_counts().index, start=1):
        keywords = cluster_keywords.get(cluster_id, [])
        topics.append(
            {
                "id": cluster_id,
                "name": f"Topic {rank}: {', '.join(keywords[:3])}",
                "keywords": keywords,
            }
        )
    topic_names = {topic["id"]: topic["name"] for topic in topics}
    df["cluster_name"] = df["cluster"].map(topic_names)

    metrics = calculate_metrics(df)
    spam_comments, spam_percentage = detect_spam(df)

    author_counts = df["author"].value_counts().rename_axis("author").reset_index(name="comments")
    repeat_authors = author_counts[author_counts["comments"] > 2].head(8)

    insights = generate_insights(
        metrics["positive_percentage"],
        metrics["negative_percentage"],
        metrics["toxicity_percentage"],
        spam_percentage,
        metrics["satisfaction_score"],
    )

    return {
        "video": video_meta,
        "comments": df,
        "topics": topics,
        "metrics": metrics,
        "spam_comments": spam_comments,
        "spam_percentage": spam_percentage,
        "repeat_authors": repeat_authors,
        "insights": insights,
    }
