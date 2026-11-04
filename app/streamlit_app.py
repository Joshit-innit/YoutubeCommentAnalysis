import inspect
import os
from collections import Counter

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from googleapiclient.errors import HttpError

from components.analysis import run_analysis
from components.charts import (
    comment_length_chart,
    likes_chart,
    ranked_bar_chart,
    sentiment_timeline_chart,
    toxicity_heatmap,
)
from components.fetch import extract_video_id
from components.ui import (
    compact_number,
    empty_state,
    kpi_row,
    load_css,
    page_title,
    panel,
    score_rows,
    sentiment_split,
    status_rows,
    video_header,
)


st.set_page_config(
    page_title="YouTube Comment Analytics",
    page_icon=":material/monitoring:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Newer Streamlit replaces use_container_width with width="stretch".
STRETCH = (
    {"width": "stretch"}
    if "width" in inspect.signature(st.plotly_chart).parameters
    else {"use_container_width": True}
)

PAGES = {
    "Overview": ":material/dashboard:",
    "Sentiment": ":material/sentiment_satisfied:",
    "Toxicity": ":material/report:",
    "Spam": ":material/block:",
    "Engagement": ":material/thumb_up:",
    "Topics": ":material/category:",
    "Comments": ":material/forum:",
    "Data": ":material/table:",
}

COLUMN_LABELS = {
    "author": "Author",
    "comment": "Comment",
    "likes": "Likes",
    "published_at": "Published",
    "clean_comment": "Cleaned text",
    "toxicity_prediction": "Flagged",
    "sentiment": "Sentiment",
    "cluster": "Topic ID",
    "cluster_name": "Topic",
    "comment_length": "Length",
    "emotion_intensity": "Intensity",
    "comments": "Comments",
    "times_posted": "Times posted",
    "influence_score": "Influence",
}


def get_api_key():
    load_dotenv()

    try:
        api_key = st.secrets.get("YOUTUBE_API_KEY")
    except FileNotFoundError:
        api_key = None

    return api_key or os.getenv("YOUTUBE_API_KEY")


def show_table(df, columns=None, height=None):
    view = df[columns] if columns else df
    config = {
        column: st.column_config.Column(
            COLUMN_LABELS.get(column, column),
            width="large" if column == "comment" else None,
        )
        for column in view.columns
    }

    if height is None:
        height = min(420, 38 + 35 * len(view))

    st.dataframe(
        view,
        height=height,
        hide_index=True,
        column_config=config,
        **STRETCH,
    )


def show_chart(figure, key):
    st.plotly_chart(
        figure,
        key=key,
        theme=None,
        config={"displayModeBar": False},
        **STRETCH,
    )


def keyword_counts(df, limit):
    words = " ".join(df["clean_comment"]).split()
    return Counter(words).most_common(limit)


def signed(value):
    return f"{value:+.1f}"


def render_overview(result):
    df = result["comments"]
    metrics = result["metrics"]
    total = metrics["total_comments"]
    flagged = int(df["toxicity_prediction"].sum())
    spam_count = len(result["spam_comments"])
    positive = metrics["positive_percentage"]
    negative = metrics["negative_percentage"]

    kpi_row(
        [
            ("Comments analyzed", f"{total:,}", f"{df['likes'].mean():.1f} likes per comment"),
            ("Positive", f"{positive:.1f}%", f"{(df['sentiment'] == 'Positive').sum():,} comments"),
            ("Negative", f"{negative:.1f}%", f"{(df['sentiment'] == 'Negative').sum():,} comments"),
            ("Flagged toxic", f"{metrics['toxicity_percentage']:.1f}%", f"{flagged:,} comments"),
            ("Spam rate", f"{result['spam_percentage']:.1f}%", f"{spam_count:,} repeated messages"),
        ]
    )

    trend, split = st.columns([8, 4])
    with trend:
        with panel("Comment sentiment over time"):
            show_chart(sentiment_timeline_chart(df), "overview_timeline")
    with split:
        with panel("Sentiment split"):
            sentiment_split(df["sentiment"].value_counts().to_dict(), total)

    keywords, heatmap = st.columns(2)
    with keywords:
        with panel("Top keywords", "Most frequent words after removing stopwords"):
            words = keyword_counts(df, 10)
            if words:
                labels, counts = zip(*words)
                show_chart(ranked_bar_chart(labels, counts, "Mentions", 320), "overview_keywords")
            else:
                st.caption("No keywords found.")
    with heatmap:
        with panel("Flagged comments by day and hour"):
            show_chart(toxicity_heatmap(df, 320), "overview_heatmap")

    signals, scores = st.columns(2)
    with signals:
        with panel("Signals"):
            status_rows(
                [
                    (
                        "Audience",
                        "Positive" if positive >= negative else "Mixed",
                        "good" if positive >= negative else "warning",
                    ),
                    (
                        "Moderation",
                        "Elevated" if metrics["toxicity_percentage"] > 25 else "Controlled",
                        "critical" if metrics["toxicity_percentage"] > 25 else "good",
                    ),
                    (
                        "Spam",
                        "Detected" if spam_count else "Clean",
                        "warning" if spam_count else "good",
                    ),
                    (
                        "Polarization",
                        "Divided" if metrics["controversy_score"] > 35 else "Not divided",
                        "warning" if metrics["controversy_score"] > 35 else "neutral",
                    ),
                ],
                result["insights"],
            )
    with scores:
        with panel(
            "Composite scores",
            "Rule-based estimates from sentiment, toxicity and likes, not model predictions",
        ):
            net = metrics["satisfaction_score"]
            mood = metrics["viewer_mood_index"]
            score_rows(
                [
                    ("Net sentiment (-100 to 100)", net, (net + 100) / 2),
                    ("Viewer mood (-100 to 100)", mood, (mood + 100) / 2),
                    ("Engagement", metrics["engagement_score"], metrics["engagement_score"]),
                    ("Controversy", metrics["controversy_score"], metrics["controversy_score"]),
                    ("Audience loyalty", metrics["audience_loyalty_score"], metrics["audience_loyalty_score"]),
                    ("Viral potential", metrics["viral_potential_score"], metrics["viral_potential_score"]),
                ]
            )


def render_sentiment(result):
    df = result["comments"]
    metrics = result["metrics"]
    counts = df["sentiment"].value_counts().to_dict()
    total = metrics["total_comments"]

    kpi_row(
        [
            ("Positive", f"{metrics['positive_percentage']:.1f}%", f"{counts.get('Positive', 0):,} comments"),
            ("Neutral", f"{counts.get('Neutral', 0) / total * 100:.1f}%", f"{counts.get('Neutral', 0):,} comments"),
            ("Negative", f"{metrics['negative_percentage']:.1f}%", f"{counts.get('Negative', 0):,} comments"),
            ("Net sentiment", signed(metrics["satisfaction_score"]), "Positive share minus negative share"),
        ]
    )

    split, trend = st.columns([4, 8])
    with split:
        with panel("Sentiment split"):
            sentiment_split(counts, total)
    with trend:
        with panel("Comment sentiment over time"):
            show_chart(sentiment_timeline_chart(df), "sentiment_timeline")

    for column, sentiment in zip(st.columns(2), ["Positive", "Negative"]):
        with column:
            with panel(f"Most-liked {sentiment.lower()} comments"):
                matching = df[df["sentiment"] == sentiment].nlargest(8, "likes")
                if matching.empty:
                    st.caption(f"No {sentiment.lower()} comments in this sample.")
                else:
                    show_table(matching, ["comment", "likes"])


def render_toxicity(result):
    df = result["comments"]
    flagged = df[df["toxicity_prediction"] == 1]

    kpi_row(
        [
            ("Flagged comments", f"{len(flagged):,}", f"of {len(df):,} analyzed"),
            ("Toxic rate", f"{result['metrics']['toxicity_percentage']:.1f}%", "Share of the sample"),
            ("Likes on flagged", f"{int(flagged['likes'].sum()):,}", "Total across flagged comments"),
            ("Flagged authors", f"{flagged['author'].nunique():,}", "Distinct accounts"),
        ]
    )

    with panel("Flagged comments by day and hour"):
        show_chart(toxicity_heatmap(df, 320), "toxicity_heatmap")

    with panel("Flagged comments", "Most-liked first"):
        if flagged.empty:
            st.caption("No comments in this sample were flagged as toxic.")
        else:
            show_table(flagged.nlargest(50, "likes"), ["author", "comment", "likes"])


def render_spam(result):
    spam_comments = result["spam_comments"]
    repeat_authors = result["repeat_authors"]

    kpi_row(
        [
            ("Spam rate", f"{result['spam_percentage']:.1f}%", "Repeated messages per comment"),
            ("Repeated messages", f"{len(spam_comments):,}", "Posted more than 3 times"),
            ("Repeat authors", f"{len(repeat_authors):,}", "More than 2 comments"),
        ]
    )

    messages, authors = st.columns([7, 5])
    with messages:
        with panel("Repeated messages"):
            if spam_comments.empty:
                st.caption("No message was posted more than 3 times.")
            else:
                spam_df = spam_comments.rename_axis("comment").reset_index(name="times_posted")
                show_table(spam_df.head(25))
    with authors:
        with panel("Repeat authors"):
            if repeat_authors.empty:
                st.caption("No author posted more than 2 comments.")
            else:
                show_table(repeat_authors)


def render_engagement(result):
    df = result["comments"]

    kpi_row(
        [
            ("Average likes", f"{df['likes'].mean():.1f}", "Per comment"),
            ("Median likes", f"{df['likes'].median():.0f}", "Per comment"),
            ("Most-liked comment", compact_number(int(df["likes"].max())), "Likes"),
            ("Average length", f"{df['comment_length'].mean():.0f}", "Characters"),
        ]
    )

    likes, length = st.columns(2)
    with likes:
        with panel("Likes per comment"):
            show_chart(likes_chart(df), "engagement_likes")
    with length:
        with panel("Comment length"):
            show_chart(comment_length_chart(df), "engagement_length")

    with panel("Longest comments"):
        show_table(df.nlargest(10, "comment_length"), ["author", "comment", "comment_length"])


def render_topics(result):
    df = result["comments"]
    topics = result["topics"]

    distribution, detail = st.columns([5, 7])
    with distribution:
        with panel("Comments per topic", "Grouped by KMeans on comment text and named by top keywords"):
            counts = df["cluster_name"].value_counts()
            show_chart(
                ranked_bar_chart(counts.index, counts.values, height=max(220, 56 * len(counts))),
                "topic_distribution",
            )
    with detail:
        with panel("Comments in topic"):
            selected = st.selectbox(
                "Topic",
                topics,
                format_func=lambda topic: topic["name"],
                label_visibility="collapsed",
            )
            st.caption("Keywords: " + ", ".join(selected["keywords"]))
            matching = df[df["cluster"] == selected["id"]].nlargest(25, "likes")
            show_table(matching, ["comment", "likes", "sentiment"], height=320)


def render_comments(result):
    df = result["comments"]

    with panel("Most-liked comments"):
        show_table(df.nlargest(20, "likes"), ["author", "comment", "likes", "sentiment"])

    influencers, keywords = st.columns([7, 5])
    with influencers:
        with panel("Most influential authors", "Influence = likes × 1.6 + comment length × 0.25"):
            ranked = (
                df.assign(influence_score=(df["likes"] * 1.6 + df["comment_length"] * 0.25).round(1))
                .sort_values("influence_score", ascending=False)
                .drop_duplicates(subset=["author"])
                .head(10)
            )
            show_table(ranked, ["author", "likes", "influence_score"])
    with keywords:
        with panel("Keyword frequency"):
            words = pd.DataFrame(keyword_counts(df, 20), columns=["Keyword", "Mentions"])
            show_table(words)

    with panel("Highest-intensity comments", "Intensity combines length, likes and the toxicity flag"):
        intense = df.nlargest(12, "emotion_intensity").round({"emotion_intensity": 1})
        show_table(intense, ["author", "comment", "emotion_intensity", "sentiment"])


def render_data(result):
    df = result["comments"]

    filter_flagged, show_all, _ = st.columns([2, 2, 5])
    with filter_flagged:
        flagged_only = st.toggle("Flagged comments only")
    with show_all:
        all_columns = st.toggle("Show all columns")

    view = df[df["toxicity_prediction"] == 1] if flagged_only else df

    if not all_columns:
        view = view[["author", "comment", "likes", "sentiment", "toxicity_prediction", "cluster"]]

    with panel("Analyzed comments", f"{len(view):,} rows"):
        show_table(view, height=520)
        st.download_button(
            "Download CSV",
            data=view.to_csv(index=False),
            file_name="youtube_analysis.csv",
            mime="text/csv",
        )


RENDERERS = {
    "Overview": render_overview,
    "Sentiment": render_sentiment,
    "Toxicity": render_toxicity,
    "Spam": render_spam,
    "Engagement": render_engagement,
    "Topics": render_topics,
    "Comments": render_comments,
    "Data": render_data,
}


def analyze(video_url, max_comments):
    """Run an analysis and keep it in the session, so controls don't clear it."""
    api_key = get_api_key()
    video_id = extract_video_id(video_url)

    if not api_key:
        st.error("YOUTUBE_API_KEY is missing. Add it to .streamlit/secrets.toml or .env.")
        return

    if not video_id:
        st.error("That doesn't look like a YouTube video URL. Paste the full link and try again.")
        return

    try:
        with st.spinner("Fetching and analyzing comments..."):
            result = run_analysis(video_id, api_key, max_comments)
    except HttpError as error:
        reason = "Comments are turned off for this video." if error.status_code == 403 else str(error.reason)
        st.error(f"YouTube returned an error. {reason}")
        return

    if result is None:
        st.error("This video has no comments to analyze.")
        return

    st.session_state["analysis"] = result


load_css()

with st.sidebar:
    st.html('<div class="brand">YouTube Comment Analytics</div>')

    with st.form("analyze_form", border=False):
        video_url = st.text_input("Video URL", placeholder="youtube.com/watch?v=...")
        max_comments = st.slider(
            "Comments to analyze",
            min_value=100,
            max_value=1000,
            value=500,
            step=100,
        )
        submitted = st.form_submit_button("Analyze", type="primary", **STRETCH)

if submitted:
    analyze(video_url, max_comments)

analysis = st.session_state.get("analysis")

if analysis is None:
    empty_state()
else:
    with st.sidebar:
        st.divider()
        selected_page = st.radio(
            "Page",
            list(PAGES),
            format_func=lambda name: f"{PAGES[name]} {name}",
            key="page",
            label_visibility="collapsed",
        )

    if analysis["video"]:
        video_header(analysis["video"], analysis["metrics"]["total_comments"])

    page_title(selected_page)
    RENDERERS[selected_page](analysis)
