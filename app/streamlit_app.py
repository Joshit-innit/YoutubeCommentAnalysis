# =========================================================
# app/streamlit_app.py
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv

# =========================================================
# COMPONENT IMPORTS
# =========================================================

from components.fetch import (
    extract_video_id,
    fetch_comments
)

from components.preprocessing import (
    clean_text
)

from components.sentiment import (
    get_sentiment
)

from components.toxicity import (
    predict_toxicity
)

from components.spam_detection import (
    detect_spam
)

from components.metrics import (
    calculate_metrics
)

from components.charts import (
    sentiment_chart,
    toxicity_chart,
    comment_length_chart,
    likes_chart,
    generate_wordcloud
)

from components.insights import (
    generate_insights
)

from components.clustering import (
    cluster_comments
)

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="YouTube Audience Intelligence",
    page_icon="🧠",
    layout="wide"
)

# =========================================================
# LOAD ENV VARIABLES
# =========================================================

load_dotenv()

API_KEY = os.getenv("YOUTUBE_API_KEY")

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.block-container {
    padding-top: 2rem;
}

h1, h2, h3 {
    color: white;
}

[data-testid="metric-container"] {
    background-color: #1E1E1E;
    border: 1px solid #333333;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 20px;
}

.stTabs [data-baseweb="tab"] {
    background-color: #1E1E1E;
    border-radius: 10px;
    padding: 10px 20px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title(
    "🧠 YouTube AI Analytics"
)

st.sidebar.markdown("""
Analyze:
- Audience Sentiment
- Toxicity
- Spam Activity
- Viewer Psychology
- Engagement
- Audience Satisfaction
""")

video_url = st.sidebar.text_input(
    "📺 Paste YouTube URL"
)

max_comments = st.sidebar.slider(
    "Maximum Comments",
    min_value=100,
    max_value=1000,
    value=500,
    step=100
)

show_toxic_only = st.sidebar.checkbox(
    "Show Toxic Comments Only"
)

analyze_button = st.sidebar.button(
    "🚀 Analyze Video"
)

# =========================================================
# TITLE
# =========================================================

st.title(
    "🧠 AI-Powered YouTube Audience Intelligence"
)

st.markdown("""
Discover:
- what viewers feel
- why they react
- audience behavior patterns
- spam activity
- toxic discussions
- engagement insights
""")

# =========================================================
# MAIN ANALYSIS
# =========================================================

if analyze_button:

    video_id = extract_video_id(
        video_url
    )

    if not video_id:

        st.error(
            "❌ Invalid YouTube URL"
        )

    else:

        with st.spinner(
            "Analyzing audience reactions..."
        ):

            # =========================================
            # FETCH COMMENTS
            # =========================================

            df = fetch_comments(
                video_id,
                API_KEY
            )

            # Limit comments
            df = df.head(max_comments)

            # =========================================
            # PREPROCESSING
            # =========================================

            df["clean_comment"] = (
                df["comment"]
                .apply(clean_text)
            )

            # =========================================
            # TOXICITY PREDICTION
            # =========================================

            df["toxicity_prediction"] = (
                predict_toxicity(
                    df["clean_comment"]
                )
            )

            # =========================================
            # SENTIMENT ANALYSIS
            # =========================================

            df["sentiment"] = (
                df["comment"]
                .apply(get_sentiment)
            )

            # =====================================
            # COMMENT CLUSTERING
            # =====================================

            clusters, cluster_keywords = (
                cluster_comments(
                    df["clean_comment"]
                )
            )

            df["cluster"] = clusters

            # =========================================
            # COMMENT LENGTH
            # =========================================

            df["comment_length"] = (
                df["comment"]
                .astype(str)
                .apply(len)
            )

            # =========================================
            # METRICS
            # =========================================

            metrics = calculate_metrics(df)

            total_comments = metrics[
                "total_comments"
            ]

            toxicity_percentage = metrics[
                "toxicity_percentage"
            ]

            positive_percentage = metrics[
                "positive_percentage"
            ]

            negative_percentage = metrics[
                "negative_percentage"
            ]

            satisfaction_score = metrics[
                "satisfaction_score"
            ]

            # =========================================
            # SPAM DETECTION
            # =========================================

            spam_comments, spam_percentage = (
                detect_spam(df)
            )

            # =========================================
            # CONTROVERSY SCORE
            # =========================================

            controversy_score = min(
                positive_percentage,
                negative_percentage
            )

            # =========================================
            # KPI CARDS
            # =========================================

            st.markdown(
                "## 📊 Dashboard Overview"
            )

            col1, col2, col3, col4, col5, col6 = st.columns(6)

            with col1:
                st.metric(
                    "Comments",
                    total_comments
                )

            with col2:
                st.metric(
                    "Positive %",
                    f"{positive_percentage:.1f}%"
                )

            with col3:
                st.metric(
                    "Negative %",
                    f"{negative_percentage:.1f}%"
                )

            with col4:
                st.metric(
                    "Toxicity %",
                    f"{toxicity_percentage:.1f}%"
                )

            with col5:
                st.metric(
                    "Spam %",
                    f"{spam_percentage:.1f}%"
                )

            with col6:
                st.metric(
                    "Controversy",
                    f"{controversy_score:.1f}"
                )

            # =========================================
            # FILTER COMMENTS
            # =========================================

            if show_toxic_only:

                filtered_df = df[
                    df["toxicity_prediction"] == 1
                ]

            else:

                filtered_df = df

            # =========================================
            # TABS
            # =========================================

            # =========================================================
            # 4️⃣ UPDATE TABS
            # =========================================================

            tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
                "📈 Sentiment",
                "☠️ Toxicity",
                "🚨 Spam",
                "🔥 Top Comments",
                "📊 Engagement",
                "📋 Raw Data",
                "🧠 Clusters"
            ])

            # =========================================
            # TAB 1 - SENTIMENT
            # =========================================

            with tab1:

                st.markdown(
                    "## 😊 Audience Mood"
                )

                fig1 = sentiment_chart(df)

                st.plotly_chart(
                    fig1,
                    use_container_width=True
                )

                st.markdown(
                    "## ☁️ Most Discussed Topics"
                )

                wordcloud_fig = generate_wordcloud(df)

                st.pyplot(wordcloud_fig)

            # =========================================
            # TAB 2 - TOXICITY
            # =========================================

            with tab2:

                st.markdown(
                    "## ☠️ Toxicity Analysis"
                )

                fig2 = toxicity_chart(df)

                st.plotly_chart(
                    fig2,
                    use_container_width=True
                )

                st.markdown(
                    "## 🚨 Most Toxic Comments"
                )

                toxic_df = df[
                    df["toxicity_prediction"] == 1
                ]

                st.dataframe(
                    toxic_df[
                        [
                            "author",
                            "comment",
                            "likes"
                        ]
                    ].head(20),
                    use_container_width=True
                )

            # =========================================
            # TAB 3 - SPAM
            # =========================================

            with tab3:

                st.markdown(
                    "## 🚨 Spam Detection"
                )

                if len(spam_comments) > 0:

                    st.warning(
                        "⚠️ Potential spam activity detected"
                    )

                    spam_df = spam_comments.reset_index()

                    spam_df.columns = [
                        "Comment",
                        "Occurrences"
                    ]

                    st.dataframe(
                        spam_df.head(20),
                        use_container_width=True
                    )

                else:

                    st.success(
                        "✅ No major spam detected"
                    )

            # =========================================
            # TAB 4 - TOP COMMENTS
            # =========================================

            with tab4:

                st.markdown(
                    "## 🔥 Top 20 Most Liked Comments"
                )

                top_comments = df.sort_values(
                    by="likes",
                    ascending=False
                )

                st.dataframe(
                    top_comments[
                        [
                            "author",
                            "comment",
                            "likes",
                            "sentiment"
                        ]
                    ].head(20),
                    use_container_width=True
                )

                # =====================================
                # COMMON WORDS
                # =====================================

                st.markdown(
                    "## 📌 Most Common Keywords"
                )

                from collections import Counter

                all_words = " ".join(
                    df["clean_comment"]
                ).split()

                common_words = Counter(
                    all_words
                )

                common_df = pd.DataFrame(
                    common_words.most_common(20),
                    columns=[
                        "Keyword",
                        "Frequency"
                    ]
                )

                st.dataframe(
                    common_df,
                    use_container_width=True
                )

            # =========================================
            # TAB 5 - ENGAGEMENT
            # =========================================

            with tab5:

                st.markdown(
                    "## 📊 Engagement Analysis"
                )

                # Comment length graph
                fig3 = comment_length_chart(df)

                st.plotly_chart(
                    fig3,
                    use_container_width=True
                )

                # Likes graph
                fig4 = likes_chart(df)

                st.plotly_chart(
                    fig4,
                    use_container_width=True
                )

                # Longest comments
                st.markdown(
                    "## 📝 Longest Comments"
                )

                longest_comments = df.sort_values(
                    by="comment_length",
                    ascending=False
                )

                st.dataframe(
                    longest_comments[
                        [
                            "author",
                            "comment",
                            "comment_length"
                        ]
                    ].head(10),
                    use_container_width=True
                )

            # =========================================
            # TAB 6 - RAW DATA
            # =========================================

            with tab6:

                st.markdown(
                    "## 📋 Full Comment Dataset"
                )

                st.dataframe(
                    filtered_df,
                    use_container_width=True
                )

                # Download button
                csv = filtered_df.to_csv(
                    index=False
                )

                st.download_button(
                    label="📥 Download Analysis Report",
                    data=csv,
                    file_name="youtube_analysis.csv",
                    mime="text/csv"
                )
            # =========================================
            # TAB 7 - COMMENT CLUSTERS
            # =========================================

            with tab7:

                st.markdown(
                    "## 🧠 Audience Discussion Clusters"
                )

                st.write(
                    """
                    Similar comments are grouped together
                    to understand what the audience is
                    discussing most.
                    """
                )

                for cluster_id, keywords in cluster_keywords.items():

                    st.subheader(
                        f"Cluster {cluster_id + 1}"
                    )

                    st.write(
                        f"Main Topics: {', '.join(keywords)}"
                    )

                    cluster_comments_df = df[
                        df["cluster"] == cluster_id
                    ]

                    st.dataframe(
                        cluster_comments_df[
                            [
                                "author",
                                "comment",
                                "likes",
                                "sentiment"
                            ]
                        ].head(10),
                        use_container_width=True
                    )

            # =========================================
            # AI INSIGHTS
            # =========================================

            st.markdown(
                "## 🧠 AI Audience Insights"
            )

            insights = generate_insights(
                positive_percentage,
                negative_percentage,
                toxicity_percentage,
                spam_percentage,
                satisfaction_score
            )

            for insight in insights:

                st.info(insight)

            # =========================================
            # ADVANCED INSIGHTS
            # =========================================

            st.markdown(
                "## 🔍 Behavioral Analysis"
            )

            if controversy_score > 35:

                st.warning(
                    "Audience appears highly divided. "
                    "The video may be controversial."
                )

            if toxicity_percentage > 25:

                st.error(
                    "High toxic engagement detected "
                    "in audience discussions."
                )

            if positive_percentage > 70:

                st.success(
                    "Audience reception is extremely positive."
                )

            if spam_percentage > 5:

                st.warning(
                    "Possible coordinated spam behavior detected."
                )

            avg_comment_length = (
                df["comment_length"]
                .mean()
            )

            st.write(
                f"Average Comment Length: "
                f"{avg_comment_length:.2f}"
            )

            st.write(
                f"Average Likes per Comment: "
                f"{df['likes'].mean():.2f}"
            )