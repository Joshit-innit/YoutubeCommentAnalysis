import os
from collections import Counter

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from components.charts import (
    cluster_distribution_chart,
    comment_length_chart,
    generate_wordcloud,
    likes_chart,
    sentiment_chart,
    timeline_sentiment_chart,
    toxicity_chart,
    toxicity_heatmap,
)
from components.clustering import cluster_comments
from components.fetch import extract_video_id, fetch_comments, fetch_video_metadata
from components.insights import generate_insights
from components.metrics import calculate_metrics
from components.preprocessing import clean_text
from components.sentiment import get_sentiment
from components.spam_detection import detect_spam
from components.toxicity import predict_toxicity


st.set_page_config(
    page_title="YouTube Audience Intelligence",
    page_icon="AI",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_KEY = st.secrets["YOUTUBE_API_KEY"]


def load_css():
    css_path = os.path.join("app", "styles", "custom.css")

    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as css_file:
            st.markdown(
                f"<style>{css_file.read()}</style>",
                unsafe_allow_html=True,
            )


def section_header(kicker, title, copy=None):
    copy_markup = f"<p>{copy}</p>" if copy else ""
    st.markdown(
        f"""
        <div class="section-header fade-up">
            <span>{kicker}</span>
            <h2>{title}</h2>
            {copy_markup}
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(label, value, accent, icon, trend):
    st.markdown(
        f"""
        <div class="kpi-card fade-up">
            <div class="kpi-topline">
                <span class="kpi-icon">{icon}</span>
                <span class="kpi-trend">{trend}</span>
            </div>
            <div class="kpi-value" style="--accent:{accent};">{value}</div>
            <div class="kpi-label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def insight_card(message, index):
    tones = ["cyan", "violet", "blue", "amber"]
    tone = tones[index % len(tones)]

    st.markdown(
        f"""
        <div class="insight-card {tone} fade-up">
            <div class="insight-dot"></div>
            <div>
                <span>AI insight {index + 1}</span>
                <p>{message}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def status_card(title, value, detail, tone="cyan"):
    st.markdown(
        f"""
        <div class="status-card {tone} fade-up">
            <span>{title}</span>
            <strong>{value}</strong>
            <p>{detail}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def intelligence_panel(title, value, subtitle, tone="blue"):
    st.markdown(
        f"""
        <div class="intel-panel {tone} fade-up">
            <span>{title}</span>
            <h3>{value}</h3>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def user_card(name, score, detail):
    st.markdown(
        f"""
        <div class="user-card fade-up">
            <div>
                <strong>{name}</strong>
                <p>{detail}</p>
            </div>
            <span>{score}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_table(df, columns, height=420):
    st.dataframe(
        df[columns],
        use_container_width=True,
        height=height,
        hide_index=True,
    )


load_css()

st.sidebar.markdown(
    """
    <div class="sidebar-brand">
        <div class="brand-mark">AI</div>
        <div>
            <strong>YouTube Intelligence</strong>
            <span>Audience analytics console</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown('<div class="sidebar-panel">', unsafe_allow_html=True)
max_comments = st.sidebar.slider(
    "Comment sample size",
    min_value=100,
    max_value=1000,
    value=500,
    step=100,
)

show_toxic_only = st.sidebar.toggle(
    "Focus toxic comments",
    value=False,
)

show_raw_columns = st.sidebar.toggle(
    "Expanded raw data",
    value=False,
)
st.sidebar.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="ambient-layer"></div>', unsafe_allow_html=True)

st.markdown(
    """
    <section class="hero-shell fade-up">
        <div class="particle-field"></div>
        <div class="hero-badge">
            <span></span>
            Real-time AI audience intelligence
        </div>
        <h1>YouTube Audience Intelligence Platform</h1>
        <p>
            Decode audience sentiment, toxicity, spam behavior, engagement
            signals, and discussion clusters from YouTube comments.
        </p>
    </section>
    """,
    unsafe_allow_html=True,
)

input_left, input_center, input_right = st.columns([0.8, 2.4, 0.8])

with input_center:
    with st.container():
        st.markdown('<div class="command-center fade-up">', unsafe_allow_html=True)
        video_url = st.text_input(
            "YouTube video URL",
            placeholder="Paste a YouTube video URL...",
            label_visibility="collapsed",
        )
        analyze_button = st.button(
            "Analyze Video",
            use_container_width=True,
            type="primary",
        )
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    """
    <div class="signal-strip fade-up">
        <div><span>NLP</span><strong>Semantic parsing</strong></div>
        <div><span>ML</span><strong>Toxicity model</strong></div>
        <div><span>AI</span><strong>Audience insights</strong></div>
        <div><span>DATA</span><strong>Interactive analytics</strong></div>
    </div>
    """,
    unsafe_allow_html=True,
)


if analyze_button:
    video_id = extract_video_id(video_url)

    if not API_KEY:
        st.error("YOUTUBE_API_KEY is missing. Add it to your local .env file.")
    elif not video_id:
        st.error("Invalid YouTube URL. Please paste a complete YouTube video link.")
    else:
        with st.spinner("Activating audience intelligence engine..."):
            video_meta = fetch_video_metadata(video_id, API_KEY)
            df = fetch_comments(video_id, API_KEY).head(max_comments)

            if df.empty:
                st.error("No comments were returned for this video. Comments may be disabled or unavailable.")
                st.stop()

            df["clean_comment"] = df["comment"].apply(clean_text)
            df["toxicity_prediction"] = predict_toxicity(df["clean_comment"])
            df["sentiment"] = df["comment"].apply(get_sentiment)

            cluster_count = min(4, max(1, len(df)))
            clusters, cluster_keywords = cluster_comments(
                df["clean_comment"],
                n_clusters=cluster_count,
            )
            df["cluster"] = clusters

            df["comment_length"] = df["comment"].astype(str).apply(len)
            df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")

            metrics = calculate_metrics(df)
            total_comments = metrics["total_comments"]
            toxicity_percentage = metrics["toxicity_percentage"]
            positive_percentage = metrics["positive_percentage"]
            negative_percentage = metrics["negative_percentage"]
            satisfaction_score = metrics["satisfaction_score"]
            engagement_score = metrics["engagement_score"]
            viewer_mood_index = metrics["viewer_mood_index"]
            viral_potential_score = metrics["viral_potential_score"]
            audience_loyalty_score = metrics["audience_loyalty_score"]

            spam_comments, spam_percentage = detect_spam(df)
            controversy_score = metrics["controversy_score"]
            neutral_percentage = max(
                0,
                100 - positive_percentage - negative_percentage,
            )
            avg_comment_length = df["comment_length"].mean()
            avg_likes = df["likes"].mean()
            repeated_users = (
                df["author"]
                .value_counts()
                .reset_index()
                .rename(columns={"index": "author", "author": "comments"})
            )
            repeated_users.columns = ["author", "comments"]
            suspicious_users = repeated_users[repeated_users["comments"] > 2].head(8)

            df["emotion_intensity"] = (
                df["comment_length"] * 0.35
                + df["likes"] * 1.8
                + df["toxicity_prediction"] * 28
            )

            if show_toxic_only:
                filtered_df = df[df["toxicity_prediction"] == 1]
            else:
                filtered_df = df

            cluster_theme = {
                0: ("🎬 Editing Praise", "cyan"),
                1: ("💸 Monetization Complaints", "violet"),
                2: ("😂 Meme Discussions", "blue"),
                3: ("🚨 Spam Promotions", "amber"),
            }
            df["cluster_name"] = df["cluster"].apply(
                lambda val: cluster_theme.get(val, (f"Cluster {val + 1}", "blue"))[0]
            )

        if video_meta:
            st.markdown('<div class="video-meta-shell fade-up">', unsafe_allow_html=True)
            meta_left, meta_right = st.columns([1, 2.2])
            with meta_left:
                if video_meta.get("thumbnail"):
                    st.image(video_meta["thumbnail"], use_container_width=True)
            with meta_right:
                st.markdown(
                    f"""
                    <div class="video-meta-content">
                        <span>Video Intelligence Header</span>
                        <h3>{video_meta.get("title", "Video")}</h3>
                        <p>{video_meta.get("channel", "Unknown channel")}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                stats = st.columns(4)
                with stats[0]:
                    intelligence_panel("Views", f"{video_meta.get('views', 0):,}", "Lifetime views", "cyan")
                with stats[1]:
                    intelligence_panel("Likes", f"{video_meta.get('likes', 0):,}", "Video likes", "blue")
                with stats[2]:
                    subscriber_count = video_meta.get("subscribers")
                    intelligence_panel(
                        "Subscribers",
                        f"{subscriber_count:,}" if subscriber_count else "N/A",
                        "Channel audience",
                        "violet",
                    )
                with stats[3]:
                    published = str(video_meta.get("published_at", "")).replace("T", " ").replace("Z", "")
                    intelligence_panel("Uploaded", published[:16] if published else "Unknown", "Publish time", "amber")
            st.markdown("</div>", unsafe_allow_html=True)

        section_header(
            "Audience Overview",
            "Real-time intelligence dashboard",
            "A consolidated view of audience mood, risk, engagement, and discussion behavior.",
        )

        kpi_cols = st.columns(8)
        with kpi_cols[0]:
            kpi_card("Total Comments", f"{total_comments:,}", "#22d3ee", "C", "live sample")
        with kpi_cols[1]:
            kpi_card("Positive", f"{positive_percentage:.1f}%", "#34d399", "+", "sentiment")
        with kpi_cols[2]:
            kpi_card("Toxicity", f"{toxicity_percentage:.1f}%", "#fb7185", "!", "risk")
        with kpi_cols[3]:
            kpi_card("Spam", f"{spam_percentage:.1f}%", "#fbbf24", "S", "pattern")
        with kpi_cols[4]:
            kpi_card("Satisfaction", f"{satisfaction_score:.1f}", "#a78bfa", "A", "score")
        with kpi_cols[5]:
            kpi_card("Controversy", f"{controversy_score:.1f}", "#38bdf8", "X", "divide")
        with kpi_cols[6]:
            kpi_card("Engagement", f"{engagement_score:.1f}", "#60a5fa", "E", "momentum")
        with kpi_cols[7]:
            kpi_card("Viewer Mood", f"{viewer_mood_index:.1f}", "#ff0033", "M", "emotion")

        overview_left, overview_mid, overview_right = st.columns([1.15, 1, 1])
        with overview_left:
            status_card(
                "Audience temperature",
                "Positive" if positive_percentage >= negative_percentage else "Mixed",
                f"{positive_percentage:.1f}% positive, {negative_percentage:.1f}% negative, {neutral_percentage:.1f}% neutral.",
                "cyan",
            )
        with overview_mid:
            status_card(
                "Moderation risk",
                "Elevated" if toxicity_percentage > 25 else "Controlled",
                f"{toxicity_percentage:.1f}% of analyzed comments were flagged by the model.",
                "violet",
            )
        with overview_right:
            status_card(
                "Engagement pulse",
                f"{avg_likes:.1f} likes/comment",
                f"Average comment length is {avg_comment_length:.1f} characters.",
                "blue",
            )

        section_header(
            "Audience Intelligence Engine",
            "Behavioral and emotional signals in one command layer",
            "Understand why viewers praise, complain, polarize, and re-engage with precision.",
        )
        intel_cols = st.columns(4)
        with intel_cols[0]:
            intelligence_panel(
                "Viral Potential Score",
                f"{viral_potential_score:.1f}",
                "Composite momentum and discussion energy",
                "cyan",
            )
        with intel_cols[1]:
            intelligence_panel(
                "Audience Loyalty Score",
                f"{audience_loyalty_score:.1f}",
                "Likelihood of sustained positive retention",
                "violet",
            )
        with intel_cols[2]:
            intelligence_panel(
                "Controversial Topic Risk",
                f"{controversy_score:.1f}",
                "Polarization signal from sentiment split",
                "amber",
            )
        with intel_cols[3]:
            praise_rate = max(0, positive_percentage - toxicity_percentage * 0.5)
            intelligence_panel(
                "Praise Pattern Index",
                f"{praise_rate:.1f}",
                "Positive reinforcement intensity",
                "blue",
            )

        (
            overview_tab,
            sentiment_tab,
            toxicity_tab,
            spam_tab,
            engagement_tab,
            clusters_tab,
            insights_tab,
            top_comments_tab,
            psychology_tab,
            raw_data_tab,
        ) = st.tabs(
            [
                "Overview",
                "Sentiment",
                "Toxicity",
                "Spam",
                "Engagement",
                "Clusters",
                "AI Insights",
                "Top Comments",
                "Audience Psychology",
                "Raw Data",
            ]
        )

        with overview_tab:
            left, right = st.columns([1.05, 0.95])
            with left:
                st.plotly_chart(
                    sentiment_chart(df),
                    use_container_width=True,
                    key="overview_sentiment_chart",
                )
            with right:
                st.plotly_chart(
                    toxicity_chart(df),
                    use_container_width=True,
                    key="overview_toxicity_chart",
                )

            section_header("Topic Signal", "High-frequency audience language")
            wordcloud_fig = generate_wordcloud(df)
            st.pyplot(wordcloud_fig, use_container_width=True)
            overview_bottom_left, overview_bottom_right = st.columns(2)
            with overview_bottom_left:
                st.plotly_chart(
                    timeline_sentiment_chart(df),
                    use_container_width=True,
                    key="timeline_sentiment_chart",
                )
            with overview_bottom_right:
                st.plotly_chart(
                    toxicity_heatmap(df),
                    use_container_width=True,
                    key="toxicity_heatmap_chart",
                )

        with sentiment_tab:
            section_header(
                "Sentiment Analysis",
                "Audience mood distribution",
                "Interactive breakdown of positive, negative, and neutral reactions.",
            )
            st.plotly_chart(
                sentiment_chart(df),
                use_container_width=True,
                key="sentiment_detail_chart",
            )

            sentiment_counts = (
                df["sentiment"]
                .value_counts()
                .rename_axis("Sentiment")
                .reset_index(name="Count")
            )
            render_table(sentiment_counts, ["Sentiment", "Count"], height=180)

        with toxicity_tab:
            section_header(
                "Toxicity Analysis",
                "Moderation risk intelligence",
                "Model-based separation of toxic and safe audience comments.",
            )
            st.plotly_chart(
                toxicity_chart(df),
                use_container_width=True,
                key="toxicity_detail_chart",
            )

            toxic_df = df[df["toxicity_prediction"] == 1]
            if len(toxic_df) > 0:
                render_table(toxic_df.head(20), ["author", "comment", "likes"], height=420)
            else:
                status_card(
                    "Toxic comments",
                    "None detected",
                    "The current sample did not contain comments flagged as toxic.",
                    "cyan",
                )

        with spam_tab:
            section_header(
                "Advanced Spam & Bot Detection",
                "Repeated user, message, and suspicious pattern intelligence",
                "Detect coordinated behavior, promotional spam, and repetitive emoji content.",
            )

            spam_overview = st.columns(3)
            with spam_overview[0]:
                intelligence_panel("Spam Rate", f"{spam_percentage:.1f}%", "Detected repeated comments", "amber")
            with spam_overview[1]:
                intelligence_panel("Repeated Users", f"{len(suspicious_users):,}", "Potential bot-like accounts", "violet")
            with spam_overview[2]:
                intelligence_panel(
                    "Suspicious Patterns",
                    f"{max(0, len(spam_comments) + len(suspicious_users))}",
                    "Combined repeated text + user patterns",
                    "cyan",
                )

            if len(spam_comments) > 0:
                spam_df = spam_comments.reset_index()
                spam_df.columns = ["Comment", "Occurrences"]
                status_card(
                    "Spam activity",
                    "Detected",
                    f"{spam_percentage:.1f}% of the sample shows repeated comment behavior.",
                    "amber",
                )
                render_table(spam_df.head(25), ["Comment", "Occurrences"], height=420)
            else:
                status_card(
                    "Spam activity",
                    "Clean",
                    "No major repeated-comment pattern was detected in this sample.",
                    "cyan",
                )

            if not suspicious_users.empty:
                section_header("Suspicious Users", "Potential coordinated accounts")
                render_table(suspicious_users, ["author", "comments"], height=260)

        with engagement_tab:
            section_header(
                "Engagement Analytics",
                "Interaction and comment-depth signals",
                "Analyze how much effort and response intensity appears in the comment section.",
            )
            chart_left, chart_right = st.columns(2)
            with chart_left:
                st.plotly_chart(
                    comment_length_chart(df),
                    use_container_width=True,
                    key="comment_length_chart",
                )
            with chart_right:
                st.plotly_chart(
                    likes_chart(df),
                    use_container_width=True,
                    key="likes_chart",
                )
            engagement_lower_left, engagement_lower_right = st.columns(2)
            with engagement_lower_left:
                st.plotly_chart(
                    timeline_sentiment_chart(df),
                    use_container_width=True,
                    key="engagement_timeline_chart",
                )
            with engagement_lower_right:
                st.plotly_chart(
                    toxicity_heatmap(df),
                    use_container_width=True,
                    key="engagement_heatmap_chart",
                )

            longest_comments = df.sort_values(by="comment_length", ascending=False)
            render_table(
                longest_comments.head(10),
                ["author", "comment", "comment_length"],
                height=360,
            )

        with clusters_tab:
            section_header(
                "Audience Clusters",
                "Audience conversation clusters with semantic labels",
                "Topic-focused grouping reveals what each segment cares about most.",
            )
            st.plotly_chart(
                cluster_distribution_chart(df),
                use_container_width=True,
                key="cluster_distribution_chart",
            )

            for cluster_id, keywords in cluster_keywords.items():
                cluster_comments_df = df[df["cluster"] == cluster_id]
                cluster_name, cluster_tone = cluster_theme.get(cluster_id, (f"Cluster {cluster_id + 1}", "blue"))
                with st.expander(
                    f"{cluster_name}: {', '.join(keywords)}",
                    expanded=cluster_id == 0,
                ):
                    status_card(
                        "Cluster volume",
                        f"{len(cluster_comments_df)} comments",
                        f"Main topic signals: {', '.join(keywords)}",
                        cluster_tone,
                    )
                    render_table(
                        cluster_comments_df.head(10),
                        ["author", "comment", "likes", "sentiment"],
                        height=320,
                    )

        with insights_tab:
            section_header(
                "AI Insights",
                "Copilot-style audience interpretation",
                "Automated observations synthesized from sentiment, toxicity, spam, and satisfaction signals.",
            )

            insights = generate_insights(
                positive_percentage,
                negative_percentage,
                toxicity_percentage,
                spam_percentage,
                satisfaction_score,
            )

            for index, insight in enumerate(insights):
                insight_card(insight, index)

            behavior_left, behavior_right = st.columns(2)
            with behavior_left:
                if controversy_score > 35:
                    status_card(
                        "Behavioral signal",
                        "Highly divided audience",
                        "Positive and negative reactions are both substantial.",
                        "amber",
                    )
                elif positive_percentage > 70:
                    status_card(
                        "Behavioral signal",
                        "Strong audience approval",
                        "Positive reactions dominate the current sample.",
                        "cyan",
                    )
                else:
                    status_card(
                        "Behavioral signal",
                        "Balanced response",
                        "The audience reaction is not heavily polarized.",
                        "blue",
                    )

            with behavior_right:
                if toxicity_percentage > 25:
                    status_card(
                        "Moderation signal",
                        "High toxic engagement",
                        "The discussion may need closer comment review.",
                        "violet",
                    )
                elif spam_percentage > 5:
                    status_card(
                        "Moderation signal",
                        "Possible coordinated spam",
                        "Repeated comments may deserve manual inspection.",
                        "amber",
                    )
                else:
                    status_card(
                        "Moderation signal",
                        "Stable discussion quality",
                        "Toxicity and spam indicators are within a controlled range.",
                        "cyan",
                    )

            section_header(
                "Smart AI Summaries",
                "Structured interpretation of emotional and controversy trends",
            )
            summary_cols = st.columns(2)
            with summary_cols[0]:
                intelligence_panel(
                    "Most likely audience response",
                    "Praise-led momentum" if positive_percentage > 55 else "Mixed or critical",
                    "Behavior model based on sentiment and toxicity",
                    "blue",
                )
            with summary_cols[1]:
                intelligence_panel(
                    "Repeated complaints",
                    "Elevated" if negative_percentage > 35 else "Contained",
                    "Estimated from complaint-rich and negative patterns",
                    "amber",
                )

        with top_comments_tab:
            section_header(
                "Top Comments",
                "Most visible audience reactions",
                "Highest-liked comments and repeated keywords from the analyzed sample.",
            )
            top_comments = df.sort_values(by="likes", ascending=False)
            render_table(
                top_comments.head(20),
                ["author", "comment", "likes", "sentiment"],
                height=460,
            )

            all_words = " ".join(df["clean_comment"]).split()
            common_words = Counter(all_words)
            common_df = pd.DataFrame(
                common_words.most_common(20),
                columns=["Keyword", "Frequency"],
            )
            render_table(common_df, ["Keyword", "Frequency"], height=420)

            section_header(
                "Most Influential Users",
                "Top commenters ranked by engagement and emotional impact",
            )
            influencer_df = (
                df.assign(influence_score=df["likes"] * 1.6 + df["comment_length"] * 0.25)
                .sort_values("influence_score", ascending=False)
                .drop_duplicates(subset=["author"])
                .head(5)
            )
            influencer_cols = st.columns(min(5, max(1, len(influencer_df))))
            for idx, (_, row) in enumerate(influencer_df.iterrows()):
                with influencer_cols[idx]:
                    user_card(
                        row["author"],
                        f"{row['influence_score']:.1f}",
                        f"{int(row['likes'])} likes",
                    )

            section_header(
                "Most Emotional Comments",
                "High-intensity reactions based on engagement and polarity",
            )
            emotional_df = df.sort_values("emotion_intensity", ascending=False).head(12)
            render_table(
                emotional_df,
                ["author", "comment", "emotion_intensity", "sentiment"],
                height=340,
            )

        with psychology_tab:
            section_header(
                "Audience Psychology Analysis",
                "Mindset, friction points, and motivation signals",
                "A synthetic behavioral map of what drives audience approval, complaints, and conflict.",
            )
            psych_cols = st.columns(3)
            with psych_cols[0]:
                intelligence_panel(
                    "Emotional Stability",
                    f"{max(0, 100 - controversy_score):.1f}",
                    "Higher is more coherent audience response",
                    "cyan",
                )
            with psych_cols[1]:
                intelligence_panel(
                    "Negative Reaction Trigger",
                    "Toxicity" if toxicity_percentage > negative_percentage else "Critical sentiment",
                    "Estimated dominant pain source",
                    "violet",
                )
            with psych_cols[2]:
                intelligence_panel(
                    "Audience Friction Level",
                    f"{(negative_percentage + toxicity_percentage * 0.7):.1f}",
                    "Composite conflict indicator",
                    "amber",
                )
            st.plotly_chart(
                timeline_sentiment_chart(df),
                use_container_width=True,
                key="psychology_timeline_chart",
            )
            st.plotly_chart(
                toxicity_heatmap(df),
                use_container_width=True,
                key="psychology_heatmap_chart",
            )

        with raw_data_tab:
            section_header(
                "Raw Data",
                "Complete analyzed comment dataset",
                "Export the enriched table with sentiment, toxicity, spam context, and cluster labels.",
            )
            if show_raw_columns:
                raw_view = filtered_df
            else:
                raw_view = filtered_df[
                    [
                        "author",
                        "comment",
                        "likes",
                        "sentiment",
                        "toxicity_prediction",
                        "cluster",
                    ]
                ]

            st.dataframe(
                raw_view,
                use_container_width=True,
                height=520,
                hide_index=True,
            )

            csv = raw_view.to_csv(index=False)
            st.download_button(
                label="Download Analysis Report",
                data=csv,
                file_name="youtube_analysis.csv",
                mime="text/csv",
                use_container_width=True,
            )

else:
    st.markdown(
        """
        <div class="empty-state fade-up">
            <span>Awaiting signal</span>
            <h3>Paste a YouTube URL to initialize the intelligence dashboard.</h3>
            <p>The analytics console will expand here after analysis.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
