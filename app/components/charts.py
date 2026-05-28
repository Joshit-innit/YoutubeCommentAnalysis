import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from wordcloud import WordCloud


CHART_TEMPLATE = "plotly_dark"
PAPER_BG = "rgba(0,0,0,0)"
PLOT_BG = "rgba(7,12,26,0.34)"
FONT_COLOR = "#dce8f8"
GRID_COLOR = "rgba(148,163,184,0.14)"
CYAN = "#22d3ee"
BLUE = "#60a5fa"
VIOLET = "#a78bfa"
PINK = "#f472b6"
GREEN = "#34d399"
AMBER = "#fbbf24"
RED = "#fb7185"
YT_RED = "#ff0033"


def style_figure(fig, title):
    fig.update_layout(
        template=CHART_TEMPLATE,
        title={
            "text": title,
            "x": 0.03,
            "xanchor": "left",
            "font": {
                "size": 22,
                "color": "#f8fbff",
                "family": "Inter, system-ui, sans-serif",
            },
        },
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        font={
            "color": FONT_COLOR,
            "family": "Inter, system-ui, sans-serif",
        },
        margin={
            "l": 32,
            "r": 28,
            "t": 78,
            "b": 40,
        },
        hoverlabel={
            "bgcolor": "rgba(7,12,26,0.96)",
            "bordercolor": CYAN,
            "font_color": "#f8fbff",
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
        transition={
            "duration": 550,
            "easing": "cubic-in-out",
        },
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor=GRID_COLOR,
        zeroline=False,
        linecolor="rgba(148,163,184,0.25)",
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=GRID_COLOR,
        zeroline=False,
        linecolor="rgba(148,163,184,0.25)",
    )

    return fig


def sentiment_chart(df):
    sentiment_counts = (
        df["sentiment"]
        .value_counts()
        .reset_index()
    )

    sentiment_counts.columns = [
        "Sentiment",
        "Count",
    ]

    color_map = {
        "Positive": GREEN,
        "Neutral": BLUE,
        "Negative": RED,
    }

    fig = px.pie(
        sentiment_counts,
        names="Sentiment",
        values="Count",
        hole=0.58,
        color="Sentiment",
        color_discrete_map=color_map,
    )

    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        marker={
            "line": {
                "color": "rgba(255,255,255,0.18)",
                "width": 1.4,
            }
        },
        hovertemplate="<b>%{label}</b><br>Comments: %{value}<br>Share: %{percent}<extra></extra>",
        pull=[0.03] * len(sentiment_counts),
    )

    return style_figure(fig, "Audience Sentiment Matrix")


def toxicity_chart(df):
    toxicity_counts = (
        df["toxicity_prediction"]
        .value_counts()
        .reset_index()
    )

    toxicity_counts.columns = [
        "Type",
        "Count",
    ]

    toxicity_counts["Type"] = toxicity_counts["Type"].map(
        {
            0: "Safe",
            1: "Toxic",
        }
    )

    toxicity_counts["Color"] = toxicity_counts["Type"].map(
        {
            "Safe": GREEN,
            "Toxic": RED,
        }
    )

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=toxicity_counts["Type"],
            y=toxicity_counts["Count"],
            marker={
                "color": toxicity_counts["Color"],
                "line": {
                    "color": "rgba(255,255,255,0.22)",
                    "width": 1,
                },
            },
            text=toxicity_counts["Count"],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Comments: %{y}<extra></extra>",
        )
    )

    fig.update_layout(bargap=0.42)

    return style_figure(fig, "Toxicity Risk Distribution")


def comment_length_chart(df):
    fig = px.histogram(
        df,
        x="comment_length",
        nbins=32,
        color_discrete_sequence=[CYAN],
    )

    fig.update_traces(
        marker={
            "line": {
                "color": "rgba(255,255,255,0.10)",
                "width": 1,
            }
        },
        hovertemplate="Length: %{x}<br>Comments: %{y}<extra></extra>",
    )

    fig.update_layout(bargap=0.08)
    fig.update_xaxes(title_text="Comment length")
    fig.update_yaxes(title_text="Comments")

    return style_figure(fig, "Comment Length Spectrum")


def likes_chart(df):
    fig = px.histogram(
        df,
        x="likes",
        nbins=32,
        color_discrete_sequence=[VIOLET],
    )

    fig.update_traces(
        marker={
            "line": {
                "color": "rgba(255,255,255,0.10)",
                "width": 1,
            }
        },
        hovertemplate="Likes: %{x}<br>Comments: %{y}<extra></extra>",
    )

    fig.update_layout(bargap=0.08)
    fig.update_xaxes(title_text="Likes")
    fig.update_yaxes(title_text="Comments")

    return style_figure(fig, "Engagement Pulse")


def generate_wordcloud(df):
    all_words = " ".join(
        df["comment"]
        .astype(str)
    )

    wordcloud = WordCloud(
        width=1400,
        height=620,
        background_color="#070c1a",
        colormap="cool",
        contour_color="#22d3ee",
        contour_width=1,
        prefer_horizontal=0.92,
    ).generate(all_words)

    fig, ax = plt.subplots(
        figsize=(14, 6.2),
        facecolor="#070c1a",
    )

    ax.imshow(wordcloud, interpolation="bilinear")
    ax.set_facecolor("#070c1a")
    ax.axis("off")

    return fig


def timeline_sentiment_chart(df):
    timeline_df = df.copy()
    timeline_df["published_at"] = pd.to_datetime(
        timeline_df["published_at"],
        errors="coerce",
    )
    timeline_df = timeline_df.dropna(subset=["published_at"])

    if timeline_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No timeline data available",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font={"size": 16, "color": FONT_COLOR},
        )
        return style_figure(fig, "Comment Timeline Analysis")

    timeline_df["date"] = timeline_df["published_at"].dt.date
    grouped = (
        timeline_df.groupby(
            [
                "date",
                "sentiment",
            ]
        )
        .size()
        .reset_index(name="count")
    )

    fig = px.area(
        grouped,
        x="date",
        y="count",
        color="sentiment",
        color_discrete_map={
            "Positive": GREEN,
            "Negative": RED,
            "Neutral": BLUE,
        },
    )
    fig.update_traces(mode="lines", line={"width": 2.2})
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Comment volume")
    return style_figure(fig, "Comment Timeline Analysis")


def toxicity_heatmap(df):
    heat_df = df.copy()
    heat_df["published_at"] = pd.to_datetime(heat_df["published_at"], errors="coerce")
    heat_df = heat_df.dropna(subset=["published_at"])

    if heat_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No timestamp data for heatmap",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font={"size": 16, "color": FONT_COLOR},
        )
        return style_figure(fig, "Toxic Activity Heatmap")

    heat_df["hour"] = heat_df["published_at"].dt.hour
    heat_df["day"] = heat_df["published_at"].dt.day_name().str.slice(0, 3)
    day_order = [
        "Mon",
        "Tue",
        "Wed",
        "Thu",
        "Fri",
        "Sat",
        "Sun",
    ]

    matrix = (
        heat_df.groupby(["day", "hour"])["toxicity_prediction"]
        .mean()
        .reset_index(name="toxicity_rate")
    )
    matrix["day"] = pd.Categorical(matrix["day"], categories=day_order, ordered=True)
    matrix = matrix.sort_values(["day", "hour"])

    fig = px.density_heatmap(
        matrix,
        x="hour",
        y="day",
        z="toxicity_rate",
        color_continuous_scale=[
            [0.0, "rgba(34,211,238,0.2)"],
            [0.5, "rgba(167,139,250,0.7)"],
            [1.0, YT_RED],
        ],
    )
    fig.update_layout(coloraxis_colorbar={"title": "Toxicity"})
    fig.update_xaxes(title_text="Hour of day")
    fig.update_yaxes(title_text="Day")
    return style_figure(fig, "Toxic Activity Heatmap")


def cluster_distribution_chart(df):
    cluster_counts = df["cluster_name"].value_counts().reset_index()
    cluster_counts.columns = ["Cluster", "Comments"]
    fig = px.bar(
        cluster_counts,
        x="Cluster",
        y="Comments",
        color="Cluster",
        color_discrete_sequence=[CYAN, VIOLET, BLUE, PINK, AMBER],
    )
    fig.update_traces(
        marker_line_color="rgba(255,255,255,0.20)",
        marker_line_width=1.2,
        hovertemplate="<b>%{x}</b><br>Comments: %{y}<extra></extra>",
    )
    fig.update_layout(showlegend=False)
    return style_figure(fig, "Audience Cluster Distribution")
