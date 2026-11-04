import pandas as pd
import plotly.graph_objects as go

SURFACE = "#ffffff"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#6f6d68"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
ACCENT = "#256abf"
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'

# Sentiment is polarity, so it uses a diverging pair around a gray midpoint.
# Blue and red also stay distinct for red-green colorblind readers.
SENTIMENT_ORDER = ["Negative", "Neutral", "Positive"]
SENTIMENT_COLORS = {
    "Positive": "#2a78d6",
    "Neutral": "#c3c2b7",
    "Negative": "#e34948",
}

HEAT_SCALE = [
    [0.0, "#f4f3ef"],
    [0.01, "#cde2fb"],
    [0.5, "#5598e7"],
    [1.0, "#0d366b"],
]


def _style(fig, height=300, x_title=None, y_title=None, legend=False):
    fig.update_layout(
        height=height,
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font={"family": FONT, "size": 12, "color": INK_SECONDARY},
        margin={"l": 4, "r": 8, "t": 28 if legend else 8, "b": 4},
        showlegend=legend,
        legend={
            "orientation": "h",
            "x": 0,
            "xanchor": "left",
            "y": 1,
            "yanchor": "bottom",
            "traceorder": "reversed",
            "title": {"text": ""},
        },
        hoverlabel={
            "bgcolor": SURFACE,
            "bordercolor": AXIS,
            "font": {"family": FONT, "size": 12, "color": INK},
        },
    )
    fig.update_xaxes(
        title={"text": x_title, "font": {"size": 12, "color": INK_MUTED}},
        showgrid=False,
        zeroline=False,
        linecolor=AXIS,
        ticks="",
        tickfont={"color": INK_MUTED},
        automargin=True,
    )
    fig.update_yaxes(
        title={"text": y_title, "font": {"size": 12, "color": INK_MUTED}},
        showgrid=True,
        gridcolor=GRID,
        zeroline=False,
        showline=False,
        ticks="",
        tickfont={"color": INK_MUTED},
        rangemode="tozero",
        automargin=True,
    )
    return fig


def _empty(message, height=300):
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 13, "color": INK_MUTED},
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(
        height=height,
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        margin={"l": 8, "r": 8, "t": 8, "b": 8},
    )
    return fig


def _time_bucket(dates):
    """Pick the finest bucket that keeps the chart under about 45 columns."""
    days = (dates.max() - dates.min()).days

    if days <= 45:
        return "D", "Day"
    if days <= 315:
        return "W-MON", "Week starting"
    return "MS", "Month"


def sentiment_timeline_chart(df, height=300):
    timeline = df.dropna(subset=["published_at"]).copy()

    if timeline.empty:
        return _empty("No timestamps available", height)

    timeline["published_at"] = pd.to_datetime(timeline["published_at"]).dt.tz_localize(None)
    frequency, label = _time_bucket(timeline["published_at"])
    label_format = "%b %Y" if frequency == "MS" else "%b %-d, %Y"
    closed = {"closed": "left", "label": "left"} if frequency == "W-MON" else {}

    grouped = (
        timeline.set_index("published_at")
        .groupby("sentiment")
        .resample(frequency, **closed)
        .size()
        .rename("count")
        .reset_index()
    )

    fig = go.Figure()

    for sentiment in SENTIMENT_ORDER:
        series = grouped[grouped["sentiment"] == sentiment]
        fig.add_trace(
            go.Bar(
                name=sentiment,
                x=series["published_at"],
                y=series["count"],
                marker={
                    "color": SENTIMENT_COLORS[sentiment],
                    "line": {"color": SURFACE, "width": 1},
                },
                customdata=series["published_at"].dt.strftime(label_format),
                hovertemplate=(
                    f"{label} %{{customdata}}<br>"
                    f"{sentiment}: %{{y:,}}<extra></extra>"
                ),
            )
        )

    fig.update_layout(barmode="stack", bargap=0.2)
    return _style(fig, height=height, y_title="Comments", legend=True)


def toxicity_heatmap(df, height=300):
    timed = df.dropna(subset=["published_at"]).copy()

    if timed.empty:
        return _empty("No timestamps available", height)

    published = pd.to_datetime(timed["published_at"])
    timed["hour"] = published.dt.hour
    timed["day"] = published.dt.day_name().str.slice(0, 3)

    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    hours = list(range(24))

    def weekly_grid(frame):
        return (
            frame.groupby(["day", "hour"])
            .size()
            .unstack(fill_value=0)
            .reindex(index=days, columns=hours, fill_value=0)
        )

    flagged = weekly_grid(timed[timed["toxicity_prediction"] == 1])
    totals = weekly_grid(timed)

    fig = go.Figure(
        go.Heatmap(
            z=flagged.values,
            x=hours,
            y=days,
            customdata=totals.values,
            colorscale=HEAT_SCALE,
            zmin=0,
            zmax=max(1, flagged.values.max()),
            xgap=2,
            ygap=2,
            colorbar={
                "title": {"text": "Flagged", "font": {"size": 12}},
                "thickness": 10,
                "outlinewidth": 0,
                "tickformat": "d",
                "dtick": max(1, round(int(flagged.values.max()) / 4)),
            },
            hovertemplate=(
                "%{y} %{x}:00 UTC<br>"
                "%{z:,} flagged of %{customdata:,} comments<extra></extra>"
            ),
        )
    )
    _style(fig, height=height, x_title="Hour (UTC)")
    fig.update_xaxes(dtick=3, showline=False)
    fig.update_yaxes(autorange="reversed", showgrid=False, rangemode="normal")
    return fig


def ranked_bar_chart(labels, values, value_name="Comments", height=300):
    """Horizontal bars, largest on top, in one hue."""
    fig = go.Figure(
        go.Bar(
            x=list(values)[::-1],
            y=list(labels)[::-1],
            orientation="h",
            marker={"color": ACCENT, "cornerradius": 3},
            text=[f"{value:,}" for value in list(values)[::-1]],
            textposition="outside",
            cliponaxis=False,
            textfont={"color": INK_SECONDARY},
            hovertemplate=f"%{{y}}<br>{value_name}: %{{x:,}}<extra></extra>",
        )
    )
    fig.update_layout(bargap=0.3)
    _style(fig, height=height)
    fig.update_xaxes(visible=False, rangemode="tozero")
    fig.update_yaxes(showgrid=False, automargin=True, rangemode="normal", tickfont={"color": INK_SECONDARY}, ticksuffix="  ")
    fig.update_layout(margin={"r": 48})
    return fig


def comment_length_chart(df, height=300):
    fig = go.Figure(
        go.Histogram(
            x=df["comment_length"],
            nbinsx=30,
            marker={"color": ACCENT, "line": {"color": SURFACE, "width": 1}},
            hovertemplate="%{x} characters<br>Comments: %{y:,}<extra></extra>",
        )
    )
    return _style(fig, height=height, x_title="Characters", y_title="Comments")


def likes_chart(df, height=300):
    """Likes are heavily skewed, so they are shown in widening bands."""
    bands = [
        ("0", 0, 0),
        ("1", 1, 1),
        ("2–5", 2, 5),
        ("6–20", 6, 20),
        ("21–100", 21, 100),
        ("101+", 101, float("inf")),
    ]
    labels = [band[0] for band in bands]
    counts = [int(df["likes"].between(low, high).sum()) for _, low, high in bands]

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=counts,
            marker={"color": ACCENT, "cornerradius": 3},
            hovertemplate="%{x} likes<br>Comments: %{y:,}<extra></extra>",
        )
    )
    fig.update_layout(bargap=0.3)
    _style(fig, height=height, x_title="Likes per comment", y_title="Comments")
    fig.update_xaxes(type="category")
    return fig
