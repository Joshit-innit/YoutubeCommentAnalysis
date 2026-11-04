"""Small HTML building blocks for the dashboard.

Everything that comes from YouTube (titles, authors, comments) is escaped here,
and st.html is used so that text is never parsed as Markdown.
"""

import re
from contextlib import contextmanager
from datetime import datetime
from html import escape
from pathlib import Path

import streamlit as st

from components.charts import SENTIMENT_COLORS

CSS_PATH = Path(__file__).resolve().parent.parent / "styles" / "custom.css"

STATUS_ICONS = {
    "good": "✓",
    "warning": "▲",
    "critical": "■",
    "neutral": "–",
}


def load_css():
    st.markdown(
        f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>",
        unsafe_allow_html=True,
    )


def compact_number(value):
    if value is None:
        return "N/A"

    for threshold, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(value) >= threshold:
            return f"{value / threshold:.1f}".removesuffix(".0") + suffix

    return f"{value:,}"


def format_date(value):
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).strftime("%b %-d, %Y")
    except ValueError:
        return "Unknown date"


def video_header(meta, analyzed):
    thumbnail = meta.get("thumbnail")
    image = f'<img src="{escape(thumbnail, quote=True)}" alt="">' if thumbnail else ""
    facts = [
        meta.get("channel", "Unknown channel"),
        f"{compact_number(meta.get('views', 0))} views",
        f"{compact_number(meta.get('likes', 0))} likes",
        f"{compact_number(meta.get('subscribers'))} subscribers",
        f"Uploaded {format_date(meta.get('published_at'))}",
        f"{analyzed:,} comments analyzed",
    ]

    st.html(
        f"""
        <div class="video-header">
            {image}
            <div>
                <p class="video-title">{escape(meta.get("title", "Untitled video"))}</p>
                <p class="video-meta">{escape(" · ".join(facts))}</p>
            </div>
        </div>
        """
    )


def page_title(title):
    st.html(f'<h1 class="page-title">{escape(title)}</h1>')


def kpi_row(items):
    """Render (label, value, note) tuples as a responsive metric strip."""
    cards = "".join(
        f"""
        <div class="kpi">
            <div class="kpi-label">{escape(label)}</div>
            <div class="kpi-value">{escape(str(value))}</div>
            <div class="kpi-note">{escape(note)}</div>
        </div>
        """
        for label, value, note in items
    )
    st.html(f'<div class="kpi-row">{cards}</div>')


@contextmanager
def panel(title, note=None):
    """A bordered dashboard card with a title and an optional one-line note."""
    key = "panel_" + re.sub(r"\W+", "_", title.lower())

    with st.container(border=True, key=key):
        note_markup = f'<div class="panel-note">{escape(note)}</div>' if note else ""
        st.html(
            f"""
            <div class="panel-head">
                <div class="panel-title">{escape(title)}</div>
                {note_markup}
            </div>
            """
        )
        yield


def sentiment_split(counts, total):
    """A thin segmented bar above a list carrying each sentiment's figures."""
    order = ["Positive", "Neutral", "Negative"]
    segments = "".join(
        f'<span style="flex:{counts.get(name, 0)};background:{SENTIMENT_COLORS[name]}" '
        f'title="{name}: {counts.get(name, 0):,}"></span>'
        for name in order
        if counts.get(name, 0)
    )
    rows = "".join(
        f"""
        <div class="row">
            <span class="swatch" style="background:{SENTIMENT_COLORS[name]}"></span>
            <span class="row-label">{name}</span>
            <span class="row-value">{counts.get(name, 0):,}</span>
            <span class="row-share">{counts.get(name, 0) / total * 100:.1f}%</span>
        </div>
        """
        for name in order
    )
    st.html(f'<div class="split-bar">{segments}</div><div class="rows">{rows}</div>')


def status_rows(items, notes=()):
    """Render (label, status text, tone) rows, then optional plain-text notes."""
    rows = "".join(
        f"""
        <div class="row">
            <span class="row-label">{escape(label)}</span>
            <span class="status status-{tone}">
                <span aria-hidden="true">{STATUS_ICONS[tone]}</span>{escape(text)}
            </span>
        </div>
        """
        for label, text, tone in items
    )
    note_list = ""
    if notes:
        entries = "".join(f"<li>{escape(note)}</li>" for note in notes)
        note_list = f'<ul class="note-list">{entries}</ul>'

    st.html(f'<div class="rows">{rows}</div>{note_list}')


def score_rows(items):
    """Render (label, value, 0-100 fill) rows with a small meter."""
    rows = "".join(
        f"""
        <div class="row">
            <span class="row-label">{escape(label)}</span>
            <span class="meter"><span style="width:{max(0, min(100, fill)):.0f}%"></span></span>
            <span class="row-value">{value:.1f}</span>
        </div>
        """
        for label, value, fill in items
    )
    st.html(f'<div class="rows">{rows}</div>')


def empty_state():
    st.html(
        """
        <div class="empty">
            <h1>Analyze a video's comments</h1>
            <p>See sentiment, toxicity, spam and discussion topics for any public YouTube video.</p>
            <ol>
                <li>Paste a video URL in the sidebar.</li>
                <li>Choose how many comments to sample.</li>
                <li>Select Analyze.</li>
            </ol>
        </div>
        """
    )
