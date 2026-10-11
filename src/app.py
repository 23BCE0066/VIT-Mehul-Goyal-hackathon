"""
Interactive Financial Risk & Intelligence Platform
S&P Global & CRISIL Campus Hackathon 2026
Candidate: Mehul Goyal | VIT

UI: Clean Corporate Dashboard (Novera-style)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.engine.ingestion import IngestionPipeline
from src.engine.nlp_processor import NLPRiskEngine
from src.module_a.index_rebalancer import TacticalIndexRebalancer
from src.module_b.synthetic_book import WholesalePortfolioBook
from src.module_b.stress_engine import StrategicStressTester

st.set_page_config(
    page_title="S&P Global & CRISIL | AI/NLP Risk Intelligence Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# CLEAN CORPORATE CSS  (Novera-inspired: white, minimal borders, clean type)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ═══════ GLOBAL ═══════ */
:root {
    --bg: #FAFBFD;
    --white: #FFFFFF;
    --border: #E8ECF1;
    --border-light: #F0F2F5;
    --text-primary: #1A1D26;
    --text-secondary: #5E6470;
    --text-muted: #9BA1B0;
    --accent: #2563EB;
    --accent-bg: #EEF2FF;
    --green: #16A34A;
    --green-bg: #F0FDF4;
    --red: #DC2626;
    --red-bg: #FEF2F2;
    --orange: #EA580C;
    --orange-bg: #FFF7ED;
    --radius: 12px;
    --radius-sm: 8px;
    --shadow: 0 1px 3px rgba(0,0,0,0.04), 0 1px 2px rgba(0,0,0,0.02);
    --shadow-md: 0 4px 12px rgba(0,0,0,0.05), 0 1px 3px rgba(0,0,0,0.03);
}

html, body, [data-testid="stAppViewContainer"], .stApp {
    background-color: var(--bg) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    color: var(--text-primary) !important;
}
[data-testid="stHeader"] {
    background: var(--white) !important;
    border-bottom: 1px solid var(--border) !important;
}
[data-testid="stMainBlockContainer"] {
    padding-top: 2rem !important;
    max-width: 1400px;
}

/* ═══════ SIDEBAR ═══════ */
[data-testid="stSidebar"] {
    background: var(--white) !important;
    border-right: 1px solid var(--border) !important;
    box-shadow: none !important;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label {
    color: var(--text-secondary) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
}

/* Sidebar nav radio - replicate Novera pill style */
[data-testid="stSidebar"] .stRadio > div {
    gap: 2px !important;
}
[data-testid="stSidebar"] .stRadio > div > label {
    background: transparent !important;
    border-radius: 8px !important;
    padding: 9px 14px !important;
    border: none !important;
    box-shadow: none !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    color: var(--text-secondary) !important;
    transition: all 0.15s ease !important;
}
[data-testid="stSidebar"] .stRadio > div > label:hover {
    background: var(--bg) !important;
    color: var(--text-primary) !important;
}
[data-testid="stSidebar"] .stRadio > div > label[data-checked="true"],
[data-testid="stSidebar"] .stRadio > div > label[aria-checked="true"] {
    background: var(--accent-bg) !important;
    color: var(--accent) !important;
    font-weight: 600 !important;
}

/* ═══════ CARD COMPONENT ═══════ */
.card {
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 24px;
    margin-bottom: 16px;
    box-shadow: var(--shadow);
}
.card:hover {
    box-shadow: var(--shadow-md);
}
.card-title {
    font-size: 13px;
    font-weight: 500;
    color: var(--text-muted);
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.card-title .dots {
    color: var(--text-muted);
    font-size: 16px;
    letter-spacing: 2px;
    cursor: pointer;
}
.card-value {
    font-size: 32px;
    font-weight: 700;
    color: var(--text-primary);
    line-height: 1.1;
    letter-spacing: -0.5px;
}
.card-value.medium {
    font-size: 26px;
}
.card-sub {
    font-size: 13px;
    color: var(--text-muted);
    margin-top: 6px;
    font-weight: 400;
}

/* ═══════ DELTA BADGES ═══════ */
.delta {
    display: inline-flex;
    align-items: center;
    gap: 3px;
    font-size: 12px;
    font-weight: 600;
    padding: 3px 8px;
    border-radius: 6px;
}
.delta-up {
    color: var(--green);
    background: var(--green-bg);
}
.delta-down {
    color: var(--red);
    background: var(--red-bg);
}
.delta-neutral {
    color: var(--text-muted);
    background: var(--bg);
}

/* ═══════ BADGES / PILLS ═══════ */
.pill {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.3px;
}
.pill-danger {
    background: var(--red-bg);
    color: var(--red);
}
.pill-success {
    background: var(--green-bg);
    color: var(--green);
}
.pill-info {
    background: var(--accent-bg);
    color: var(--accent);
}
.pill-warning {
    background: var(--orange-bg);
    color: var(--orange);
}
.pill-muted {
    background: var(--bg);
    color: var(--text-secondary);
    border: 1px solid var(--border);
}

/* ═══════ FEED / LIST ITEMS ═══════ */
.feed-row {
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 16px 20px;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 16px;
    transition: box-shadow 0.15s ease;
}
.feed-row:hover {
    box-shadow: var(--shadow-md);
}
.feed-row.systemic {
    border-left: 3px solid var(--red);
}
.feed-row.standard {
    border-left: 3px solid var(--green);
}
.feed-icon {
    width: 40px;
    height: 40px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    flex-shrink: 0;
}
.feed-icon.news { background: var(--accent-bg); }
.feed-icon.social { background: var(--orange-bg); }

.feed-body { flex: 1; min-width: 0; }
.feed-headline {
    font-size: 13.5px;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 4px;
    line-height: 1.4;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.feed-snippet {
    font-size: 12.5px;
    color: var(--text-muted);
    line-height: 1.4;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
}
.feed-meta {
    display: flex;
    gap: 8px;
    align-items: center;
    margin-top: 8px;
    flex-wrap: wrap;
}
.feed-stats {
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    gap: 4px;
    flex-shrink: 0;
    min-width: 90px;
}
.feed-stat-value {
    font-size: 18px;
    font-weight: 700;
}
.feed-stat-label {
    font-size: 11px;
    color: var(--text-muted);
    font-weight: 500;
}

/* ═══════ SECTION HEADERS ═══════ */
.page-title {
    font-size: 24px;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.3px;
    margin-bottom: 4px;
}
.page-subtitle {
    font-size: 14px;
    color: var(--text-secondary);
    line-height: 1.5;
    margin-bottom: 24px;
    font-weight: 400;
}
.section-label {
    font-size: 13px;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 6px;
}
.section-title {
    font-size: 16px;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.section-title .dots {
    color: var(--text-muted);
    font-size: 16px;
    letter-spacing: 2px;
}

/* ═══════ AI INSIGHT BOX ═══════ */
.insight-box {
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 20px;
    margin-bottom: 16px;
}
.insight-box .insight-header {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 14px;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 8px;
}
.insight-box .insight-text {
    font-size: 13px;
    color: var(--text-secondary);
    line-height: 1.6;
}

/* ═══════ SPARKLINE BAR ═══════ */
.sparkbar {
    display: inline-flex;
    align-items: flex-end;
    gap: 2px;
    height: 28px;
    vertical-align: middle;
    margin-left: 12px;
}
.sparkbar .bar {
    width: 4px;
    background: var(--accent);
    border-radius: 2px;
    opacity: 0.5;
}

/* ═══════ STREAMLIT OVERRIDES ═══════ */
[data-testid="stMetric"] {
    background: var(--white) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 20px !important;
    box-shadow: var(--shadow) !important;
}
[data-testid="stMetricLabel"] {
    color: var(--text-muted) !important;
    font-weight: 500 !important;
    font-size: 13px !important;
}
[data-testid="stMetricValue"] {
    color: var(--text-primary) !important;
    font-weight: 700 !important;
    font-size: 24px !important;
}

[data-testid="stExpander"] {
    background: var(--white) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    box-shadow: none !important;
    margin-bottom: 8px !important;
}
[data-testid="stExpander"] summary {
    color: var(--text-primary) !important;
    font-weight: 500 !important;
    font-size: 14px !important;
}

.stSlider > div > div > div {
    background: var(--accent) !important;
}

.stButton > button {
    background: var(--white) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    box-shadow: var(--shadow) !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    padding: 8px 20px !important;
    transition: all 0.15s ease !important;
}
.stButton > button:hover {
    background: var(--bg) !important;
    border-color: var(--accent) !important;
    color: var(--accent) !important;
}
.stButton > button[kind="primary"],
.stButton > button[data-testid="stBaseButton-primary"] {
    background: var(--accent) !important;
    color: var(--white) !important;
    border: none !important;
}
.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="stBaseButton-primary"]:hover {
    background: #1D4ED8 !important;
    color: var(--white) !important;
}

.stDownloadButton > button {
    background: var(--white) !important;
    color: var(--text-secondary) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    font-size: 13px !important;
}

.stTextArea textarea {
    background: var(--white) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
    color: var(--text-primary) !important;
}
.stTextArea textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.08) !important;
}

[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    overflow: hidden;
}

.stAlert {
    border-radius: var(--radius-sm) !important;
    border-left-width: 4px !important;
}

/* ═══════ SCROLLBAR ═══════ */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: #D1D5DB; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #9CA3AF; }

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY CLEAN THEME
# ─────────────────────────────────────────────────────────────────────────────
CLEAN_LAYOUT = dict(
    paper_bgcolor="rgba(255,255,255,0)",
    plot_bgcolor="rgba(255,255,255,0)",
    font=dict(family="Inter, sans-serif", color="#1A1D26", size=12),
    title_font=dict(size=14, color="#1A1D26", family="Inter"),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#5E6470", size=11)),
    xaxis=dict(gridcolor="#F0F2F5", zerolinecolor="#E8ECF1", linecolor="#E8ECF1"),
    yaxis=dict(gridcolor="#F0F2F5", zerolinecolor="#E8ECF1", linecolor="#E8ECF1"),
    margin=dict(l=40, r=20, t=44, b=40),
)
CLEAN_COLORS = ["#2563EB", "#16A34A", "#EA580C", "#DC2626", "#7C3AED", "#0891B2", "#CA8A04"]

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def metric_card(title, value, delta=None, delta_dir="up", subtitle=""):
    delta_html = ""
    if delta:
        cls = "delta-up" if delta_dir == "up" else ("delta-down" if delta_dir == "down" else "delta-neutral")
        arrow = "↑" if delta_dir == "up" else ("↓" if delta_dir == "down" else "")
        delta_html = f'<span class="delta {cls}">{arrow} {delta}</span>'

    sub_html = f'<div class="card-sub">{subtitle}</div>' if subtitle else ""
    return f"""
    <div class="card">
        <div class="card-title">{title}<span class="dots">···</span></div>
        <div style="display: flex; align-items: baseline; gap: 12px;">
            <div class="card-value">{value}</div>
            {delta_html}
        </div>
        {sub_html}
    </div>
    """


def feed_item_html(item):
    is_sys = item["impact_score"] >= 7
    row_cls = "feed-row systemic" if is_sys else "feed-row standard"
    icon_cls = "feed-icon news" if item["source_type"] == "news" else "feed-icon social"
    icon_emoji = "📰" if item["source_type"] == "news" else "💬"

    pills = ""
    if is_sys:
        pills += '<span class="pill pill-danger">Systemic</span> '
    else:
        pills += '<span class="pill pill-success">Standard</span> '
    pills += f'<span class="pill pill-info">{item["event_classification"]}</span> '
    for e in item.get("target_entities", [])[:3]:
        pills += f'<span class="pill pill-muted">{e}</span> '

    sent = item["sentiment_score"]
    sent_color = "var(--green)" if sent > 0 else "var(--red)"

    return f"""
    <div class="{row_cls}">
        <div class="{icon_cls}">{icon_emoji}</div>
        <div class="feed-body">
            <div class="feed-headline">{item.get('headline') or item['text_snippet'][:70]}</div>
            <div class="feed-snippet">{item['text_snippet'][:140]}</div>
            <div class="feed-meta">{pills}</div>
        </div>
        <div class="feed-stats">
            <div class="feed-stat-value" style="color: {sent_color};">{sent:+.2f}</div>
            <div class="feed-stat-label">Sentiment</div>
            <div class="feed-stat-value" style="color: var(--text-primary);">{item['impact_score']}<span style="font-size:12px; color: var(--text-muted); font-weight:400;">/10</span></div>
            <div class="feed-stat-label">Impact</div>
        </div>
    </div>
    """


# ─────────────────────────────────────────────────────────────────────────────
# LOAD COMPONENTS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_components():
    return IngestionPipeline(), NLPRiskEngine(), TacticalIndexRebalancer(), StrategicStressTester()

pipeline, nlp, rebalancer, stress_tester = load_components()

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.markdown("""
<div style="padding: 8px 0 16px 0;">
    <div style="display: flex; align-items: center; gap: 10px;">
        <div style="width: 36px; height: 36px; background: #2563EB; border-radius: 10px; display: flex; align-items: center; justify-content: center; color: white; font-weight: 700; font-size: 16px;">S&P</div>
        <div>
            <div style="font-size: 16px; font-weight: 700; color: #1A1D26;">Risk Engine</div>
            <div style="font-size: 11px; color: #9BA1B0;">Hackathon 2026</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown('<div class="section-label">MAIN MENU</div>', unsafe_allow_html=True)

app_mode = st.sidebar.radio(
    "Navigation",
    [
        "📊  Dashboard",
        "🧠  NLP Risk Engine",
        "📈  Index Rebalancer",
        "🛡  Stress Testing",
        "🔬  Scenario Sandbox"
    ],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown('<div class="section-label">SYSTEM</div>', unsafe_allow_html=True)
st.sidebar.markdown("""
<div style="padding: 4px 0;">
    <div style="display:flex; align-items:center; gap:8px; padding: 7px 0; font-size:14px; color:#5E6470;"><span style="font-size:15px;">⚡</span> Pipeline Status</div>
    <div style="display:flex; align-items:center; gap:8px; padding: 7px 0; font-size:14px; color:#5E6470;"><span style="font-size:15px;">📡</span> Data Sources</div>
    <div style="display:flex; align-items:center; gap:8px; padding: 7px 0; font-size:14px; color:#5E6470;"><span style="font-size:15px;">⚙️</span> Settings</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="padding: 6px 0;">
    <div style="display: flex; align-items: center; gap: 10px;">
        <div style="width:32px; height:32px; border-radius:50%; background: #EEF2FF; display:flex; align-items:center; justify-content:center; font-size:13px; font-weight:600; color:#2563EB;">MG</div>
        <div>
            <div style="font-size:13px; font-weight:600; color:#1A1D26;">Mehul Goyal</div>
            <div style="font-size:11px; color:#9BA1B0;">VIT · Campus Hackathon</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────────────
raw_stream = pipeline.load_unified_stream()
processed_signals = nlp.process_batch(raw_stream)


# ═══════════════════════════════════════════════════════════════════════════════
# VIEW 1: DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
if app_mode == "📊  Dashboard":
    st.markdown('<div class="page-title">Dashboard</div>', unsafe_allow_html=True)
    st.markdown("")
    st.markdown('<div class="page-title" style="font-size:20px;">Risk Intelligence Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Real-time financial risk signals extracted from news and social media feeds via AI/NLP processing.</div>', unsafe_allow_html=True)

    systemic_count = sum(1 for s in processed_signals if s["impact_score"] >= 7)
    avg_sent = float(np.mean([s["sentiment_score"] for s in processed_signals]))
    avg_impact = float(np.mean([s["impact_score"] for s in processed_signals]))

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(metric_card("Total Signals Ingested", str(len(processed_signals)), "Live", "up", "from 2 data sources"), unsafe_allow_html=True)
    with c2:
        d = "up" if avg_sent >= 0 else "down"
        st.markdown(metric_card("Net Market Sentiment", f"{avg_sent:+.2f}", f"{abs(avg_sent)*100:.0f}% polarity", d), unsafe_allow_html=True)
    with c3:
        st.markdown(metric_card("Systemic Risk Triggers", str(systemic_count), f"{systemic_count}/{len(processed_signals)} events", "down" if systemic_count > 0 else "up"), unsafe_allow_html=True)
    with c4:
        st.markdown(metric_card("Avg Impact Score", f"{avg_impact:.1f}", "out of 10", "neutral"), unsafe_allow_html=True)

    # ── Charts ──
    df_signals = pd.DataFrame(processed_signals)

    ch1, ch2 = st.columns([3, 2])
    with ch1:
        st.markdown("""<div class="card"><div class="section-title">Sentiment Distribution<span class="dots">···</span></div></div>""", unsafe_allow_html=True)
        fig_sent = go.Figure()
        colors_bar = ["#16A34A" if v > 0 else "#DC2626" for v in df_signals["sentiment_score"]]
        fig_sent.add_trace(go.Bar(
            x=df_signals["signal_id"],
            y=df_signals["sentiment_score"],
            marker_color=colors_bar,
            marker_cornerradius=4,
            hovertemplate="Signal: %{x}<br>Sentiment: %{y:.2f}<extra></extra>"
        ))
        fig_sent.update_layout(**CLEAN_LAYOUT, showlegend=False, yaxis_title="Sentiment Score", height=320)
        fig_sent.add_hline(y=0, line_width=1, line_color="#E8ECF1")
        st.plotly_chart(fig_sent, use_container_width=True)

    with ch2:
        st.markdown("""<div class="card"><div class="section-title">Event Classification<span class="dots">···</span></div></div>""", unsafe_allow_html=True)
        cat_counts = df_signals["event_classification"].value_counts().reset_index()
        cat_counts.columns = ["Category", "Count"]
        fig_cat = px.pie(
            cat_counts, names="Category", values="Count",
            hole=0.55, color_discrete_sequence=CLEAN_COLORS
        )
        fig_cat.update_layout(**CLEAN_LAYOUT, height=320, showlegend=True,
                              legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5))
        fig_cat.update_traces(textposition="inside", textfont_size=11, marker_line_width=0)
        st.plotly_chart(fig_cat, use_container_width=True)

    # ── Live Feed ──
    st.markdown("")
    st.markdown("""<div class="section-title" style="margin-bottom: 12px;">Signal Feed<span class="dots">···</span></div>""", unsafe_allow_html=True)

    f1, f2 = st.columns(2)
    with f1:
        src_filter = st.multiselect("Source", ["news", "social"], default=["news", "social"], label_visibility="collapsed")
    with f2:
        cat_filter = st.multiselect(
            "Category",
            ["Macroeconomic", "Geopolitical", "Credit Event", "Product Launch", "Merger/Acquisition"],
            default=["Macroeconomic", "Geopolitical", "Credit Event", "Product Launch", "Merger/Acquisition"],
            label_visibility="collapsed"
        )

    filtered = [s for s in processed_signals if s["source_type"] in src_filter and s["event_classification"] in cat_filter]
    for item in filtered:
        st.markdown(feed_item_html(item), unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# VIEW 2: NLP RISK ENGINE
# ═══════════════════════════════════════════════════════════════════════════════
elif app_mode == "🧠  NLP Risk Engine":
    st.markdown('<div class="page-title">Dashboard</div>', unsafe_allow_html=True)
    st.markdown("")
    st.markdown('<div class="page-title" style="font-size:20px;">Core AI/NLP Risk Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Detailed breakdown of NLP extraction outputs — event taxonomy, sentiment-impact correlation, and structured signal table.</div>', unsafe_allow_html=True)

    df_signals = pd.DataFrame(processed_signals)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""<div class="card"><div class="section-title">Event Classification Breakdown<span class="dots">···</span></div></div>""", unsafe_allow_html=True)
        cat_counts = df_signals["event_classification"].value_counts().reset_index()
        cat_counts.columns = ["Event Category", "Frequency"]
        fig = px.bar(cat_counts, x="Event Category", y="Frequency", color="Event Category",
                     color_discrete_sequence=CLEAN_COLORS)
        fig.update_layout(**CLEAN_LAYOUT, showlegend=False, height=340)
        fig.update_traces(marker_cornerradius=6, marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("""<div class="card"><div class="section-title">Sentiment vs Impact Matrix<span class="dots">···</span></div></div>""", unsafe_allow_html=True)
        fig_sc = px.scatter(
            df_signals, x="sentiment_score", y="impact_score",
            color="event_classification", size="impact_score",
            hover_data=["headline", "target_entities"],
            labels={"sentiment_score": "Sentiment (-1 to +1)", "impact_score": "Impact (1-10)"},
            color_discrete_sequence=CLEAN_COLORS
        )
        fig_sc.add_hline(y=7, line_dash="dot", line_color="#DC2626", line_width=1,
                         annotation_text="Systemic ≥ 7", annotation_font=dict(color="#DC2626", size=11))
        fig_sc.update_layout(**CLEAN_LAYOUT, height=340)
        st.plotly_chart(fig_sc, use_container_width=True)

    st.markdown("""<div class="section-title" style="margin-top: 8px;">Structured Signals Table<span class="dots">···</span></div>""", unsafe_allow_html=True)
    st.dataframe(
        df_signals[["signal_id", "source_type", "event_classification", "sentiment_score", "impact_score", "confidence", "target_entities"]],
        use_container_width=True
    )
    st.download_button("⬇ Download Signals (JSON)", data=df_signals.to_json(orient="records", indent=2),
                       file_name="structured_risk_signals.json", mime="application/json")


# ═══════════════════════════════════════════════════════════════════════════════
# VIEW 3: INDEX REBALANCER
# ═══════════════════════════════════════════════════════════════════════════════
elif app_mode == "📈  Index Rebalancer":
    st.markdown('<div class="page-title">Dashboard</div>', unsafe_allow_html=True)
    st.markdown("")
    st.markdown('<div class="page-title" style="font-size:20px;">Tactical Index Rebalancer</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Dynamically adjusts weights across a 12-stock S&P 100 basket using live NLP sentiment scores. Overweights positive momentum, trims negative exposures.</div>', unsafe_allow_html=True)

    tilt_strength = st.slider("Tilt Aggressiveness (λ)", min_value=0.1, max_value=1.5, value=0.65, step=0.05)
    rebalancer.tilt_intensity = tilt_strength

    net_sentiments = rebalancer.aggregate_signals_by_ticker(processed_signals)
    rebal_df = rebalancer.compute_rebalanced_weights(net_sentiments)
    history_df, metrics = rebalancer.run_backtest_simulation(rebal_df)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        d = "up" if metrics['generated_alpha_bps'] > 0 else "down"
        st.markdown(metric_card("Tactical Return", f"{metrics['tactical_total_return_pct']}%",
                                f"{metrics['generated_alpha_bps']} bps alpha", d), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card("Benchmark Return", f"{metrics['benchmark_total_return_pct']}%",
                                "S&P 100 Basket", "neutral"), unsafe_allow_html=True)
    with c3:
        st.markdown(metric_card("Sharpe Ratio", f"{metrics['sharpe_ratio']}",
                                "risk-adjusted", "up" if metrics['sharpe_ratio'] > 1 else "neutral"), unsafe_allow_html=True)
    with c4:
        st.markdown(metric_card("Annualized Vol", f"{metrics['tactical_annualized_vol']}%",
                                "portfolio σ", "neutral"), unsafe_allow_html=True)

    w_col, p_col = st.columns(2)
    with w_col:
        st.markdown("""<div class="card"><div class="section-title">Weight Allocation<span class="dots">···</span></div></div>""", unsafe_allow_html=True)
        fig_w = go.Figure()
        fig_w.add_trace(go.Bar(name='Benchmark', x=rebal_df['ticker'], y=rebal_df['benchmark_weight']*100,
                               marker_color='#E8ECF1', marker_cornerradius=4))
        fig_w.add_trace(go.Bar(name='NLP-Tilted', x=rebal_df['ticker'], y=rebal_df['rebalanced_weight']*100,
                               marker_color='#2563EB', marker_cornerradius=4))
        fig_w.update_layout(barmode='group', yaxis_title="Allocation (%)", height=360, **CLEAN_LAYOUT)
        st.plotly_chart(fig_w, use_container_width=True)

    with p_col:
        st.markdown("""<div class="card"><div class="section-title">Cumulative Performance<span class="dots">···</span></div></div>""", unsafe_allow_html=True)
        fig_p = go.Figure()
        fig_p.add_trace(go.Scatter(x=history_df.index, y=history_df["Benchmark_Index"],
                                   name="Benchmark", line=dict(color="#D1D5DB", width=2, dash="dot")))
        fig_p.add_trace(go.Scatter(x=history_df.index, y=history_df["Tactical_NLP_Index"],
                                   name="NLP Tactical", line=dict(color="#2563EB", width=2.5),
                                   fill="tonexty", fillcolor="rgba(37,99,235,0.06)"))
        fig_p.update_layout(yaxis_title="Index (Base 100)", height=360, **CLEAN_LAYOUT)
        st.plotly_chart(fig_p, use_container_width=True)

    st.markdown("""<div class="section-title">Rebalancing Ledger<span class="dots">···</span></div>""", unsafe_allow_html=True)
    st.dataframe(
        rebal_df[["ticker", "name", "sector", "net_sentiment", "benchmark_weight", "rebalanced_weight", "weight_delta", "allocation_status"]],
        use_container_width=True
    )


# ═══════════════════════════════════════════════════════════════════════════════
# VIEW 4: STRESS TESTING
# ═══════════════════════════════════════════════════════════════════════════════
elif app_mode == "🛡  Stress Testing":
    st.markdown('<div class="page-title">Dashboard</div>', unsafe_allow_html=True)
    st.markdown("")
    st.markdown('<div class="page-title" style="font-size:20px;">Portfolio Stress Testing</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Subscribes to Event Classification and Impact Score. When systemic disruption is detected (Impact ≥ 7), macro/credit shocks are applied to a $505M wholesale banking portfolio.</div>', unsafe_allow_html=True)

    systemic_events = [s for s in processed_signals if s["impact_score"] >= 7]
    event_options = [f"{s['event_classification']}  ·  Impact {s['impact_score']}  ·  {s['headline'][:45]}..." for s in systemic_events]
    event_options.append("Custom Scenario...")

    chosen_event_str = st.selectbox("Select Stress Trigger", event_options)

    if "Custom Scenario" in chosen_event_str:
        cc1, cc2 = st.columns(2)
        with cc1:
            sel_cat = st.selectbox("Category", ["Geopolitical", "Macroeconomic", "Credit Event"])
        with cc2:
            sel_impact = st.slider("Impact Score", 7, 10, 8)
    else:
        idx = event_options.index(chosen_event_str)
        sel_event = systemic_events[idx]
        sel_cat = sel_event["event_classification"]
        sel_impact = sel_event["impact_score"]

    stress_results = stress_tester.run_stress_test(sel_cat, sel_impact)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(metric_card("Pre-Stress Value", f"${stress_results['total_pre_stress_mv']/1e6:.1f}M",
                                "Total Portfolio", "neutral"), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card("Post-Stress Value", f"${stress_results['total_post_stress_mv']/1e6:.1f}M",
                                f"-{stress_results['portfolio_drawdown_pct']}%", "down"), unsafe_allow_html=True)
    with c3:
        st.markdown(metric_card("Expected Loss (EL)", f"${stress_results['post_stress_expected_loss']/1e6:.2f}M",
                                f"+{stress_results['expected_loss_surge_pct']}% surge", "down"), unsafe_allow_html=True)
    with c4:
        st.markdown(metric_card("Shock Multiplier", f"{stress_results['scaling_factor']}x",
                                sel_cat, "neutral"), unsafe_allow_html=True)

    # Insight box
    st.markdown(f"""
    <div class="insight-box">
        <div class="insight-header">
            <span>⚡</span> Active Scenario: {stress_results['scenario_name']}
        </div>
        <div class="insight-text">
            Impact Score <strong>{sel_impact}/10</strong> · Severity Multiplier <strong>{stress_results['scaling_factor']}x</strong> · Category: <strong>{sel_cat}</strong><br>
            Portfolio drawdown of <strong style="color: var(--red);">-${stress_results['total_portfolio_loss_usd']/1e6:.1f}M</strong> across loans, bonds, and derivatives.
        </div>
    </div>
    """, unsafe_allow_html=True)

    detailed_df = stress_results["detailed_asset_impact"]
    col_pie, col_bar = st.columns(2)

    with col_pie:
        st.markdown("""<div class="card"><div class="section-title">Loss by Asset Class<span class="dots">···</span></div></div>""", unsafe_allow_html=True)
        loss_by_type = detailed_df.groupby("asset_type")["valuation_loss_usd"].sum().reset_index()
        fig_d = px.pie(loss_by_type, names="asset_type", values="valuation_loss_usd",
                       hole=0.5, color_discrete_sequence=["#DC2626", "#EA580C", "#7C3AED", "#0891B2"])
        fig_d.update_layout(**CLEAN_LAYOUT, height=340,
                            legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5))
        fig_d.update_traces(textposition="inside", textfont_size=11, marker_line_width=0)
        st.plotly_chart(fig_d, use_container_width=True)

    with col_bar:
        st.markdown("""<div class="card"><div class="section-title">Haircut by Facility<span class="dots">···</span></div></div>""", unsafe_allow_html=True)
        fig_b = px.bar(
            detailed_df.sort_values(by="loss_pct", ascending=True),
            y="facility_name", x="loss_pct", color="borrower_sector",
            orientation="h", labels={"loss_pct": "Haircut (%)", "facility_name": ""},
            color_discrete_sequence=CLEAN_COLORS
        )
        fig_b.update_layout(**CLEAN_LAYOUT, height=340)
        fig_b.update_traces(marker_cornerradius=3)
        st.plotly_chart(fig_b, use_container_width=True)

    st.markdown("""<div class="section-title">Basel Credit Risk Ledger<span class="dots">···</span></div>""", unsafe_allow_html=True)
    st.dataframe(
        detailed_df[["asset_id", "facility_name", "asset_type", "borrower_sector", "notional_usd",
                      "baseline_pd_pct", "stressed_pd_pct", "baseline_mv_usd", "stressed_mv_usd",
                      "valuation_loss_usd", "loss_pct"]],
        use_container_width=True
    )


# ═══════════════════════════════════════════════════════════════════════════════
# VIEW 5: SANDBOX
# ═══════════════════════════════════════════════════════════════════════════════
elif app_mode == "🔬  Scenario Sandbox":
    st.markdown('<div class="page-title">Dashboard</div>', unsafe_allow_html=True)
    st.markdown("")
    st.markdown('<div class="page-title" style="font-size:20px;">Ad-Hoc Scenario Sandbox</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Paste any breaking financial headline, earnings release, or social post. The full NLP pipeline will extract structured signals and trigger downstream actions in real-time.</div>', unsafe_allow_html=True)

    custom_input = st.text_area(
        "Enter breaking event text:",
        value="Middle Eastern export terminal faces maritime drone disruption; crude futures surge $12 while global sovereign debt spreads widen by 85bps as inflation fears mount.",
        height=100
    )

    if st.button("Run AI Risk Analysis", type="primary", use_container_width=True):
        record = {
            "id": f"LIVE-{int(datetime.utcnow().timestamp())}",
            "source_type": "adhoc_input",
            "channel": "Live Jury Tester",
            "raw_text": custom_input
        }
        res = nlp.process_record(record)

        st.markdown("")

        r1, r2, r3, r4 = st.columns(4)
        with r1:
            d = "up" if res["sentiment_score"] > 0 else "down"
            st.markdown(metric_card("Sentiment Score", f"{res['sentiment_score']:+.2f}",
                                    "positive" if d == "up" else "negative", d), unsafe_allow_html=True)
        with r2:
            st.markdown(metric_card("Event Category", res['event_classification'],
                                    f"{res['confidence']*100:.0f}% confidence", "neutral"), unsafe_allow_html=True)
        with r3:
            st.markdown(metric_card("Impact Score", f"{res['impact_score']}/10",
                                    "systemic" if res['impact_score'] >= 7 else "standard",
                                    "down" if res['impact_score'] >= 7 else "up"), unsafe_allow_html=True)
        with r4:
            if res["impact_score"] >= 7:
                st.markdown(metric_card("Systemic Alert", "YES",
                                        "stress test triggered", "down"), unsafe_allow_html=True)
            else:
                st.markdown(metric_card("Systemic Alert", "NO",
                                        "below threshold", "up"), unsafe_allow_html=True)

        st.markdown("")

        t1, t2 = st.columns(2)
        with t1:
            entities_str = ", ".join(res["target_entities"]) if res["target_entities"] else "No specific tickers"
            if res["target_entities"]:
                action_text = f"Identified: <strong>{entities_str}</strong>. Reallocating weights with net sentiment <strong>{res['sentiment_score']:+.2f}</strong>."
            else:
                action_text = "No specific ticker identified. General macro sentiment shift applied across entire index universe."
            st.markdown(f"""
            <div class="insight-box">
                <div class="insight-header"><span>📈</span> Module A: Tactical Rebalancer</div>
                <div class="insight-text">{action_text}</div>
            </div>
            """, unsafe_allow_html=True)

        with t2:
            if res["impact_score"] >= 7:
                action_text = f'<strong style="color: var(--red);">SYSTEMIC STRESS TRIGGERED!</strong> Shock parameters queued for <strong>{res["event_classification"]}</strong>. $505M portfolio under evaluation.'
            else:
                action_text = "Impact below systemic threshold (&lt; 7). No emergency credit provisions required."
            st.markdown(f"""
            <div class="insight-box">
                <div class="insight-header"><span>🛡</span> Module B: Stress Tester</div>
                <div class="insight-text">{action_text}</div>
            </div>
            """, unsafe_allow_html=True)
