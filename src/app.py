"""
AI/NLP Financial Risk & Intelligence Platform
S&P Global & CRISIL Campus Hackathon 2026
Candidate: Mehul Goyal | Campus: VIT | Email: mehul.goyal2023@vitstudent.ac.in

UI: Clean Institutional Corporate Dashboard (Novera / Linear Design System)
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

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="S&P Global & CRISIL | AI/NLP Risk Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# CLEAN CORPORATE CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

:root {
    --bg: #F8FAFC;
    --card-bg: #FFFFFF;
    --border: #E2E8F0;
    --border-subtle: #F1F5F9;
    --text-primary: #0F172A;
    --text-secondary: #475569;
    --text-muted: #94A3B8;
    --accent: #2563EB;
    --accent-light: #EFF6FF;
    --green: #16A34A;
    --green-bg: #DCFCE7;
    --red: #DC2626;
    --red-bg: #FEE2E2;
    --orange: #EA580C;
    --orange-bg: #FFEDD5;
    --radius: 12px;
    --radius-sm: 8px;
    --shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05), 0 1px 2px -1px rgba(0, 0, 0, 0.05);
    --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
}

/* Global resets */
html, body, [data-testid="stAppViewContainer"], .stApp {
    background-color: var(--bg) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    color: var(--text-primary) !important;
}

[data-testid="stHeader"] {
    background: #FFFFFF !important;
    border-bottom: 1px solid var(--border) !important;
    height: 56px !important;
}

[data-testid="stMainBlockContainer"] {
    padding-top: 1.5rem !important;
    padding-bottom: 3rem !important;
    max-width: 1440px;
}

/* Top App Header Banner */
.top-brand-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 0 20px 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 20px;
}
.brand-left {
    display: flex;
    align-items: center;
    gap: 14px;
}
.brand-badge {
    background: #1E293B;
    color: #FFFFFF;
    font-weight: 800;
    font-size: 13px;
    padding: 6px 12px;
    border-radius: 8px;
    letter-spacing: 0.5px;
}
.brand-title {
    font-size: 18px;
    font-weight: 700;
    color: var(--text-primary);
    line-height: 1.2;
}
.brand-subtitle {
    font-size: 12px;
    color: var(--text-muted);
}
.candidate-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: #FFFFFF;
    border: 1px solid var(--border);
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    color: var(--text-secondary);
    box-shadow: var(--shadow);
}
.candidate-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--green);
}

/* Card components */
.metric-card-wrapper {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 20px;
    box-shadow: var(--shadow);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
    margin-bottom: 16px;
    height: 100%;
}
.metric-card-wrapper:hover {
    box-shadow: var(--shadow-md);
    border-color: #CBD5E1;
}
.metric-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 10px;
}
.metric-title {
    font-size: 12.5px;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.4px;
}
.dots {
    color: var(--text-muted);
    font-size: 16px;
    cursor: default;
}
.metric-val-row {
    display: flex;
    align-items: baseline;
    gap: 10px;
    margin-bottom: 6px;
}
.metric-val {
    font-size: 32px;
    font-weight: 800;
    color: var(--text-primary);
    line-height: 1.1;
    letter-spacing: -0.5px;
}
.delta-badge {
    display: inline-flex;
    align-items: center;
    gap: 3px;
    font-size: 11.5px;
    font-weight: 700;
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
    color: var(--text-secondary);
    background: var(--border-subtle);
}
.metric-sub {
    font-size: 12.5px;
    color: var(--text-muted);
    font-weight: 400;
}

/* Seamless Container Card Wrapper for Plotly charts */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--card-bg) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 20px 20px 8px 20px !important;
    box-shadow: var(--shadow) !important;
    margin-bottom: 18px !important;
}

.chart-card-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 14px;
    padding-bottom: 10px;
    border-bottom: 1px solid var(--border-subtle);
}
.chart-card-title {
    font-size: 15px;
    font-weight: 700;
    color: var(--text-primary);
    display: flex;
    align-items: center;
    gap: 8px;
}
.chart-card-badge {
    font-size: 11px;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 4px;
    background: var(--accent-light);
    color: var(--accent);
}

/* Page titles */
.page-title {
    font-size: 24px;
    font-weight: 800;
    color: var(--text-primary);
    letter-spacing: -0.4px;
    margin-bottom: 4px;
}
.page-subtitle {
    font-size: 14px;
    color: var(--text-secondary);
    line-height: 1.5;
    margin-bottom: 22px;
}

/* Feed items */
.feed-row {
    background: #FFFFFF;
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 14px 18px;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 16px;
    box-shadow: var(--shadow);
    transition: all 0.15s ease;
}
.feed-row:hover {
    border-color: #CBD5E1;
    box-shadow: var(--shadow-md);
}
.feed-row.systemic {
    border-left: 4px solid var(--red);
}
.feed-row.standard {
    border-left: 4px solid var(--green);
}
.feed-icon {
    width: 38px;
    height: 38px;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    flex-shrink: 0;
}
.feed-icon.news { background: var(--accent-light); }
.feed-icon.social { background: var(--orange-bg); }
.feed-body { flex: 1; min-width: 0; }
.feed-headline {
    font-size: 14px;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 3px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.feed-snippet {
    font-size: 12.5px;
    color: var(--text-secondary);
    line-height: 1.4;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
}
.feed-meta {
    display: flex;
    gap: 6px;
    align-items: center;
    margin-top: 6px;
    flex-wrap: wrap;
}
.pill {
    display: inline-block;
    padding: 3px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 600;
}
.pill-danger { background: var(--red-bg); color: var(--red); }
.pill-success { background: var(--green-bg); color: var(--green); }
.pill-info { background: var(--accent-light); color: var(--accent); }
.pill-warning { background: var(--orange-bg); color: var(--orange); }
.pill-muted { background: var(--border-subtle); color: var(--text-secondary); border: 1px solid var(--border); }

.feed-stats {
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    gap: 2px;
    flex-shrink: 0;
    min-width: 80px;
}
.feed-stat-value {
    font-size: 16px;
    font-weight: 700;
}
.feed-stat-label {
    font-size: 10.5px;
    color: var(--text-muted);
    font-weight: 500;
    text-transform: uppercase;
}

/* Insight Box */
.insight-box {
    background: #FFFFFF;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 18px 22px;
    margin-bottom: 18px;
    box-shadow: var(--shadow);
}
.insight-header {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 14px;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 6px;
}
.insight-text {
    font-size: 13px;
    color: var(--text-secondary);
    line-height: 1.5;
}

/* ═══════ GUARANTEED PINNED PERMANENT SIDEBAR ═══════ */
[data-testid="stSidebar"] {
    display: block !important;
    transform: none !important;
    min-width: 275px !important;
    max-width: 275px !important;
    width: 275px !important;
    position: fixed !important;
    top: 0 !important;
    left: 0 !important;
    height: 100vh !important;
    z-index: 9999 !important;
    background: #FFFFFF !important;
    border-right: 1px solid var(--border) !important;
    box-shadow: 1px 0 4px rgba(0,0,0,0.03) !important;
    visibility: visible !important;
}

/* Hide collapse and expand toggle chevrons */
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapsedControl"],
button[data-testid="stSidebarCollapseButton"],
[data-testid="collapsedControl"] {
    display: none !important;
    visibility: hidden !important;
}

/* Push the main container so it never overlaps the pinned 275px sidebar */
[data-testid="stAppViewContainer"] {
    margin-left: 275px !important;
    width: calc(100% - 275px) !important;
}

[data-testid="stHeader"] {
    left: 275px !important;
    width: calc(100% - 275px) !important;
}

/* Sidebar Radio Pills - Clean SaaS Design */
[data-testid="stSidebar"] .stRadio > div {
    gap: 4px !important;
}

[data-testid="stSidebar"] .stRadio > div > label {
    display: flex !important;
    align-items: center !important;
    padding: 10px 14px !important;
    border-radius: 8px !important;
    background: transparent !important;
    color: #475569 !important;
    font-weight: 500 !important;
    font-size: 13.5px !important;
    cursor: pointer !important;
    transition: all 0.15s ease !important;
    border: 1px solid transparent !important;
}

/* Hide radio circle bullet */
[data-testid="stSidebar"] .stRadio > div > label > div:first-child {
    display: none !important;
}

/* Hover on item */
[data-testid="stSidebar"] .stRadio > div > label:hover {
    background: #F8FAFC !important;
    color: #0F172A !important;
    border-color: #E2E8F0 !important;
}

/* Active item */
[data-testid="stSidebar"] .stRadio > div > label:has(input:checked),
[data-testid="stSidebar"] .stRadio > div > label[data-checked="true"] {
    background: #EFF6FF !important;
    border-color: #DBEAFE !important;
}

[data-testid="stSidebar"] .stRadio > div > label:has(input:checked) p,
[data-testid="stSidebar"] .stRadio > div > label[data-checked="true"] p {
    color: #2563EB !important;
    font-weight: 700 !important;
}

/* Sidebar markdown text */
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] span {
    font-family: 'Inter', sans-serif !important;
    font-size: 13.5px !important;
    color: var(--text-secondary) !important;
}

/* Streamlit Button & Overrides */
.stButton > button {
    background: #FFFFFF !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    box-shadow: var(--shadow) !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 13.5px !important;
    padding: 8px 18px !important;
    transition: all 0.15s ease !important;
}
.stButton > button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    box-shadow: var(--shadow-md) !important;
}
.stButton > button[kind="primary"],
.stButton > button[data-testid="stBaseButton-primary"] {
    background: var(--accent) !important;
    color: #FFFFFF !important;
    border: none !important;
}
.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="stBaseButton-primary"]:hover {
    background: #1D4ED8 !important;
    color: #FFFFFF !important;
}

[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    background: #FFFFFF !important;
}

/* Custom Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: #CBD5E1; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #94A3B8; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY THEME CONFIGURATION (WITHOUT title_font TO PREVENT "undefined" TEXT)
# ─────────────────────────────────────────────────────────────────────────────
def apply_clean_layout(fig, height=320, showlegend=False, legend_override=None, **kwargs):
    leg = dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#475569", size=11))
    if legend_override:
        leg.update(legend_override)
    base = dict(
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0)",
        font=dict(family="Inter, sans-serif", color="#0F172A", size=12),
        xaxis=dict(gridcolor="#F1F5F9", zerolinecolor="#E2E8F0", linecolor="#E2E8F0"),
        yaxis=dict(gridcolor="#F1F5F9", zerolinecolor="#E2E8F0", linecolor="#E2E8F0"),
        margin=dict(l=40, r=20, t=20, b=40),
        height=height,
        showlegend=showlegend,
        legend=leg
    )
    base.update(kwargs)
    fig.update_layout(**base)
    return fig

CLEAN_COLORS = ["#2563EB", "#16A34A", "#EA580C", "#DC2626", "#7C3AED", "#0891B2", "#CA8A04"]

# ─────────────────────────────────────────────────────────────────────────────
# METRIC CARD HELPER
# ─────────────────────────────────────────────────────────────────────────────
def metric_card(title, value, delta=None, delta_dir="up", subtitle=""):
    delta_html = ""
    if delta:
        cls = "delta-up" if delta_dir == "up" else ("delta-down" if delta_dir == "down" else "delta-neutral")
        arrow = "↑" if delta_dir == "up" else ("↓" if delta_dir == "down" else "")
        delta_html = f'<span class="delta-badge {cls}">{arrow} {delta}</span>'

    sub_html = f'<div class="metric-sub">{subtitle}</div>' if subtitle else ""
    return f"""
    <div class="metric-card-wrapper">
        <div class="metric-header">
            <span class="metric-title">{title}</span>
            <span class="dots">···</span>
        </div>
        <div class="metric-val-row">
            <span class="metric-val">{value}</span>
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
        pills += '<span class="pill pill-danger">Systemic Shock</span> '
    else:
        pills += '<span class="pill pill-success">Standard Event</span> '
    pills += f'<span class="pill pill-info">{item["event_classification"]}</span> '
    for e in item.get("target_entities", [])[:3]:
        pills += f'<span class="pill pill-muted">{e}</span> '

    sent = item["sentiment_score"]
    sent_color = "#16A34A" if sent > 0 else "#DC2626"

    return f"""
    <div class="{row_cls}">
        <div class="{icon_cls}">{icon_emoji}</div>
        <div class="feed-body">
            <div class="feed-headline">{item.get('headline') or item['text_snippet'][:75]}</div>
            <div class="feed-snippet">{item['text_snippet'][:150]}</div>
            <div class="feed-meta">{pills}</div>
        </div>
        <div class="feed-stats">
            <div class="feed-stat-value" style="color: {sent_color};">{sent:+.2f}</div>
            <div class="feed-stat-label">Sentiment</div>
            <div class="feed-stat-value" style="color: #0F172A;">{item['impact_score']}<span style="font-size:11px; color:#94A3B8;">/10</span></div>
            <div class="feed-stat-label">Impact</div>
        </div>
    </div>
    """

# ─────────────────────────────────────────────────────────────────────────────
# LOAD CORE ENGINES
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_components():
    return IngestionPipeline(), NLPRiskEngine(), TacticalIndexRebalancer(), StrategicStressTester()

pipeline, nlp, rebalancer, stress_tester = load_components()
raw_stream = pipeline.load_unified_stream()
processed_signals = nlp.process_batch(raw_stream)

# ─────────────────────────────────────────────────────────────────────────────
# TOP APP HEADER BANNER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="top-brand-bar">
    <div class="brand-left">
        <span class="brand-badge">S&P GLOBAL · CRISIL</span>
        <div>
            <div class="brand-title">Autonomous AI/NLP Financial Risk Engine</div>
            <div class="brand-subtitle">Real-Time Multi-Feed Ingestion, Tactical Alpha Rebalancing & Wholesale Stress Testing</div>
        </div>
    </div>
    <div class="candidate-badge">
        <span class="candidate-dot"></span>
        <span>Mehul Goyal · VIT · Hackathon 2026</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────────────────────────────────────
nav_options = [
    "📊  Dashboard",
    "🧠  NLP Risk Engine",
    "📈  Index Rebalancer",
    "🛡  Stress Testing",
    "🔬  Scenario Sandbox"
]

st.sidebar.markdown("""
<div style="padding: 10px 0 16px 0;">
    <div style="display: flex; align-items: center; gap: 10px;">
        <div style="width: 38px; height: 38px; background: #2563EB; border-radius: 8px; display: flex; align-items: center; justify-content: center; color: white; font-weight: 800; font-size: 15px;">S&P</div>
        <div>
            <div style="font-size: 15px; font-weight: 700; color: #0F172A; line-height: 1.2;">Risk Engine</div>
            <div style="font-size: 11px; color: #94A3B8;">Code to Connect 2026</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown('<div style="font-size: 11px; font-weight: 700; color: #94A3B8; letter-spacing: 0.5px; margin-bottom: 6px;">MAIN MENU</div>', unsafe_allow_html=True)
app_mode = st.sidebar.radio(
    "Navigation",
    nav_options,
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown('<div style="font-size: 11px; font-weight: 700; color: #94A3B8; letter-spacing: 0.5px; margin-bottom: 6px;">SYSTEM PIPELINE</div>', unsafe_allow_html=True)
st.sidebar.markdown("""
<div style="font-size: 12.5px; color: #475569; line-height: 1.8;">
    <div>⚡ <strong>Data Ingestion:</strong> GDELT & Tweets</div>
    <div>🧠 <strong>Core NLP:</strong> FinBERT & Taxonomy</div>
    <div>📈 <strong>Module A:</strong> 12-Stock S&P 100 Tilt</div>
    <div>🛡 <strong>Module B:</strong> $505M Wholesale ECL</div>
    <div>📡 <strong>FastAPI Service:</strong> Port 8000</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="padding: 4px 0;">
    <div style="font-size: 11px; font-weight: 700; color: #94A3B8; letter-spacing: 0.5px; margin-bottom: 4px;">CANDIDATE SUBMISSION</div>
    <div style="font-size: 13.5px; font-weight: 700; color: #0F172A;">Mehul Goyal</div>
    <div style="font-size: 12px; color: #64748B;">Vellore Institute of Technology (VIT)</div>
    <div style="font-size: 11px; color: #2563EB; word-break: break-all;">mehul.goyal2023@vitstudent.ac.in</div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# VIEW 1: DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
if app_mode == "📊  Dashboard":
    st.markdown('<div class="page-title">Risk Intelligence Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Real-time financial risk signals extracted from news and social media feeds via AI/NLP processing.</div>', unsafe_allow_html=True)

    systemic_count = sum(1 for s in processed_signals if s["impact_score"] >= 7)
    avg_sent = float(np.mean([s["sentiment_score"] for s in processed_signals]))
    avg_impact = float(np.mean([s["impact_score"] for s in processed_signals]))

    # Top KPI Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(metric_card("Total Signals Ingested", str(len(processed_signals)), "Live", "up", "from 2 data sources"), unsafe_allow_html=True)
    with c2:
        d = "up" if avg_sent >= 0 else "down"
        st.markdown(metric_card("Net Market Sentiment", f"{avg_sent:+.2f}", f"{abs(avg_sent)*100:.0f}% polarity", d, "continuous [-1.0, 1.0]"), unsafe_allow_html=True)
    with c3:
        st.markdown(metric_card("Systemic Risk Triggers", str(systemic_count), f"{systemic_count}/{len(processed_signals)} events", "down" if systemic_count > 0 else "up", "impact severity ≥ 7/10"), unsafe_allow_html=True)
    with c4:
        st.markdown(metric_card("Avg Impact Score", f"{avg_impact:.1f}", "out of 10", "neutral", "normalized severity"), unsafe_allow_html=True)

    df_signals = pd.DataFrame(processed_signals)

    # Charts in Seamless Border Containers
    ch1, ch2 = st.columns([3, 2])
    with ch1:
        with st.container(border=True):
            st.markdown("""
            <div class="chart-card-header">
                <span class="chart-card-title">Sentiment Distribution Across Ingested Signals</span>
                <span class="chart-card-badge">Polarity Score</span>
            </div>
            """, unsafe_allow_html=True)
            fig_sent = go.Figure()
            colors_bar = ["#16A34A" if v > 0 else "#DC2626" for v in df_signals["sentiment_score"]]
            fig_sent.add_trace(go.Bar(
                x=df_signals["signal_id"],
                y=df_signals["sentiment_score"],
                marker_color=colors_bar,
                marker_cornerradius=4,
                hovertemplate="Signal ID: %{x}<br>Sentiment: %{y:+.2f}<extra></extra>"
            ))
            apply_clean_layout(fig_sent, height=300, showlegend=False, yaxis_title="Sentiment Score [-1.0, +1.0]")
            fig_sent.add_hline(y=0, line_width=1, line_color="#E2E8F0")
            st.plotly_chart(fig_sent, width="stretch")

    with ch2:
        with st.container(border=True):
            st.markdown("""
            <div class="chart-card-header">
                <span class="chart-card-title">Event Taxonomy Breakdown</span>
                <span class="chart-card-badge">5 Categories</span>
            </div>
            """, unsafe_allow_html=True)
            cat_counts = df_signals["event_classification"].value_counts().reset_index()
            cat_counts.columns = ["Category", "Count"]
            fig_cat = px.pie(
                cat_counts, names="Category", values="Count",
                hole=0.55, color_discrete_sequence=CLEAN_COLORS
            )
            apply_clean_layout(
                fig_cat,
                height=300,
                showlegend=True,
                legend_override=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5)
            )
            fig_cat.update_traces(textposition="inside", textfont_size=11, marker_line_width=0)
            st.plotly_chart(fig_cat, width="stretch")

    # Real-Time Signal Feed
    st.markdown('<div class="page-title" style="font-size: 18px; margin-top: 10px;">Live Ingested Signals Feed</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle" style="margin-bottom: 12px;">Filter signals by source channel and risk taxonomy.</div>', unsafe_allow_html=True)

    f1, f2 = st.columns([1, 2])
    with f1:
        src_filter = st.multiselect("Source Filter", ["news", "social"], default=["news", "social"], label_visibility="collapsed")
    with f2:
        cat_filter = st.multiselect(
            "Category Filter",
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
    st.markdown('<div class="page-title">Core AI/NLP Risk Engine Processing</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Deep inspection of structured output attributes: Sentiment Score, Event Classification taxonomy, Impact Severity (1-10), and Target Entity resolution.</div>', unsafe_allow_html=True)

    df_signals = pd.DataFrame(processed_signals)

    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown("""
            <div class="chart-card-header">
                <span class="chart-card-title">Event Category Frequencies</span>
                <span class="chart-card-badge">Distribution</span>
            </div>
            """, unsafe_allow_html=True)
            cat_counts = df_signals["event_classification"].value_counts().reset_index()
            cat_counts.columns = ["Event Category", "Frequency"]
            fig_bar = px.bar(
                cat_counts, x="Event Category", y="Frequency", color="Event Category",
                color_discrete_sequence=CLEAN_COLORS
            )
            apply_clean_layout(fig_bar, height=320, showlegend=False)
            fig_bar.update_traces(marker_cornerradius=5, marker_line_width=0)
            st.plotly_chart(fig_bar, width="stretch")

    with col2:
        with st.container(border=True):
            st.markdown("""
            <div class="chart-card-header">
                <span class="chart-card-title">Sentiment vs. Impact Scatter Matrix</span>
                <span class="chart-card-badge">Systemic Quadrant</span>
            </div>
            """, unsafe_allow_html=True)
            fig_sc = px.scatter(
                df_signals, x="sentiment_score", y="impact_score",
                color="event_classification", size="impact_score",
                hover_data=["headline", "target_entities"],
                labels={"sentiment_score": "Sentiment (-1 to +1)", "impact_score": "Impact (1-10)"},
                color_discrete_sequence=CLEAN_COLORS
            )
            fig_sc.add_hline(
                y=7, line_dash="dot", line_color="#DC2626", line_width=1.5,
                annotation_text="Systemic Threshold ≥ 7", annotation_font=dict(color="#DC2626", size=11)
            )
            apply_clean_layout(fig_sc, height=320, showlegend=True)
            st.plotly_chart(fig_sc, width="stretch")

    with st.container(border=True):
        st.markdown("""
        <div class="chart-card-header">
            <span class="chart-card-title">Structured Signals Output Table</span>
            <span class="chart-card-badge">Machine-Readable Schema</span>
        </div>
        """, unsafe_allow_html=True)
        st.dataframe(
            df_signals[["signal_id", "source_type", "event_classification", "sentiment_score", "impact_score", "confidence", "target_entities"]],
            use_container_width=True
        )
        st.download_button(
            "⬇ Download Structured Signals JSON",
            data=df_signals.to_json(orient="records", indent=2),
            file_name="structured_risk_signals.json",
            mime="application/json"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# VIEW 3: INDEX REBALANCER (MODULE A)
# ═══════════════════════════════════════════════════════════════════════════════
elif app_mode == "📈  Index Rebalancer":
    st.markdown('<div class="page-title">Module A: Tactical Index Rebalancer</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Subscribes to live sentiment scores to dynamically tilt a 12-stock S&P 100 constituent basket. Long-only floor (2%), single-stock cap (22%), exact 100% sum constraint.</div>', unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown('<div style="font-weight: 700; font-size: 14px; margin-bottom: 6px;">Tilt Intensity Tuning (λ)</div>', unsafe_allow_html=True)
        tilt_strength = st.slider(
            "Adjust sensitivity to sentiment signals:",
            min_value=0.1, max_value=1.5, value=0.65, step=0.05
        )
        rebalancer.tilt_intensity = tilt_strength

    net_sentiments = rebalancer.aggregate_signals_by_ticker(processed_signals)
    rebal_df = rebalancer.compute_rebalanced_weights(net_sentiments)
    history_df, metrics = rebalancer.run_backtest_simulation(rebal_df)

    # Rebalance KPIs
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        d = "up" if metrics['generated_alpha_bps'] > 0 else "down"
        st.markdown(metric_card("Tactical Portfolio Return", f"{metrics['tactical_total_return_pct']}%", f"{metrics['generated_alpha_bps']} bps alpha", d, "30-day simulated backtest"), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card("Benchmark Return", f"{metrics['benchmark_total_return_pct']}%", "Base Index", "neutral", "S&P 100 12-Stock Basket"), unsafe_allow_html=True)
    with c3:
        st.markdown(metric_card("Tactical Sharpe Ratio", f"{metrics['sharpe_ratio']}", "Risk-Adjusted", "up" if metrics['sharpe_ratio'] > 1 else "neutral", "rf = 4.5% annual"), unsafe_allow_html=True)
    with c4:
        st.markdown(metric_card("Annualized Volatility", f"{metrics['tactical_annualized_vol']}%", "Controlled Risk", "neutral", "portfolio σ"), unsafe_allow_html=True)

    w_col, p_col = st.columns(2)
    with w_col:
        with st.container(border=True):
            st.markdown("""
            <div class="chart-card-header">
                <span class="chart-card-title">Weight Allocation: Benchmark vs. NLP-Tilted</span>
                <span class="chart-card-badge">Box Constrained [2%, 22%]</span>
            </div>
            """, unsafe_allow_html=True)
            fig_w = go.Figure()
            fig_w.add_trace(go.Bar(
                name='Benchmark', x=rebal_df['ticker'], y=rebal_df['benchmark_weight']*100,
                marker_color='#CBD5E1', marker_cornerradius=4
            ))
            fig_w.add_trace(go.Bar(
                name='NLP Tactical', x=rebal_df['ticker'], y=rebal_df['rebalanced_weight']*100,
                marker_color='#2563EB', marker_cornerradius=4
            ))
            apply_clean_layout(
                fig_w,
                height=320,
                showlegend=True,
                barmode='group',
                yaxis_title="Weight Allocation (%)"
            )
            st.plotly_chart(fig_w, width="stretch")

    with p_col:
        with st.container(border=True):
            st.markdown("""
            <div class="chart-card-header">
                <span class="chart-card-title">Cumulative Performance (Base 100)</span>
                <span class="chart-card-badge">Alpha Spread</span>
            </div>
            """, unsafe_allow_html=True)
            fig_p = go.Figure()
            fig_p.add_trace(go.Scatter(
                x=history_df.index, y=history_df["Benchmark_Index"],
                name="Benchmark", line=dict(color="#94A3B8", width=2, dash="dot")
            ))
            fig_p.add_trace(go.Scatter(
                x=history_df.index, y=history_df["Tactical_NLP_Index"],
                name="NLP Tactical", line=dict(color="#2563EB", width=2.5),
                fill="tonexty", fillcolor="rgba(37,99,235,0.06)"
            ))
            apply_clean_layout(
                fig_p,
                height=320,
                showlegend=True,
                yaxis_title="Normalized Value"
            )
            st.plotly_chart(fig_p, width="stretch")

    with st.container(border=True):
        st.markdown("""
        <div class="chart-card-header">
            <span class="chart-card-title">Constituent Rebalancing Ledger</span>
            <span class="chart-card-badge">12 Tickers</span>
        </div>
        """, unsafe_allow_html=True)
        st.dataframe(
            rebal_df[["ticker", "name", "sector", "net_sentiment", "benchmark_weight", "rebalanced_weight", "weight_delta", "allocation_status"]],
            use_container_width=True
        )


# ═══════════════════════════════════════════════════════════════════════════════
# VIEW 4: STRESS TESTING (MODULE B)
# ═══════════════════════════════════════════════════════════════════════════════
elif app_mode == "🛡  Stress Testing":
    st.markdown('<div class="page-title">Module B: Wholesale Portfolio Stress Testing</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Monitors Event Classification and Impact Score. When systemic disruption is detected (Impact ≥ 7), macro/credit shocks are applied to a $505M synthetic wholesale banking portfolio.</div>', unsafe_allow_html=True)

    systemic_events = [s for s in processed_signals if s["impact_score"] >= 7]
    event_options = [f"{s['event_classification']} · Impact {s['impact_score']} · {s['headline'][:45]}..." for s in systemic_events]
    event_options.append("Custom Scenario...")

    chosen_event_str = st.selectbox("Select Systemic Shock Event:", event_options)

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

    # Stress KPIs
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(metric_card("Pre-Stress Value", f"${stress_results['total_pre_stress_mv']/1e6:.1f}M", "Total Book", "neutral", "14 wholesale facilities"), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card("Post-Stress Value", f"${stress_results['total_post_stress_mv']/1e6:.1f}M", f"-{stress_results['portfolio_drawdown_pct']}%", "down", "valuation haircut"), unsafe_allow_html=True)
    with c3:
        st.markdown(metric_card("Expected Loss (EL)", f"${stress_results['post_stress_expected_loss']/1e6:.2f}M", f"+{stress_results['expected_loss_surge_pct']}% surge", "down", "Basel EL = EAD × PD × LGD"), unsafe_allow_html=True)
    with c4:
        st.markdown(metric_card("Shock Multiplier", f"{stress_results['scaling_factor']}x", sel_cat, "neutral", f"Impact {sel_impact}/10"), unsafe_allow_html=True)

    st.markdown(f"""
    <div class="insight-box">
        <div class="insight-header">
            <span>⚡</span> Active Stress Scenario: {stress_results['scenario_name']}
        </div>
        <div class="insight-text">
            Triggered by <strong>{sel_cat}</strong> event with severity rating <strong>{sel_impact}/10</strong> (Severity Multiplier <strong>{stress_results['scaling_factor']}x</strong>).<br>
            Total portfolio valuation contraction: <strong style="color: #DC2626;">-${stress_results['total_portfolio_loss_usd']/1e6:.2f}M</strong> across Commercial Loans, Corporate Bonds, and Derivatives.
        </div>
    </div>
    """, unsafe_allow_html=True)

    detailed_df = stress_results["detailed_asset_impact"]
    col_pie, col_bar = st.columns(2)

    with col_pie:
        with st.container(border=True):
            st.markdown("""
            <div class="chart-card-header">
                <span class="chart-card-title">Loss Allocation by Asset Class</span>
                <span class="chart-card-badge">Valuation Loss</span>
            </div>
            """, unsafe_allow_html=True)
            loss_by_type = detailed_df.groupby("asset_type")["valuation_loss_usd"].sum().reset_index()
            fig_d = px.pie(
                loss_by_type, names="asset_type", values="valuation_loss_usd",
                hole=0.5, color_discrete_sequence=["#DC2626", "#EA580C", "#7C3AED", "#0891B2"]
            )
            apply_clean_layout(
                fig_d,
                height=320,
                showlegend=True,
                legend_override=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
            )
            fig_d.update_traces(textposition="inside", textfont_size=11, marker_line_width=0)
            st.plotly_chart(fig_d, width="stretch")

    with col_bar:
        with st.container(border=True):
            st.markdown("""
            <div class="chart-card-header">
                <span class="chart-card-title">Haircut Percentage by Facility</span>
                <span class="chart-card-badge">Ranked Severity</span>
            </div>
            """, unsafe_allow_html=True)
            fig_b = px.bar(
                detailed_df.sort_values(by="loss_pct", ascending=True),
                y="facility_name", x="loss_pct", color="borrower_sector",
                orientation="h", labels={"loss_pct": "Haircut (%)", "facility_name": ""},
                color_discrete_sequence=CLEAN_COLORS
            )
            apply_clean_layout(fig_b, height=320, showlegend=False)
            fig_b.update_traces(marker_cornerradius=3)
            st.plotly_chart(fig_b, width="stretch")

    with st.container(border=True):
        st.markdown("""
        <div class="chart-card-header">
            <span class="chart-card-title">Basel Credit Risk Asset Ledger</span>
            <span class="chart-card-badge">14 Facilities</span>
        </div>
        """, unsafe_allow_html=True)
        st.dataframe(
            detailed_df[["asset_id", "facility_name", "asset_type", "borrower_sector", "notional_usd",
                          "baseline_pd_pct", "stressed_pd_pct", "baseline_mv_usd", "stressed_mv_usd",
                          "valuation_loss_usd", "loss_pct"]],
            use_container_width=True
        )


# ═══════════════════════════════════════════════════════════════════════════════
# VIEW 5: SCENARIO SANDBOX (LIVE JURY EVALUATION)
# ═══════════════════════════════════════════════════════════════════════════════
elif app_mode == "🔬  Scenario Sandbox":
    st.markdown('<div class="page-title">Live Ad-Hoc Scenario Sandbox</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Test any custom news wire, earnings leak, or tweet in real time. The NLP Engine will extract sentiment, classify taxonomy, score impact, and immediately activate both downstream engines.</div>', unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown('<div style="font-weight: 700; font-size: 14px; margin-bottom: 6px;">Input Unstructured Text:</div>', unsafe_allow_html=True)
        custom_input = st.text_area(
            "Text Input",
            value="Middle Eastern export terminal faces maritime drone disruption; crude futures surge $12 while global sovereign debt spreads widen by 85bps as inflation fears mount.",
            height=90,
            label_visibility="collapsed"
        )
        analyze_clicked = st.button("🚀 Run Live AI Risk Analysis", type="primary")

    if analyze_clicked or True:
        record = {
            "id": f"LIVE-{int(datetime.utcnow().timestamp())}",
            "source_type": "adhoc_input",
            "channel": "Live Jury Tester",
            "raw_text": custom_input
        }
        res = nlp.process_record(record)

        r1, r2, r3, r4 = st.columns(4)
        with r1:
            d = "up" if res["sentiment_score"] > 0 else "down"
            st.markdown(metric_card("Sentiment Score", f"{res['sentiment_score']:+.2f}", "positive" if d == "up" else "negative", d, "calibrated polarity"), unsafe_allow_html=True)
        with r2:
            st.markdown(metric_card("Event Classification", res['event_classification'], f"{res['confidence']*100:.0f}% confidence", "neutral", "institutional taxonomy"), unsafe_allow_html=True)
        with r3:
            st.markdown(metric_card("Impact Score", f"{res['impact_score']}/10", "systemic" if res['impact_score'] >= 7 else "standard", "down" if res['impact_score'] >= 7 else "up", "severity metric"), unsafe_allow_html=True)
        with r4:
            if res["impact_score"] >= 7:
                st.markdown(metric_card("Systemic Risk Trigger", "TRIGGERED", "≥ 7 threshold", "down", "activates Module B"), unsafe_allow_html=True)
            else:
                st.markdown(metric_card("Systemic Risk Trigger", "STANDARD", "< 7 threshold", "up", "normal monitoring"), unsafe_allow_html=True)

        t1, t2 = st.columns(2)
        with t1:
            entities_str = ", ".join(res["target_entities"]) if res["target_entities"] else "No single ticker (Broad Market Impact)"
            st.markdown(f"""
            <div class="insight-box">
                <div class="insight-header"><span>📈</span> Module A Impact: Tactical Rebalancer</div>
                <div class="insight-text">
                    <strong>Entities Resolved:</strong> {entities_str}<br>
                    <strong>Action:</strong> Sentiment shift of <strong>{res['sentiment_score']:+.2f}</strong> propagates to S&P 100 constituent weights with dynamic box constraints [2%, 22%].
                </div>
            </div>
            """, unsafe_allow_html=True)

        with t2:
            if res["impact_score"] >= 7:
                status_html = f'<strong style="color: #DC2626;">SYSTEMIC STRESS INITIATED</strong> (Severity {res["impact_score"]}/10). Stress shocks queued for category <strong>{res["event_classification"]}</strong> across the $505M portfolio.'
            else:
                status_html = "Impact score is below systemic threshold (&lt; 7). Normal portfolio monitoring active."
            st.markdown(f"""
            <div class="insight-box">
                <div class="insight-header"><span>🛡</span> Module B Impact: Wholesale Stress Tester</div>
                <div class="insight-text">{status_html}</div>
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="display: flex; justify-content: space-between; align-items: center; color: #94A3B8; font-size: 12px; padding: 6px 0;">
    <div><strong>Autonomous AI/NLP Risk Engine</strong> · S&P Global & CRISIL Campus Hackathon 2026</div>
    <div>Candidate: <strong>Mehul Goyal</strong> · Campus: <strong>VIT</strong> · <code>mehul.goyal2023@vitstudent.ac.in</code></div>
</div>
""", unsafe_allow_html=True)
