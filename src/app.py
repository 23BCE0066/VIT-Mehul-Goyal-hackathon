"""
Interactive Financial Risk & Intelligence Platform
S&P Global & CRISIL Campus Hackathon 2026
Candidate: Mehul Goyal | VIT

UI: Neumorphism Design System
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

import sys
from pathlib import Path

# Ensure root directory is always on python sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import backend modules
from src.engine.ingestion import IngestionPipeline
from src.engine.nlp_processor import NLPRiskEngine
from src.module_a.index_rebalancer import TacticalIndexRebalancer
from src.module_b.synthetic_book import WholesalePortfolioBook
from src.module_b.stress_engine import StrategicStressTester

# Page configuration
st.set_page_config(
    page_title="S&P Global & CRISIL | AI/NLP Risk Intelligence Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# NEUMORPHISM DESIGN SYSTEM - COMPLETE CSS OVERHAUL
# ─────────────────────────────────────────────────────────────────────────────
NEUMORPH_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ═══════════ GLOBAL RESET & BASE ═══════════ */
:root {
    --bg:        #2D3250;
    --bg-light:  #3A3F5C;
    --bg-dark:   #232842;
    --surface:   #343A5C;
    --accent:    #7077A1;
    --primary:   #00D4AA;
    --danger:    #FF6B6B;
    --warning:   #FFB347;
    --info:      #7EC8E3;
    --gold:      #F7DC6F;
    --text:      #E8E8F0;
    --text-dim:  #9CA3C2;
    --text-muted:#6B72A0;
    --shadow-dark:  rgba(20, 22, 38, 0.60);
    --shadow-light: rgba(70, 78, 120, 0.25);
    --radius: 18px;
    --radius-sm: 12px;
}

/* Full app background */
.stApp, [data-testid="stAppViewContainer"], [data-testid="stAppViewBlockContainer"] {
    background: var(--bg) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: var(--text) !important;
}

[data-testid="stHeader"] {
    background: transparent !important;
}

/* ═══════════ SIDEBAR NEUMORPHISM ═══════════ */
[data-testid="stSidebar"] {
    background: var(--bg-dark) !important;
    border-right: none !important;
    box-shadow: 8px 0 24px var(--shadow-dark) !important;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] span {
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
}

/* ═══════════ NEUMORPHIC CARD CLASSES ═══════════ */

/* Raised / convex card */
.neu-card {
    background: var(--surface);
    border-radius: var(--radius);
    padding: 28px 24px;
    margin-bottom: 20px;
    box-shadow:
        8px 8px 20px var(--shadow-dark),
       -6px -6px 16px var(--shadow-light);
    border: 1px solid rgba(112, 119, 161, 0.08);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.neu-card:hover {
    transform: translateY(-2px);
    box-shadow:
        10px 10px 28px var(--shadow-dark),
        -8px -8px 20px var(--shadow-light);
}

/* Inset / pressed card */
.neu-inset {
    background: var(--bg-dark);
    border-radius: var(--radius-sm);
    padding: 22px 20px;
    margin-bottom: 16px;
    box-shadow:
        inset 5px 5px 12px var(--shadow-dark),
        inset -4px -4px 10px var(--shadow-light);
    border: 1px solid rgba(112, 119, 161, 0.06);
}

/* Flat subtle card */
.neu-flat {
    background: linear-gradient(145deg, var(--bg-light), var(--bg-dark));
    border-radius: var(--radius);
    padding: 24px 22px;
    margin-bottom: 16px;
    box-shadow:
        6px 6px 16px var(--shadow-dark),
       -4px -4px 12px var(--shadow-light);
}

/* ═══════════ METRIC TILES ═══════════ */
.neu-metric {
    background: var(--surface);
    border-radius: var(--radius);
    padding: 24px 20px;
    text-align: center;
    box-shadow:
        8px 8px 20px var(--shadow-dark),
       -6px -6px 16px var(--shadow-light);
    border: 1px solid rgba(112, 119, 161, 0.08);
    transition: transform 0.22s ease;
}
.neu-metric:hover {
    transform: translateY(-3px);
}
.neu-metric .metric-icon {
    font-size: 28px;
    margin-bottom: 8px;
}
.neu-metric .metric-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1.4px;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 6px;
}
.neu-metric .metric-value {
    font-size: 30px;
    font-weight: 800;
    line-height: 1.1;
    margin-bottom: 4px;
}
.neu-metric .metric-sub {
    font-size: 12px;
    color: var(--text-dim);
    font-weight: 400;
}

/* Color variants */
.text-primary { color: var(--primary) !important; }
.text-danger  { color: var(--danger) !important; }
.text-warning { color: var(--warning) !important; }
.text-info    { color: var(--info) !important; }
.text-gold    { color: var(--gold) !important; }
.text-text    { color: var(--text) !important; }

/* ═══════════ SIGNAL FEED ITEM ═══════════ */
.neu-feed-item {
    background: var(--surface);
    border-radius: var(--radius-sm);
    padding: 20px;
    margin-bottom: 14px;
    box-shadow:
        6px 6px 14px var(--shadow-dark),
       -4px -4px 10px var(--shadow-light);
    border-left: 4px solid var(--accent);
    display: flex;
    align-items: flex-start;
    gap: 16px;
    transition: border-color 0.2s;
}
.neu-feed-item:hover {
    border-left-color: var(--primary);
}
.neu-feed-item.systemic {
    border-left-color: var(--danger);
}

.feed-body {
    flex: 1;
}
.feed-headline {
    font-size: 14px;
    font-weight: 600;
    color: var(--text);
    margin-bottom: 6px;
    line-height: 1.4;
}
.feed-snippet {
    font-size: 12.5px;
    color: var(--text-dim);
    line-height: 1.5;
    margin-bottom: 8px;
}
.feed-meta {
    display: flex;
    gap: 14px;
    flex-wrap: wrap;
    align-items: center;
}

/* ═══════════ BADGES / PILLS ═══════════ */
.neu-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 10.5px;
    font-weight: 700;
    letter-spacing: 0.6px;
    text-transform: uppercase;
}
.badge-systemic {
    background: linear-gradient(135deg, #FF6B6B, #EE5A24);
    color: #fff;
    box-shadow: 0 3px 10px rgba(255, 107, 107, 0.30);
}
.badge-standard {
    background: linear-gradient(135deg, var(--primary), #00B894);
    color: #fff;
    box-shadow: 0 3px 10px rgba(0, 212, 170, 0.25);
}
.badge-category {
    background: var(--bg-dark);
    color: var(--info);
    border: 1px solid rgba(126, 200, 227, 0.25);
    box-shadow: inset 2px 2px 5px var(--shadow-dark), inset -2px -2px 4px var(--shadow-light);
}
.badge-entity {
    background: var(--bg-dark);
    color: var(--gold);
    border: 1px solid rgba(247, 220, 111, 0.20);
    font-family: 'SF Mono', 'Fira Code', monospace;
    font-size: 10px;
}

/* ═══════════ SECTION HEADERS ═══════════ */
.section-header {
    font-family: 'Inter', sans-serif;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 18px;
    padding-bottom: 10px;
    border-bottom: 2px solid rgba(112, 119, 161, 0.15);
}

.page-title {
    font-family: 'Inter', sans-serif;
    font-size: 32px;
    font-weight: 800;
    color: var(--text);
    margin-bottom: 6px;
    letter-spacing: -0.5px;
}
.page-subtitle {
    font-size: 15px;
    color: var(--text-dim);
    font-weight: 400;
    line-height: 1.6;
    margin-bottom: 28px;
}

/* ═══════════ STREAMLIT WIDGET OVERRIDES ═══════════ */

/* Radio buttons */
[data-testid="stSidebar"] .stRadio > div {
    gap: 4px !important;
}
[data-testid="stSidebar"] .stRadio > div > label {
    background: var(--surface) !important;
    border-radius: 10px !important;
    padding: 10px 14px !important;
    margin-bottom: 4px !important;
    box-shadow: 4px 4px 10px var(--shadow-dark), -3px -3px 8px var(--shadow-light) !important;
    transition: all 0.2s ease !important;
    border: 1px solid transparent !important;
}
[data-testid="stSidebar"] .stRadio > div > label:hover {
    border-color: var(--primary) !important;
}
[data-testid="stSidebar"] .stRadio > div > label[data-checked="true"],
[data-testid="stSidebar"] .stRadio > div > label[aria-checked="true"] {
    background: var(--bg) !important;
    box-shadow: inset 4px 4px 10px var(--shadow-dark), inset -3px -3px 8px var(--shadow-light) !important;
    border-color: var(--primary) !important;
}

/* Slider */
.stSlider > div > div > div {
    background: var(--primary) !important;
}

/* Expanders */
[data-testid="stExpander"] {
    background: var(--surface) !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    box-shadow: 6px 6px 14px var(--shadow-dark), -4px -4px 10px var(--shadow-light) !important;
    margin-bottom: 12px !important;
}
[data-testid="stExpander"] summary {
    color: var(--text) !important;
    font-weight: 500 !important;
}

/* Select boxes, multiselect */
[data-testid="stSelectbox"], [data-testid="stMultiSelect"] {
    font-family: 'Inter', sans-serif !important;
}

/* Buttons */
.stButton > button {
    background: var(--surface) !important;
    color: var(--primary) !important;
    border: 1px solid rgba(0, 212, 170, 0.20) !important;
    border-radius: var(--radius-sm) !important;
    box-shadow: 6px 6px 14px var(--shadow-dark), -4px -4px 10px var(--shadow-light) !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.3px;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: var(--primary) !important;
    color: var(--bg-dark) !important;
    transform: translateY(-1px);
}
.stButton > button[kind="primary"],
.stButton > button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, var(--primary), #00B894) !important;
    color: var(--bg-dark) !important;
    border: none !important;
    box-shadow: 6px 6px 14px var(--shadow-dark), -4px -4px 10px var(--shadow-light), 0 4px 14px rgba(0, 212, 170, 0.25) !important;
}

/* Download button */
.stDownloadButton > button {
    background: var(--surface) !important;
    color: var(--info) !important;
    border: 1px solid rgba(126, 200, 227, 0.20) !important;
    border-radius: var(--radius-sm) !important;
    box-shadow: 6px 6px 14px var(--shadow-dark), -4px -4px 10px var(--shadow-light) !important;
}

/* Text area */
.stTextArea textarea {
    background: var(--bg-dark) !important;
    color: var(--text) !important;
    border: 1px solid rgba(112, 119, 161, 0.15) !important;
    border-radius: var(--radius-sm) !important;
    box-shadow: inset 4px 4px 10px var(--shadow-dark), inset -3px -3px 8px var(--shadow-light) !important;
    font-family: 'Inter', sans-serif !important;
}

/* Metrics */
[data-testid="stMetric"] {
    background: var(--surface) !important;
    border-radius: var(--radius) !important;
    padding: 18px !important;
    box-shadow: 6px 6px 16px var(--shadow-dark), -5px -5px 12px var(--shadow-light) !important;
    border: 1px solid rgba(112, 119, 161, 0.06);
}
[data-testid="stMetricLabel"] {
    color: var(--text-muted) !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    text-transform: uppercase !important;
    font-size: 11px !important;
}
[data-testid="stMetricValue"] {
    color: var(--text) !important;
    font-weight: 800 !important;
}

/* DataFrames */
[data-testid="stDataFrame"] {
    border-radius: var(--radius-sm) !important;
    overflow: hidden;
    box-shadow: 6px 6px 16px var(--shadow-dark), -4px -4px 10px var(--shadow-light) !important;
}

/* Alert boxes */
.stAlert {
    background: var(--surface) !important;
    border-radius: var(--radius-sm) !important;
    box-shadow: 4px 4px 10px var(--shadow-dark), -3px -3px 8px var(--shadow-light) !important;
    border-left-width: 4px !important;
}

/* Tab navigation */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
}
.stTabs [data-baseweb="tab"] {
    background: var(--surface) !important;
    border-radius: 10px !important;
    box-shadow: 4px 4px 10px var(--shadow-dark), -3px -3px 8px var(--shadow-light) !important;
    color: var(--text-dim) !important;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: var(--bg-dark) !important;
    box-shadow: inset 4px 4px 10px var(--shadow-dark), inset -3px -3px 8px var(--shadow-light) !important;
    color: var(--primary) !important;
    font-weight: 700;
}

/* Scrollbar */
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: var(--bg-dark); }
::-webkit-scrollbar-thumb { background: var(--accent); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--primary); }

</style>
"""
st.markdown(NEUMORPH_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY NEUMORPHIC THEME
# ─────────────────────────────────────────────────────────────────────────────
NEU_PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(52, 58, 92, 0)",
    plot_bgcolor="rgba(35, 40, 66, 0.5)",
    font=dict(family="Inter, sans-serif", color="#E8E8F0", size=12),
    title_font=dict(size=15, color="#E8E8F0", family="Inter"),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#9CA3C2", size=11)),
    xaxis=dict(gridcolor="rgba(112,119,161,0.10)", zerolinecolor="rgba(112,119,161,0.15)"),
    yaxis=dict(gridcolor="rgba(112,119,161,0.10)", zerolinecolor="rgba(112,119,161,0.15)"),
    margin=dict(l=40, r=20, t=50, b=40),
)
NEU_COLORS = ["#00D4AA", "#7EC8E3", "#F7DC6F", "#FF6B6B", "#AB97EA", "#FFB347", "#77DD77"]


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def neu_metric_tile(icon, label, value, color_class="text-primary", subtitle=""):
    sub_html = f'<div class="metric-sub">{subtitle}</div>' if subtitle else ""
    return f"""
    <div class="neu-metric">
        <div class="metric-icon">{icon}</div>
        <div class="metric-label">{label}</div>
        <div class="metric-value {color_class}">{value}</div>
        {sub_html}
    </div>
    """

def neu_feed_card(item):
    is_systemic = item["impact_score"] >= 7
    card_class = "neu-feed-item systemic" if is_systemic else "neu-feed-item"
    badge = '<span class="neu-badge badge-systemic">⚠ SYSTEMIC</span>' if is_systemic else '<span class="neu-badge badge-standard">✓ STANDARD</span>'
    
    sent = item["sentiment_score"]
    sent_color = "text-primary" if sent > 0 else "text-danger"
    sent_icon = "📈" if sent > 0 else "📉"

    entities_html = ""
    for e in item.get("target_entities", []):
        entities_html += f'<span class="neu-badge badge-entity">{e}</span> '

    return f"""
    <div class="{card_class}">
        <div style="font-size:28px; min-width:36px;">{sent_icon}</div>
        <div class="feed-body">
            <div class="feed-headline">{item.get('headline') or item['text_snippet'][:80]}</div>
            <div class="feed-snippet">{item['text_snippet']}</div>
            <div class="feed-meta">
                {badge}
                <span class="neu-badge badge-category">{item['event_classification']}</span>
                <span style="color: var(--text-dim); font-size: 11px;">
                    Sent: <strong class="{sent_color}">{sent:+.2f}</strong> ·
                    Impact: <strong class="{'text-danger' if is_systemic else 'text-warning'}">{item['impact_score']}/10</strong> ·
                    Conf: {item['confidence']*100:.0f}%
                </span>
                {entities_html}
            </div>
        </div>
    </div>
    """


# ─────────────────────────────────────────────────────────────────────────────
# LOAD COMPONENTS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_components():
    pipeline = IngestionPipeline()
    nlp_eng = NLPRiskEngine()
    rebalancer = TacticalIndexRebalancer()
    stress_tester = StrategicStressTester()
    return pipeline, nlp_eng, rebalancer, stress_tester

pipeline, nlp, rebalancer, stress_tester = load_components()


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.markdown("""
<div style="text-align:center; padding: 12px 0 4px 0;">
    <div style="font-size: 28px; font-weight: 800; color: #00D4AA; letter-spacing: -0.5px; font-family: Inter, sans-serif;">S&P Global</div>
    <div style="font-size: 12px; color: #9CA3C2; letter-spacing: 2px; text-transform: uppercase; font-weight: 600;">& CRISIL Hackathon 2026</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

app_mode = st.sidebar.radio(
    "Navigation",
    [
        "📊  Executive Overview",
        "🧠  Core NLP Risk Engine",
        "⚡  Tactical Index Rebalancer",
        "🛡  Portfolio Stress Testing",
        "🔬  Ad-Hoc Scenario Sandbox"
    ]
)

st.sidebar.markdown("---")

# Pipeline health indicators
st.sidebar.markdown("""
<div class="neu-inset" style="margin-top: 6px;">
    <div style="font-size: 10px; letter-spacing: 1.5px; text-transform: uppercase; color: #6B72A0; font-weight: 700; margin-bottom: 10px;">Pipeline Health</div>
    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
        <div style="width: 8px; height: 8px; border-radius: 50%; background: #00D4AA; box-shadow: 0 0 8px #00D4AA;"></div>
        <span style="font-size: 12px; color: #E8E8F0;">Ingestion Active</span>
    </div>
    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
        <div style="width: 8px; height: 8px; border-radius: 50%; background: #7EC8E3; box-shadow: 0 0 8px #7EC8E3;"></div>
        <span style="font-size: 12px; color: #E8E8F0;">NLP Engine: 5 Classes</span>
    </div>
    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
        <div style="width: 8px; height: 8px; border-radius: 50%; background: #F7DC6F; box-shadow: 0 0 8px #F7DC6F;"></div>
        <span style="font-size: 12px; color: #E8E8F0;">Module A & B Online</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div style="text-align:center; padding: 14px 0 0 0;">
    <div style="font-size: 11px; color: #6B72A0;">Built by</div>
    <div style="font-size: 14px; font-weight: 700; color: #E8E8F0;">Mehul Goyal</div>
    <div style="font-size: 11px; color: #6B72A0;">VIT · mehul.goyal2023@vitstudent.ac.in</div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
raw_stream = pipeline.load_unified_stream()
processed_signals = nlp.process_batch(raw_stream)


# ═══════════════════════════════════════════════════════════════════════════════
# VIEW 1: EXECUTIVE OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if app_mode == "📊  Executive Overview":
    st.markdown('<div class="page-title">Unified AI/NLP Risk Intelligence Platform</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Ingests real-time unstructured market intelligence from financial news and social feeds, extracts quantitative risk signals via NLP, and powers tactical asset rebalancing and wholesale credit stress testing.</div>', unsafe_allow_html=True)

    # ── KPI TILES ──
    systemic_count = sum(1 for s in processed_signals if s["impact_score"] >= 7)
    avg_sent = float(np.mean([s["sentiment_score"] for s in processed_signals]))

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(neu_metric_tile("📡", "Ingested Records", len(processed_signals), "text-info", "News + Social Streams"), unsafe_allow_html=True)
    with c2:
        st.markdown(neu_metric_tile("🚨", "Systemic Triggers", systemic_count, "text-danger", "Impact Score ≥ 7"), unsafe_allow_html=True)
    with c3:
        sc = "text-primary" if avg_sent >= 0 else "text-danger"
        st.markdown(neu_metric_tile("💹", "Net Sentiment", f"{avg_sent:+.2f}", sc, "Aggregate Market Polarity"), unsafe_allow_html=True)
    with c4:
        st.markdown(neu_metric_tile("⚙️", "Active Modules", "A & B", "text-gold", "Rebalancer + Stress Tester"), unsafe_allow_html=True)

    st.markdown("")

    # ── FILTERS ──
    st.markdown('<div class="section-header">Live Multi-Source Signal Feed</div>', unsafe_allow_html=True)

    f1, f2 = st.columns(2)
    with f1:
        src_filter = st.multiselect("Source Channel", ["news", "social"], default=["news", "social"])
    with f2:
        cat_filter = st.multiselect(
            "Event Category",
            ["Macroeconomic", "Geopolitical", "Credit Event", "Product Launch", "Merger/Acquisition"],
            default=["Macroeconomic", "Geopolitical", "Credit Event", "Product Launch", "Merger/Acquisition"]
        )

    filtered_sigs = [
        s for s in processed_signals
        if s["source_type"] in src_filter and s["event_classification"] in cat_filter
    ]

    for item in filtered_sigs:
        st.markdown(neu_feed_card(item), unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# VIEW 2: CORE AI/NLP RISK ENGINE
# ═══════════════════════════════════════════════════════════════════════════════
elif app_mode == "🧠  Core NLP Risk Engine":
    st.markdown('<div class="page-title">Core AI/NLP Risk Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Deep diagnostics into the NLP extraction pipeline — event taxonomy distribution, sentiment-impact dispersion, and structured signal output table.</div>', unsafe_allow_html=True)

    df_signals = pd.DataFrame(processed_signals)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">Event Classification Breakdown</div>', unsafe_allow_html=True)
        cat_counts = df_signals["event_classification"].value_counts().reset_index()
        cat_counts.columns = ["Event Category", "Frequency"]
        fig_cat = px.bar(
            cat_counts, x="Event Category", y="Frequency",
            color="Event Category",
            color_discrete_sequence=NEU_COLORS,
            title="Distribution of Classified Institutional Events"
        )
        fig_cat.update_layout(**NEU_PLOTLY_LAYOUT)
        fig_cat.update_traces(marker_line_width=0, marker_cornerradius=8)
        st.plotly_chart(fig_cat, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Sentiment vs Impact Severity</div>', unsafe_allow_html=True)
        fig_scatter = px.scatter(
            df_signals,
            x="sentiment_score", y="impact_score",
            color="event_classification",
            size="impact_score",
            hover_data=["headline", "target_entities"],
            labels={"sentiment_score": "Sentiment (-1.0 to +1.0)", "impact_score": "Impact (1 to 10)"},
            color_discrete_sequence=NEU_COLORS,
            title="Event Impact vs Sentiment Polarity"
        )
        fig_scatter.add_hline(y=7, line_dash="dot", line_color="#FF6B6B", annotation_text="Systemic Threshold", annotation_font_color="#FF6B6B")
        fig_scatter.update_layout(**NEU_PLOTLY_LAYOUT)
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown('<div class="section-header">Structured Signals Output Table</div>', unsafe_allow_html=True)
    st.dataframe(
        df_signals[["signal_id", "source_type", "event_classification", "sentiment_score", "impact_score", "confidence", "target_entities"]],
        use_container_width=True
    )

    st.download_button(
        "⬇ Download Signals as JSON",
        data=df_signals.to_json(orient="records", indent=2),
        file_name="structured_risk_signals.json",
        mime="application/json"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# VIEW 3: MODULE A — TACTICAL INDEX REBALANCER
# ═══════════════════════════════════════════════════════════════════════════════
elif app_mode == "⚡  Tactical Index Rebalancer":
    st.markdown('<div class="page-title">Module A: Tactical Index Rebalancer</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Subscribes to live sentiment scores from the NLP engine to dynamically tilt an institutional 12-stock S&P 100 constituent basket. Overweights positive momentum, defensively trims negative exposures.</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">Tilt Configuration</div>', unsafe_allow_html=True)
    tilt_strength = st.slider("Tactical Tilt Aggressiveness (λ Multiplier)", min_value=0.1, max_value=1.5, value=0.65, step=0.05)
    rebalancer.tilt_intensity = tilt_strength

    net_sentiments = rebalancer.aggregate_signals_by_ticker(processed_signals)
    rebal_df = rebalancer.compute_rebalanced_weights(net_sentiments)
    history_df, metrics = rebalancer.run_backtest_simulation(rebal_df)

    # ── Performance Tiles ──
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(neu_metric_tile("🎯", "Tactical Return", f"{metrics['tactical_total_return_pct']}%", "text-primary", f"+{metrics['generated_alpha_bps']} bps Alpha"), unsafe_allow_html=True)
    with m2:
        st.markdown(neu_metric_tile("📊", "Benchmark Return", f"{metrics['benchmark_total_return_pct']}%", "text-info", "S&P 100 Basket"), unsafe_allow_html=True)
    with m3:
        st.markdown(neu_metric_tile("⚖️", "Sharpe Ratio", f"{metrics['sharpe_ratio']}", "text-gold", "Risk-Adjusted Performance"), unsafe_allow_html=True)
    with m4:
        st.markdown(neu_metric_tile("📐", "Ann. Volatility", f"{metrics['tactical_annualized_vol']}%", "text-warning", "Portfolio Standard Deviation"), unsafe_allow_html=True)

    st.markdown("")

    col_w, col_perf = st.columns(2)

    with col_w:
        st.markdown('<div class="section-header">Weight Allocation Comparison</div>', unsafe_allow_html=True)
        fig_weights = go.Figure()
        fig_weights.add_trace(go.Bar(
            name='Benchmark', x=rebal_df['ticker'], y=rebal_df['benchmark_weight'] * 100,
            marker_color='#7077A1', marker_cornerradius=6
        ))
        fig_weights.add_trace(go.Bar(
            name='Sentiment-Rebalanced', x=rebal_df['ticker'], y=rebal_df['rebalanced_weight'] * 100,
            marker_color='#00D4AA', marker_cornerradius=6
        ))
        fig_weights.update_layout(barmode='group', yaxis_title="Allocation (%)", title="Benchmark vs NLP-Tilted Weights", **NEU_PLOTLY_LAYOUT)
        st.plotly_chart(fig_weights, use_container_width=True)

    with col_perf:
        st.markdown('<div class="section-header">Cumulative Performance Backtest</div>', unsafe_allow_html=True)
        fig_perf = go.Figure()
        fig_perf.add_trace(go.Scatter(
            x=history_df.index, y=history_df["Benchmark_Index"],
            name="Benchmark", line=dict(color="#7077A1", width=2, dash="dot"), fill=None
        ))
        fig_perf.add_trace(go.Scatter(
            x=history_df.index, y=history_df["Tactical_NLP_Index"],
            name="NLP Tactical Index", line=dict(color="#00D4AA", width=3),
            fill="tonexty", fillcolor="rgba(0, 212, 170, 0.08)"
        ))
        fig_perf.update_layout(yaxis_title="Index Level (Base 100)", title="Tactical Sentiment Index vs Benchmark", **NEU_PLOTLY_LAYOUT)
        st.plotly_chart(fig_perf, use_container_width=True)

    st.markdown('<div class="section-header">Rebalancing Ledger</div>', unsafe_allow_html=True)
    st.dataframe(
        rebal_df[["ticker", "name", "sector", "net_sentiment", "benchmark_weight", "rebalanced_weight", "weight_delta", "allocation_status"]],
        use_container_width=True
    )


# ═══════════════════════════════════════════════════════════════════════════════
# VIEW 4: MODULE B — PORTFOLIO STRESS TESTING
# ═══════════════════════════════════════════════════════════════════════════════
elif app_mode == "🛡  Portfolio Stress Testing":
    st.markdown('<div class="page-title">Module B: Wholesale Portfolio Stress Testing</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Subscribes to Event Classification and Impact Score from the NLP Engine. When systemic disruption is detected (Impact ≥ 7), macro and credit shocks transmit across a $505M wholesale banking asset portfolio.</div>', unsafe_allow_html=True)

    # Event Trigger Selection
    systemic_events = [s for s in processed_signals if s["impact_score"] >= 7]
    event_options = [f"{s['event_classification']}  ·  Impact {s['impact_score']}  ·  {s['headline'][:45]}..." for s in systemic_events]
    event_options.append("🔧 Custom Scenario Trigger...")

    st.markdown('<div class="section-header">Select Stress Trigger</div>', unsafe_allow_html=True)
    chosen_event_str = st.selectbox("Detected NLP Event", event_options, label_visibility="collapsed")

    if "Custom Scenario Trigger" in chosen_event_str:
        cc1, cc2 = st.columns(2)
        with cc1:
            sel_cat = st.selectbox("Event Category", ["Geopolitical", "Macroeconomic", "Credit Event"])
        with cc2:
            sel_impact = st.slider("Impact Score Severity", 7, 10, 8)
    else:
        idx = event_options.index(chosen_event_str)
        sel_event = systemic_events[idx]
        sel_cat = sel_event["event_classification"]
        sel_impact = sel_event["impact_score"]

    stress_results = stress_tester.run_stress_test(sel_cat, sel_impact)

    # ── Stress KPIs ──
    b1, b2, b3, b4 = st.columns(4)
    with b1:
        st.markdown(neu_metric_tile("🏦", "Pre-Stress Value", f"${stress_results['total_pre_stress_mv']/1e6:.1f}M", "text-info", "Total Portfolio MtM"), unsafe_allow_html=True)
    with b2:
        st.markdown(neu_metric_tile("💥", "Post-Stress Value", f"${stress_results['total_post_stress_mv']/1e6:.1f}M", "text-danger", f"-{stress_results['portfolio_drawdown_pct']}% Drawdown"), unsafe_allow_html=True)
    with b3:
        st.markdown(neu_metric_tile("📉", "Expected Loss", f"${stress_results['post_stress_expected_loss']/1e6:.2f}M", "text-warning", f"+{stress_results['expected_loss_surge_pct']}% EL Surge"), unsafe_allow_html=True)
    with b4:
        st.markdown(neu_metric_tile("⚡", "Shock Multiplier", f"{stress_results['scaling_factor']}x", "text-gold", f"Scenario: {sel_cat}"), unsafe_allow_html=True)

    st.markdown("")

    st.markdown(f"""
    <div class="neu-flat">
        <div style="font-size: 13px; font-weight: 700; letter-spacing: 1px; color: #FF6B6B; text-transform: uppercase; margin-bottom: 6px;">Active Scenario</div>
        <div style="font-size: 18px; font-weight: 700; color: #E8E8F0;">{stress_results['scenario_name']}</div>
        <div style="font-size: 12px; color: #9CA3C2; margin-top: 4px;">Impact Score: {sel_impact}/10  ·  Severity Scale: {stress_results['scaling_factor']}x  ·  Category: {sel_cat}</div>
    </div>
    """, unsafe_allow_html=True)

    detailed_df = stress_results["detailed_asset_impact"]

    col_pie, col_bar = st.columns(2)

    with col_pie:
        st.markdown('<div class="section-header">Loss by Asset Class</div>', unsafe_allow_html=True)
        loss_by_type = detailed_df.groupby("asset_type")["valuation_loss_usd"].sum().reset_index()
        fig_donut = px.pie(
            loss_by_type, names="asset_type", values="valuation_loss_usd",
            hole=0.5, title="Capital Loss Distribution",
            color_discrete_sequence=["#FF6B6B", "#FFB347", "#AB97EA", "#7EC8E3"]
        )
        fig_donut.update_layout(**NEU_PLOTLY_LAYOUT)
        fig_donut.update_traces(textfont_color="#E8E8F0", marker_line_width=0)
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_bar:
        st.markdown('<div class="section-header">Individual Asset Haircuts</div>', unsafe_allow_html=True)
        fig_bar = px.bar(
            detailed_df.sort_values(by="loss_pct", ascending=True),
            y="facility_name", x="loss_pct",
            color="borrower_sector", orientation="h",
            title="Valuation Haircut by Facility (%)",
            labels={"loss_pct": "Haircut (%)", "facility_name": ""},
            color_discrete_sequence=NEU_COLORS
        )
        fig_bar.update_layout(**NEU_PLOTLY_LAYOUT)
        fig_bar.update_traces(marker_cornerradius=4)
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown('<div class="section-header">Basel Credit Risk Ledger — Pre & Post Stress</div>', unsafe_allow_html=True)
    st.dataframe(
        detailed_df[["asset_id", "facility_name", "asset_type", "borrower_sector", "notional_usd",
                      "baseline_pd_pct", "stressed_pd_pct", "baseline_mv_usd", "stressed_mv_usd",
                      "valuation_loss_usd", "loss_pct"]],
        use_container_width=True
    )


# ═══════════════════════════════════════════════════════════════════════════════
# VIEW 5: AD-HOC SCENARIO SANDBOX
# ═══════════════════════════════════════════════════════════════════════════════
elif app_mode == "🔬  Ad-Hoc Scenario Sandbox":
    st.markdown('<div class="page-title">Ad-Hoc Scenario Sandbox</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Paste any breaking financial headline, earnings release, or social post and watch the full NLP pipeline extract structured signals and trigger downstream actions in real-time.</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">Input: Breaking Event Text</div>', unsafe_allow_html=True)

    custom_input = st.text_area(
        "Enter Text:",
        value="Middle Eastern export terminal faces maritime drone disruption; crude futures surge $12 while global sovereign debt spreads widen by 85bps as inflation fears mount.",
        height=110,
        label_visibility="collapsed"
    )

    if st.button("Run Unified AI Risk Signal Analysis", type="primary", use_container_width=True):
        record = {
            "id": f"LIVE-{int(datetime.utcnow().timestamp())}",
            "source_type": "adhoc_input",
            "channel": "Live Jury Tester",
            "raw_text": custom_input
        }
        res = nlp.process_record(record)

        st.markdown("")

        # Results tiles
        r1, r2, r3, r4 = st.columns(4)
        with r1:
            sc = "text-primary" if res["sentiment_score"] > 0 else "text-danger"
            st.markdown(neu_metric_tile("💬", "Sentiment", f"{res['sentiment_score']:+.2f}", sc), unsafe_allow_html=True)
        with r2:
            st.markdown(neu_metric_tile("🏷", "Event Category", res['event_classification'], "text-info"), unsafe_allow_html=True)
        with r3:
            ic = "text-danger" if res["impact_score"] >= 7 else "text-warning"
            st.markdown(neu_metric_tile("⚡", "Impact Score", f"{res['impact_score']} / 10", ic), unsafe_allow_html=True)
        with r4:
            if res["impact_score"] >= 7:
                st.markdown(neu_metric_tile("🚨", "Systemic?", "YES", "text-danger", "Stress test triggered!"), unsafe_allow_html=True)
            else:
                st.markdown(neu_metric_tile("✅", "Systemic?", "NO", "text-primary", "Below threshold"), unsafe_allow_html=True)

        st.markdown("")
        st.markdown('<div class="section-header">Downstream Action Triggers</div>', unsafe_allow_html=True)

        t1, t2 = st.columns(2)
        with t1:
            entities_str = ", ".join(res["target_entities"]) if res["target_entities"] else "No specific tickers"
            action_a = f"Identified: <strong>{entities_str}</strong>. Reallocating portfolio weights with net sentiment <strong>{res['sentiment_score']:+.2f}</strong>." if res["target_entities"] else "No specific ticker identified. General macro sentiment shift applied across entire index universe."
            st.markdown(f"""
            <div class="neu-card">
                <div style="font-size: 12px; letter-spacing: 1.2px; text-transform: uppercase; color: #00D4AA; font-weight: 700; margin-bottom: 10px;">⚡ Module A: Tactical Rebalancer</div>
                <div style="font-size: 13px; color: #E8E8F0; line-height: 1.6;">{action_a}</div>
            </div>
            """, unsafe_allow_html=True)

        with t2:
            if res["impact_score"] >= 7:
                action_b = f"<span style='color: #FF6B6B; font-weight: 700;'>SYSTEMIC STRESS TRIGGERED!</span><br>Shock parameters queued for scenario: <strong>{res['event_classification']}</strong>. Wholesale banking portfolio $505M under evaluation."
            else:
                action_b = "Impact score is below systemic threshold (&lt; 7). No emergency credit provisions required. Portfolio remains in baseline state."
            st.markdown(f"""
            <div class="neu-card">
                <div style="font-size: 12px; letter-spacing: 1.2px; text-transform: uppercase; color: #FF6B6B; font-weight: 700; margin-bottom: 10px;">🛡 Module B: Stress Tester</div>
                <div style="font-size: 13px; color: #E8E8F0; line-height: 1.6;">{action_b}</div>
            </div>
            """, unsafe_allow_html=True)
