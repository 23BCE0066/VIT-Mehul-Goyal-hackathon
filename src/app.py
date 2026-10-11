"""
Interactive Financial Risk & Intelligence Platform
S&P Global & CRISIL Campus Hackathon 2026
Candidate: Mehul Goyal | VIT
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

# Custom Institutional Theme Styling
st.markdown("""
<style>
    .reportview-container {
        background: #0E1117;
    }
    .metric-card {
        background-color: #1A1F2C;
        border: 1px solid #2E384D;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: 700;
        color: #00D2FF;
    }
    .metric-label {
        font-size: 13px;
        color: #8C9BAE;
        text-transform: uppercase;
    }
    .badge-systemic {
        background-color: #FF4B4B;
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: bold;
    }
    .badge-normal {
        background-color: #00AA6C;
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_components():
    pipeline = IngestionPipeline()
    nlp = NLPRiskEngine()
    rebalancer = TacticalIndexRebalancer()
    stress_tester = StrategicStressTester()
    return pipeline, nlp, rebalancer, stress_tester


pipeline, nlp, rebalancer, stress_tester = load_components()

# Sidebar Navigation & Platform Header
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/87/S%26P_Global_logo.svg/320px-S%26P_Global_logo.svg.png", width=210)
st.sidebar.title("Risk Intelligence Hub")
st.sidebar.markdown("**Campus Hackathon 2026**")
st.sidebar.caption("Author: **Mehul Goyal** (VIT)")

app_mode = st.sidebar.radio(
    "Navigation Console",
    [
        "1. Executive Overview & Live Feeds",
        "2. Core AI/NLP Risk Engine",
        "3. Module A: Tactical Index Rebalancer",
        "4. Module B: Wholesale Portfolio Stress Testing",
        "5. Interactive Ad-Hoc Scenario Sandbox"
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Live Pipeline Health")
st.sidebar.success("Ingestion Sources: Active (News + Social)")
st.sidebar.info("NLP Classification: 5 Institutional Classes")
st.sidebar.warning("Downstream Models: Module A & B Online")

# Load unified data
raw_stream = pipeline.load_unified_stream()
processed_signals = nlp.process_batch(raw_stream)

# ==============================================================================
# VIEW 1: EXECUTIVE OVERVIEW & LIVE FEEDS
# ==============================================================================
if app_mode == "1. Executive Overview & Live Feeds":
    st.title("Unified AI/NLP Financial Risk & Intelligence Platform")
    st.markdown(
        "Ingests real-time unstructured market intelligence (Financial News, Social Feeds), "
        "converts text into quantitative signals via **FinBERT Sentiment**, **Event Classification**, "
        "and **Impact Severity**, then drives tactical asset reallocation and wholesale credit stress testing."
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Ingested Records</div>
            <div class="metric-value">{}</div>
        </div>
        """.format(len(processed_signals)), unsafe_allow_html=True)

    with col2:
        systemic_count = sum(1 for s in processed_signals if s["impact_score"] >= 7)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Systemic Risk Triggers</div>
            <div class="metric-value" style="color: #FF4B4B;">{systemic_count}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        avg_sent = np.mean([s["sentiment_score"] for s in processed_signals])
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Net Market Sentiment</div>
            <div class="metric-value" style="color: {'#00AA6C' if avg_sent >= 0 else '#FF4B4B'};">{avg_sent:+.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Active Downstream Modules</div>
            <div class="metric-value" style="color: #FFD700;">Both (A & B)</div>
        </div>
        """, unsafe_allow_html=True)

    st.subheader("Real-Time Multi-Source Stream Ingestion")
    
    # Filter controls
    f_col1, f_col2 = st.columns(2)
    with f_col1:
        src_filter = st.multiselect("Filter Source Channel", ["news", "social"], default=["news", "social"])
    with f_col2:
        cat_filter = st.multiselect(
            "Filter Event Category", 
            ["Macroeconomic", "Geopolitical", "Credit Event", "Product Launch", "Merger/Acquisition"],
            default=["Macroeconomic", "Geopolitical", "Credit Event", "Product Launch", "Merger/Acquisition"]
        )

    filtered_sigs = [
        s for s in processed_signals 
        if s["source_type"] in src_filter and s["event_classification"] in cat_filter
    ]

    for item in filtered_sigs:
        badge = '<span class="badge-systemic">SYSTEMIC (IMPACT ≥ 7)</span>' if item["impact_score"] >= 7 else '<span class="badge-normal">STANDARD</span>'
        sent_color = "#00AA6C" if item["sentiment_score"] > 0 else "#FF4B4B"
        
        with st.expander(f"[{item['source_type'].upper()}] {item['headline'] or item['text_snippet'][:60]} | Impact: {item['impact_score']}/10"):
            c1, c2, c3 = st.columns([3, 1, 1])
            with c1:
                st.write(f"**Channel / Source:** {item['channel']}")
                st.write(f"**Raw Text:** {item['text_snippet']}")
                st.write(f"**Identified Entities:** `{', '.join(item['target_entities']) or 'None'}`")
            with c2:
                st.metric("Sentiment Score", f"{item['sentiment_score']:+.2f}")
                st.metric("Impact Severity", f"{item['impact_score']} / 10")
            with c3:
                st.markdown(f"**Event Type:**<br>`{item['event_classification']}`", unsafe_allow_html=True)
                st.markdown(f"**Status:** {badge}", unsafe_allow_html=True)
                st.caption(f"Model Confidence: {item['confidence']*100:.1f}%")

# ==============================================================================
# VIEW 2: CORE AI/NLP RISK ENGINE
# ==============================================================================
elif app_mode == "2. Core AI/NLP Risk Engine":
    st.title("Core AI/NLP Risk Engine: Deep Signal Diagnostics")
    st.markdown("Detailed breakdown of NLP outputs, event distributions, and impact calibration.")

    df_signals = pd.DataFrame(processed_signals)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Event Classification Breakdown")
        cat_counts = df_signals["event_classification"].value_counts().reset_index()
        cat_counts.columns = ["Event Category", "Frequency"]
        fig_cat = px.bar(
            cat_counts, x="Event Category", y="Frequency", 
            color="Event Category", 
            color_discrete_sequence=px.colors.qualitative.Bold,
            title="Distribution of Classified Institutional Events"
        )
        st.plotly_chart(fig_cat, use_container_width=True)

    with col2:
        st.subheader("Sentiment vs Impact Severity Matrix")
        fig_scatter = px.scatter(
            df_signals, 
            x="sentiment_score", 
            y="impact_score", 
            color="event_classification",
            size="impact_score",
            hover_data=["headline", "target_entities"],
            labels={"sentiment_score": "Sentiment Score (-1.0 to +1.0)", "impact_score": "Impact Score (1 to 10)"},
            title="Event Impact vs Sentiment Polarity Dispersion"
        )
        fig_scatter.add_hline(y=7, line_dash="dash", line_color="red", annotation_text="Systemic Threshold (≥ 7)")
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader("Structured Signals Feed (Machine-Readable Output Table)")
    st.dataframe(
        df_signals[[
            "signal_id", "source_type", "event_classification", "sentiment_score", 
            "impact_score", "confidence", "target_entities"
        ]],
        use_container_width=True
    )

    st.download_button(
        "Download Machine-Readable JSON Export",
        data=df_signals.to_json(orient="records", indent=2),
        file_name="structured_risk_signals.json",
        mime="application/json"
    )

# ==============================================================================
# VIEW 3: MODULE A - TACTICAL INDEX REBALANCER
# ==============================================================================
elif app_mode == "3. Module A: Tactical Index Rebalancer":
    st.title("Module A: Tactical High-Frequency Index Rebalancer")
    st.markdown(
        "Subscribes to live sentiment scores from the AI/NLP engine to dynamically adjust weights across "
        "an institutional **12-stock S&P 100 constituent basket**. Overweights positive momentum and defensively trims negative exposures."
    )

    tilt_strength = st.slider("Tactical Tilt Aggressiveness (λ Multiplier)", min_value=0.1, max_value=1.5, value=0.65, step=0.05)
    rebalancer.tilt_intensity = tilt_strength

    net_sentiments = rebalancer.aggregate_signals_by_ticker(processed_signals)
    rebal_df = rebalancer.compute_rebalanced_weights(net_sentiments)
    history_df, metrics = rebalancer.run_backtest_simulation(rebal_df)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Tactical Index Return", f"{metrics['tactical_total_return_pct']}%", f"{metrics['generated_alpha_bps']} bps Alpha")
    with m2:
        st.metric("Benchmark S&P Basket Return", f"{metrics['benchmark_total_return_pct']}%")
    with m3:
        st.metric("Tactical Sharpe Ratio", f"{metrics['sharpe_ratio']}")
    with m4:
        st.metric("Annualized Volatility", f"{metrics['tactical_annualized_vol']}%")

    col_w, col_chart = st.columns([1, 1])

    with col_w:
        st.subheader("Index Constituent Weight Adjustments")
        fig_weights = go.Figure()
        fig_weights.add_trace(go.Bar(
            name='Benchmark Weight',
            x=rebal_df['ticker'],
            y=rebal_df['benchmark_weight'] * 100,
            marker_color='#5A6E85'
        ))
        fig_weights.add_trace(go.Bar(
            name='Sentiment-Rebalanced Weight',
            x=rebal_df['ticker'],
            y=rebal_df['rebalanced_weight'] * 100,
            marker_color='#00D2FF'
        ))
        fig_weights.update_layout(
            barmode='group',
            yaxis_title="Allocation (%)",
            title="Benchmark vs Tactical Rebalanced Allocation"
        )
        st.plotly_chart(fig_weights, use_container_width=True)

    with col_chart:
        st.subheader("Historical Performance Simulation")
        fig_perf = px.line(
            history_df, 
            y=["Tactical_NLP_Index", "Benchmark_Index"],
            labels={"value": "Index Level (Base 100)", "variable": "Portfolio"},
            title="Tactical Sentiment Index vs S&P 100 Benchmark"
        )
        st.plotly_chart(fig_perf, use_container_width=True)

    st.subheader("Granular Rebalancing Ledger")
    st.dataframe(
        rebal_df[["ticker", "name", "sector", "net_sentiment", "benchmark_weight", "rebalanced_weight", "weight_delta", "allocation_status"]],
        use_container_width=True
    )

# ==============================================================================
# VIEW 4: MODULE B - STRATEGIC PORTFOLIO STRESS TESTING
# ==============================================================================
elif app_mode == "4. Module B: Wholesale Portfolio Stress Testing":
    st.title("Module B: Strategic Wholesale Banking Portfolio Stress Testing")
    st.markdown(
        "Subscribes to **Event Classification** and **Impact Score** from the AI/NLP Engine. "
        "When an event triggers systemic disruption (Impact ≥ 7), macro and credit shocks are applied across "
        "a **$505,000,000 wholesale banking asset portfolio** (Commercial Loans, Corporate Bonds, Derivatives)."
    )

    # Detect high-impact triggers
    systemic_events = [s for s in processed_signals if s["impact_score"] >= 7]
    event_options = [f"{s['event_classification']} | Impact: {s['impact_score']} | {s['headline'][:50]}" for s in systemic_events]
    event_options.append("Custom Scenario Trigger...")

    chosen_event_str = st.selectbox("Select Detected NLP Event Trigger to Simulate", event_options)

    if "Custom Scenario Trigger" in chosen_event_str:
        sel_cat = st.selectbox("Event Category", ["Geopolitical", "Macroeconomic", "Credit Event"])
        sel_impact = st.slider("Impact Score Severity", 7, 10, 8)
    else:
        idx = event_options.index(chosen_event_str)
        sel_event = systemic_events[idx]
        sel_cat = sel_event["event_classification"]
        sel_impact = sel_event["impact_score"]

    stress_results = stress_tester.run_stress_test(sel_cat, sel_impact)

    b1, b2, b3, b4 = st.columns(4)
    with b1:
        st.metric("Pre-Stress Value", f"${stress_results['total_pre_stress_mv']/1e6:.1f} M")
    with b2:
        st.metric(
            "Post-Stress Value", 
            f"${stress_results['total_post_stress_mv']/1e6:.1f} M", 
            f"-${stress_results['total_portfolio_loss_usd']/1e6:.1f} M ({stress_results['portfolio_drawdown_pct']}%)",
            delta_color="inverse"
        )
    with b3:
        st.metric("Expected Credit Loss (EL)", f"${stress_results['post_stress_expected_loss']/1e6:.2f} M", f"+{stress_results['expected_loss_surge_pct']}% surge", delta_color="inverse")
    with b4:
        st.metric("Shock Severity Multiplier", f"{stress_results['scaling_factor']}x")

    st.subheader(f"Scenario Transmission: {stress_results['scenario_name']}")

    detailed_df = stress_results["detailed_asset_impact"]

    col_asset, col_waterfall = st.columns(2)
    with col_asset:
        loss_by_type = detailed_df.groupby("asset_type")["valuation_loss_usd"].sum().reset_index()
        fig_donut = px.pie(
            loss_by_type, 
            names="asset_type", 
            values="valuation_loss_usd", 
            hole=0.45,
            title="Share of Capital Loss by Wholesale Asset Class",
            color_discrete_sequence=["#FF4B4B", "#FFA726", "#AB47BC"]
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_waterfall:
        fig_bar = px.bar(
            detailed_df.sort_values(by="loss_pct", ascending=False),
            x="facility_name",
            y="loss_pct",
            color="borrower_sector",
            title="Individual Asset Valuation Haircut (%)",
            labels={"loss_pct": "Valuation Haircut (%)", "facility_name": "Facility"}
        )
        fig_bar.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Granular Basel Credit Risk Ledger Before & After Stress")
    st.dataframe(
        detailed_df[[
            "asset_id", "facility_name", "asset_type", "borrower_sector", "notional_usd",
            "baseline_pd_pct", "stressed_pd_pct", "baseline_mv_usd", "stressed_mv_usd",
            "valuation_loss_usd", "loss_pct"
        ]],
        use_container_width=True
    )

# ==============================================================================
# VIEW 5: INTERACTIVE AD-HOC SCENARIO SANDBOX
# ==============================================================================
elif app_mode == "5. Interactive Ad-Hoc Scenario Sandbox":
    st.title("Ad-Hoc Live News & Market Sentiment Sandbox")
    st.markdown("Test the unified pipeline by typing or pasting any custom breaking financial news headline or tweet.")

    custom_input = st.text_area(
        "Enter Breaking Financial Event or Social Post:",
        value="Middle Eastern export terminal faces maritime drone disruption; crude futures surge $12 while global sovereign debt spreads widen by 85bps as inflation fears mount.",
        height=100
    )

    if st.button("Run Unified AI Risk Signal Analysis", type="primary"):
        record = {
            "id": f"LIVE-{int(datetime.utcnow().timestamp())}",
            "source_type": "adhoc_input",
            "channel": "Live Jury Tester",
            "raw_text": custom_input
        }
        res = nlp.process_record(record)

        st.success("NLP Extraction Completed Successfully!")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Sentiment Score", f"{res['sentiment_score']:+.2f}")
        with c2:
            st.metric("Event Category", res['event_classification'])
        with c3:
            st.metric("Impact Score", f"{res['impact_score']} / 10")
        with c4:
            st.metric("Systemic Disruption Trigger", "YES (≥ 7)" if res['impact_score'] >= 7 else "NO (< 7)")

        st.subheader("Automatic Downstream Action Triggers")
        t_col1, t_col2 = st.columns(2)
        with t_col1:
            st.markdown("#### Module A Response:")
            if res["target_entities"]:
                st.info(f"Target tickers identified: {res['target_entities']}. Reallocating weights with net sentiment {res['sentiment_score']:+.2f}...")
            else:
                st.warning("No specific single equity ticker named. General macro sentiment shift applied across entire index universe.")

        with t_col2:
            st.markdown("#### Module B Response:")
            if res["impact_score"] >= 7:
                st.error(f"SYSTEMIC STRESS TRIGGERED! Shock parameters queued for scenario: {res['event_classification']}.")
            else:
                st.success("Impact score is below systemic threshold (< 7). No emergency credit provisions required.")
