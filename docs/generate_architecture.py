"""
Script to generate high-resolution architecture diagram
docs/architecture.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(15, 9), dpi=300)
ax.set_facecolor("#0F172A")
fig.patch.set_facecolor("#0F172A")

# Title & Subtitle
plt.text(7.5, 8.6, "S&P Global & CRISIL Campus Hackathon 2026", 
         fontsize=17, fontweight="bold", color="#38BDF8", ha="center")
plt.text(7.5, 8.2, "Unified AI/NLP Financial Risk Engine & Downstream Capital Analytics Architecture", 
         fontsize=12, color="#94A3B8", ha="center")

def draw_box(x, y, w, h, title, subtitle, items, color, bg="#1E293B"):
    # Box rectangle
    rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1,rounding_size=0.15",
                                 facecolor=bg, edgecolor=color, linewidth=2)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h - 0.35, title, fontsize=11, fontweight="bold", color=color, ha="center")
    if subtitle:
        ax.text(x + w/2, y + h - 0.65, subtitle, fontsize=8.5, color="#94A3B8", ha="center")
    
    start_y = y + h - 1.05
    for i, itm in enumerate(items):
        ax.text(x + 0.25, start_y - (i * 0.36), f"• {itm}", fontsize=8.5, color="#E2E8F0")

# 1. Multi-source Ingestion
draw_box(0.5, 4.4, 3.4, 3.2, 
         "1. MULTI-SOURCE INGESTION", "Real-Time Unstructured Feeds",
         ["Financial News Wire (GDELT, NewsAPI)", "Social Sentiment Stream (StockTwits, X)", "Text Normalization & Deduplication", "Entity & Cashtag Extraction Engine"],
         "#38BDF8")

# 2. Core AI/NLP Risk Engine
draw_box(4.5, 4.2, 4.6, 3.6,
         "2. CORE AI/NLP RISK ENGINE", "Institutional Risk Extraction",
         ["FinBERT Sentiment Polarity (-1.0 to +1.0)", "Taxonomy Classifier (5 Event Categories)", "Impact Severity Scoring (Scale 1 to 10)", "Explainability & Confidence Scoring", "RESTful FastAPI Microservice (/signals)"],
         "#F59E0B")

# 3. Downstream Module A
draw_box(9.8, 5.6, 4.7, 2.5,
         "3. MODULE A: TACTICAL REBALANCER", "High-Frequency Equity Index Tilt",
         ["S&P 100 12-Stock Liquid Universe", "Sentiment Alpha Tilting Algorithm", "Weight Capping & Normalization Box", "Cumulative PnL vs Buy & Hold Backtest"],
         "#10B981")

# 4. Downstream Module B
draw_box(9.8, 2.4, 4.7, 2.7,
         "4. MODULE B: STRESS TESTER", "Wholesale Banking Asset Simulation",
         ["$505M Synthetic Banking Book (Loans/Bonds)", "Systemic Trigger Detector (Impact >= 7)", "Macro & Credit Shock Matrix (Basel)", "Expected Loss (PD x LGD) & Haircuts"],
         "#EF4444")

# 5. Interactive Dashboard & Output
draw_box(3.5, 0.4, 7.5, 1.6,
         "5. UNIFIED INTERACTIVE RISK PLATFORM (STREAMLIT & REST API)", "Real-time Monitoring & Ad-Hoc Scenario Sandbox",
         ["Multi-Tab Interactive Dashboards with Plotly Visuals", "Live Ad-Hoc Breaking News Inference Sandbox & JSON Signal Export"],
         "#818CF8")

# Arrows
def draw_arrow(x1, y1, x2, y2, color="#94A3B8"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->,head_width=0.4,head_length=0.6", color=color, lw=2))

draw_arrow(3.9, 5.9, 4.5, 5.9, "#38BDF8")
draw_arrow(9.1, 6.7, 9.8, 6.7, "#10B981")
draw_arrow(9.1, 4.9, 9.8, 3.9, "#EF4444")
draw_arrow(6.8, 4.2, 6.8, 2.0, "#818CF8")
draw_arrow(12.1, 5.6, 10.5, 2.0, "#818CF8")
draw_arrow(12.1, 2.4, 10.5, 2.0, "#818CF8")

ax.set_xlim(0, 15)
ax.set_ylim(0, 9)
ax.axis("off")

plt.tight_layout()
plt.savefig("docs/architecture.png", dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
print("Architecture diagram saved to docs/architecture.png")
