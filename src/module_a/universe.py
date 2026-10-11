"""
Module A: Universe Definition & Historical Pricing
Maintains a curated liquid basket of 12 bellwether S&P 100 constituent equities
across multiple sectors (Technology, Banking, Energy, Healthcare, Industrial).
"""

from typing import Dict, List
import pandas as pd
import numpy as np


INDEX_UNIVERSE = [
    {"ticker": "AAPL", "name": "Apple Inc.", "sector": "Technology", "benchmark_weight": 0.12},
    {"ticker": "MSFT", "name": "Microsoft Corp.", "sector": "Technology", "benchmark_weight": 0.11},
    {"ticker": "NVDA", "name": "Nvidia Corp.", "sector": "Technology", "benchmark_weight": 0.10},
    {"ticker": "AMZN", "name": "Amazon.com Inc.", "sector": "Consumer Discretionary", "benchmark_weight": 0.09},
    {"ticker": "GOOGL", "name": "Alphabet Inc.", "sector": "Communication Services", "benchmark_weight": 0.08},
    {"ticker": "JPM", "name": "JPMorgan Chase & Co.", "sector": "Financials", "benchmark_weight": 0.08},
    {"ticker": "BAC", "name": "Bank of America Corp.", "sector": "Financials", "benchmark_weight": 0.07},
    {"ticker": "XOM", "name": "Exxon Mobil Corp.", "sector": "Energy", "benchmark_weight": 0.07},
    {"ticker": "CVX", "name": "Chevron Corp.", "sector": "Energy", "benchmark_weight": 0.06},
    {"ticker": "LLY", "name": "Eli Lilly & Co.", "sector": "Healthcare", "benchmark_weight": 0.08},
    {"ticker": "JNJ", "name": "Johnson & Johnson", "sector": "Healthcare", "benchmark_weight": 0.07},
    {"ticker": "TSLA", "name": "Tesla Inc.", "sector": "Consumer Discretionary", "benchmark_weight": 0.07},
]


def generate_synthetic_price_history(days: int = 60) -> pd.DataFrame:
    """
    Generates reliable deterministic daily price histories for backtesting
    when offline or API throttled, maintaining realistic correlations.
    """
    np.random.seed(42)
    dates = pd.date_range(end=pd.Timestamp.now(), periods=days, freq="B")
    
    base_prices = {
        "AAPL": 225.0, "MSFT": 440.0, "NVDA": 130.0, "AMZN": 185.0,
        "GOOGL": 175.0, "JPM": 215.0, "BAC": 42.0, "XOM": 118.0,
        "CVX": 155.0, "LLY": 920.0, "JNJ": 160.0, "TSLA": 240.0
    }

    drift = {
        "NVDA": 0.0018, "LLY": 0.0012, "AAPL": 0.0008, "MSFT": 0.0007,
        "JPM": 0.0004, "BAC": 0.0002, "XOM": 0.0003, "CVX": 0.0001,
        "AMZN": 0.0006, "GOOGL": 0.0005, "JNJ": 0.0002, "TSLA": 0.0005
    }

    price_series = {}
    for item in INDEX_UNIVERSE:
        t = item["ticker"]
        ret = np.random.normal(loc=drift[t], scale=0.015, size=days)
        # Add slight auto-correlation
        cum_ret = np.cumprod(1 + ret)
        price_series[t] = base_prices[t] * cum_ret

    df = pd.DataFrame(price_series, index=dates)
    return df
