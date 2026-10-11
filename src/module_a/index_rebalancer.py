"""
Module A: Tactical High-Frequency Index Rebalancer
Subscribes to Sentiment Scores from the AI/NLP Risk Engine and dynamically
reweights the stock index constituents:
- Positive sentiment: Overweight target stock
- Negative sentiment: Underweight target stock
- Enforces institutional risk constraints:
  * Minimum single-stock weight floor (e.g. 2%)
  * Maximum single-stock weight cap (e.g. 22%)
  * Total weights strictly sum to 100%
- Computes tactical alpha & backtested performance vs Benchmark.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from src.module_a.universe import INDEX_UNIVERSE, generate_synthetic_price_history


class TacticalIndexRebalancer:
    def __init__(self, tilt_intensity: float = 0.65, min_weight: float = 0.02, max_weight: float = 0.22):
        self.tilt_intensity = tilt_intensity
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.universe = INDEX_UNIVERSE

    def aggregate_signals_by_ticker(self, risk_signals: List[Dict]) -> Dict[str, float]:
        """
        Combines multiple incoming NLP signals into a single net sentiment score per ticker.
        Weights higher-impact signals more heavily.
        """
        ticker_sentiment_accum: Dict[str, float] = {item["ticker"]: 0.0 for item in self.universe}
        ticker_weight_accum: Dict[str, float] = {item["ticker"]: 0.0 for item in self.universe}

        for sig in risk_signals:
            sentiment = sig.get("sentiment_score", 0.0)
            impact = sig.get("impact_score", 5)
            weight = 1.0 + (impact / 10.0)

            for entity in sig.get("target_entities", []):
                clean_ticker = entity.replace("$", "").upper()
                if clean_ticker in ticker_sentiment_accum:
                    ticker_sentiment_accum[clean_ticker] += sentiment * weight
                    ticker_weight_accum[clean_ticker] += weight

        net_sentiments = {}
        for ticker in ticker_sentiment_accum:
            if ticker_weight_accum[ticker] > 0:
                net_sentiments[ticker] = ticker_sentiment_accum[ticker] / ticker_weight_accum[ticker]
            else:
                net_sentiments[ticker] = 0.0  # Neutral baseline
        return net_sentiments

    def compute_rebalanced_weights(self, net_sentiments: Dict[str, float]) -> pd.DataFrame:
        """
        Calculates new tactical portfolio weights based on net sentiment tilts.
        Applies bounding box constraints and normalizes to exactly 1.0.
        """
        records = []
        raw_weights = []

        for item in self.universe:
            ticker = item["ticker"]
            base_w = item["benchmark_weight"]
            sent = net_sentiments.get(ticker, 0.0)

            # Quadratic or linear tilt multiplier
            tilt_factor = 1.0 + (self.tilt_intensity * sent)
            raw_w = max(self.min_weight, base_w * tilt_factor)
            raw_weights.append(raw_w)
            records.append({
                "ticker": ticker,
                "name": item["name"],
                "sector": item["sector"],
                "benchmark_weight": base_w,
                "net_sentiment": round(sent, 3),
                "raw_weight": raw_w
            })

        # Iterative proportional fitting for min/max box constraints
        df = pd.DataFrame(records)
        weights = np.array(df["raw_weight"].values, dtype=float)

        # Normalize to sum 1.0
        weights = weights / weights.sum()

        # Clip and re-normalize until convergence
        for _ in range(5):
            weights = np.clip(weights, self.min_weight, self.max_weight)
            weights = weights / weights.sum()

        weights = np.round(weights, 4)
        # Ensure exact sum to 1.0
        weights[-1] += round(1.0 - weights.sum(), 4)
        df["rebalanced_weight"] = weights
        df["weight_delta"] = np.round(df["rebalanced_weight"] - df["benchmark_weight"], 4)
        df["allocation_status"] = df["weight_delta"].apply(
            lambda d: "OVERWEIGHT" if d > 0.005 else ("UNDERWEIGHT" if d < -0.005 else "NEUTRAL")
        )

        return df.sort_values(by="rebalanced_weight", ascending=False)

    def run_backtest_simulation(self, rebalanced_weights_df: pd.DataFrame, days: int = 45) -> Tuple[pd.DataFrame, Dict]:
        """
        Simulates cumulative portfolio performance of Tactical Sentiment-Rebalanced Index
        versus Static Benchmark S&P 100 basket.
        """
        prices = generate_synthetic_price_history(days=days)
        returns = prices.pct_change().dropna()

        bench_w = {row["ticker"]: row["benchmark_weight"] for _, row in rebalanced_weights_df.iterrows()}
        tactical_w = {row["ticker"]: row["rebalanced_weight"] for _, row in rebalanced_weights_df.iterrows()}

        # Series of daily returns
        bench_daily = sum(returns[t] * bench_w.get(t, 0) for t in returns.columns)
        tactical_daily = sum(returns[t] * tactical_w.get(t, 0) for t in returns.columns)

        cum_bench = (1 + bench_daily).cumprod()
        cum_tactical = (1 + tactical_daily).cumprod()

        history_df = pd.DataFrame({
            "Date": cum_bench.index,
            "Benchmark_Index": cum_bench.values * 100.0,
            "Tactical_NLP_Index": cum_tactical.values * 100.0,
            "Active_Alpha_Spread": (cum_tactical.values - cum_bench.values) * 100.0
        }).set_index("Date")

        total_bench_ret = float(cum_bench.iloc[-1] - 1.0)
        total_tact_ret = float(cum_tactical.iloc[-1] - 1.0)
        alpha = total_tact_ret - total_bench_ret

        vol_bench = float(bench_daily.std() * np.sqrt(252))
        vol_tact = float(tactical_daily.std() * np.sqrt(252))
        sharpe_tact = float((tactical_daily.mean() * 252 - 0.045) / vol_tact) if vol_tact > 0 else 0.0

        metrics = {
            "tactical_total_return_pct": round(total_tact_ret * 100, 2),
            "benchmark_total_return_pct": round(total_bench_ret * 100, 2),
            "generated_alpha_bps": round(alpha * 10000, 1),
            "tactical_annualized_vol": round(vol_tact * 100, 2),
            "sharpe_ratio": round(sharpe_tact, 2)
        }

        return history_df, metrics


if __name__ == "__main__":
    from src.engine.ingestion import IngestionPipeline
    from src.engine.nlp_processor import NLPRiskEngine

    pipeline = IngestionPipeline()
    nlp = NLPRiskEngine()
    signals = nlp.process_batch(pipeline.load_unified_stream())

    rebalancer = TacticalIndexRebalancer()
    sentiments = rebalancer.aggregate_signals_by_ticker(signals)
    weights_df = rebalancer.compute_rebalanced_weights(sentiments)
    history, metrics = rebalancer.run_backtest_simulation(weights_df)

    print("\n--- REBALANCED WEIGHTS ---")
    print(weights_df[["ticker", "benchmark_weight", "rebalanced_weight", "weight_delta", "allocation_status"]].to_string())
    print("\n--- PERFORMANCE METRICS ---")
    print(metrics)
