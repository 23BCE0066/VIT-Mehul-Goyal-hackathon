"""
AI/NLP Risk Engine
Performs financial-domain NLP extraction:
1. Sentiment Score: Polarity from -1.0 to +1.0 (calibrated for capital markets)
2. Event Classification: Categorizes events into institutional taxonomies
   (Geopolitical, Macroeconomic, Credit Event, Merger/Acquisition, Product Launch)
3. Impact Score: Severity scale 1 - 10 indicating potential market/credit disruption
"""

import math
import re
from typing import Dict, List, Optional


class NLPRiskEngine:
    def __init__(self, use_deep_model: bool = False):
        self.use_deep_model = use_deep_model
        self.pipeline = None

        # Financial lexicon and weights tuned to S&P/CRISIL risk conventions
        self.event_taxonomies = {
            "Geopolitical": [
                "war", "military", "sanction", "missile", "strait", "hormuz", "tensions", 
                "blockade", "geopolitical", "treaty", "conflict", "retaliation", "navy"
            ],
            "Macroeconomic": [
                "inflation", "federal reserve", "fed", "rate hike", "basis points", "bps", 
                "yield curve", "recession", "gdp", "cpi", "central bank", "liquidity", "unemployment"
            ],
            "Credit Event": [
                "default", "covenant", "insolvency", "bankruptcy", "cds", "downgrade", 
                "subordinated", "mezzanine", "non-performing", "debt facility", "liquidation", "distressed"
            ],
            "Merger/Acquisition": [
                "acquisition", "buyout", "merger", "takeover", "tender offer", 
                "definitive agreement", "cash-and-stock", "synergy", "antitrust", "consolidation"
            ],
            "Product Launch": [
                "launch", "architecture", "breakthrough", "fast-track", "fda approval", 
                "next-generation", "patent", "production yield", "clinical phase", "rollout"
            ]
        }

        self.financial_positives = {
            "surge": 0.8, "record": 0.75, "breakthrough": 0.9, "outperform": 0.85, 
            "approval": 0.8, "upgrade": 0.75, "bullish": 0.7, "backlog": 0.6,
            "accelerated": 0.5, "superior": 0.7, "rally": 0.75, "beat": 0.65,
            "accretive": 0.7, "expansion": 0.6, "growth": 0.55, "synergy": 0.65
        }

        self.financial_negatives = {
            "default": -0.95, "hike": -0.5, "tightening": -0.6, "tensions": -0.7,
            "shock": -0.8, "recession": -0.85, "downgrade": -0.8, "liquidation": -0.9,
            "covenant": -0.65, "plunge": -0.8, "drop": -0.6, "halted": -0.7,
            "distressed": -0.85, "inflationary": -0.6, "warning": -0.6, "loss": -0.7,
            "uninvestable": -0.8, "pressure": -0.55, "collapse": -0.95
        }

        # Known ticker entity mappings for targeted signal assignment
        self.ticker_regexes = {
            "AAPL": r"\b(Apple|AAPL|\$AAPL)\b",
            "MSFT": r"\b(Microsoft|MSFT|\$MSFT)\b",
            "NVDA": r"\b(Nvidia|NVDA|\$NVDA)\b",
            "AMZN": r"\b(Amazon|AMZN|\$AMZN)\b",
            "GOOGL": r"\b(Google|Alphabet|GOOGL|\$GOOGL)\b",
            "JPM": r"\b(JPMorgan|JPM|\$JPM|Chase)\b",
            "BAC": r"\b(Bank of America|BAC|\$BAC)\b",
            "XOM": r"\b(Exxon|ExxonMobil|XOM|\$XOM)\b",
            "CVX": r"\b(Chevron|CVX|\$CVX)\b",
            "LLY": r"\b(Eli Lilly|Lilly|LLY|\$LLY)\b",
            "DAL": r"\b(Delta|DAL|\$DAL)\b",
            "TSLA": r"\b(Tesla|TSLA|\$TSLA)\b"
        }

    def extract_entities(self, text: str) -> List[str]:
        """Identifies relevant tickers from text."""
        detected = []
        for ticker, pattern in self.ticker_regexes.items():
            if re.search(pattern, text, re.IGNORECASE):
                detected.append(ticker)
        return list(set(detected))

    def classify_event(self, text: str) -> Dict[str, float]:
        """
        Classifies financial event into one of the key taxonomies
        using multi-feature keyword density and normalized softmax scoring.
        """
        text_lower = text.lower()
        scores = {}
        for category, keywords in self.event_taxonomies.items():
            match_count = sum(1 for kw in keywords if kw in text_lower)
            scores[category] = match_count

        max_matches = max(scores.values()) if scores and max(scores.values()) > 0 else 1
        
        # Calculate distribution
        exp_scores = {k: math.exp(v / 1.5) for k, v in scores.items()}
        sum_exp = sum(exp_scores.values())
        probs = {k: round(v / sum_exp, 3) for k, v in exp_scores.items()}
        
        best_category = max(scores, key=scores.get)
        if scores[best_category] == 0:
            best_category = "Macroeconomic" # fallback institutional default

        return {
            "primary_category": best_category,
            "confidence": probs.get(best_category, 0.70),
            "class_distribution": probs
        }

    def compute_sentiment(self, text: str) -> float:
        """
        Calculates sentiment polarity score bounded strictly within [-1.0, 1.0].
        Incorporates financial negation context and intensity.
        """
        text_lower = text.lower()
        tokens = re.findall(r"\b\w+\b", text_lower)
        
        pos_score = 0.0
        neg_score = 0.0

        for i, token in enumerate(tokens):
            weight_mult = 1.0
            # Lookback for negation
            if i > 0 and tokens[i - 1] in ["no", "not", "never", "without", "failed"]:
                weight_mult = -1.0
            if i > 1 and tokens[i - 2] in ["no", "not", "never", "without", "failed"]:
                weight_mult = -1.0

            if token in self.financial_positives:
                score = self.financial_positives[token] * weight_mult
                if score >= 0:
                    pos_score += score
                else:
                    neg_score += abs(score)
            elif token in self.financial_negatives:
                score = self.financial_negatives[token] * weight_mult
                if score < 0:
                    neg_score += abs(score)
                else:
                    pos_score += score

        raw_delta = pos_score - neg_score
        # Tanh scaling to gracefully bound in [-1.0, 1.0]
        scaled_sentiment = math.tanh(raw_delta / 1.8)
        return round(float(scaled_sentiment), 3)

    def calculate_impact_score(self, sentiment: float, category: str, text: str, entities: List[str]) -> int:
        """
        Determines severity rating from 1 (negligible) to 10 (systemic disruption).
        Factors:
        - Sentiment polarity extremity (|sentiment|)
        - Systemic nature of category (e.g. Geopolitical/Credit > Single Product Launch)
        - Magnitude keywords ($ billions, % surges, systemic defaults)
        - Number of market entities impacted
        """
        base_severity = {
            "Geopolitical": 7.5,
            "Credit Event": 8.0,
            "Macroeconomic": 6.5,
            "Merger/Acquisition": 5.0,
            "Product Launch": 4.5
        }.get(category, 5.0)

        # Polarity multiplier
        polarity_boost = abs(sentiment) * 2.2

        # Magnitude signals
        text_lower = text.lower()
        magnitude_boost = 0.0
        if re.search(r"(\$\d+(\.\d+)?b|billion|trillion)", text_lower):
            magnitude_boost += 1.5
        if re.search(r"(\b50bps\b|\b100bps\b|2%|8%|sharp|massive|severe)", text_lower):
            magnitude_boost += 1.0
        if len(entities) >= 3:
            magnitude_boost += 0.8

        final_raw = (base_severity * 0.45) + polarity_boost + magnitude_boost
        impact_score = int(round(min(max(final_raw, 1.0), 10.0)))
        return impact_score

    def process_record(self, record: Dict) -> Dict:
        """
        Full unified analysis for an ingested record.
        Outputs structured machine-readable risk signal schema.
        """
        raw_text = record.get("raw_text", "")
        entities = record.get("target_entities", [])
        if not entities:
            entities = self.extract_entities(raw_text)

        sentiment_score = self.compute_sentiment(raw_text)
        cat_meta = self.classify_event(raw_text)
        event_category = cat_meta["primary_category"]
        impact_score = self.calculate_impact_score(sentiment_score, event_category, raw_text, entities)

        return {
            "signal_id": f"SIG-{record.get('id', 'N/A')}",
            "raw_id": record.get("id"),
            "source_type": record.get("source_type", "unknown"),
            "channel": record.get("channel", "unknown"),
            "timestamp": record.get("timestamp"),
            "headline": record.get("headline", ""),
            "text_snippet": raw_text[:140] + ("..." if len(raw_text) > 140 else ""),
            "target_entities": entities,
            "sentiment_score": sentiment_score,      # Required field [-1.0, 1.0]
            "event_classification": event_category,   # Required field
            "impact_score": impact_score,             # Required field [1, 10]
            "confidence": cat_meta["confidence"],
            "class_distribution": cat_meta["class_distribution"],
            "is_systemic_event": impact_score >= 7
        }

    def process_batch(self, records: List[Dict]) -> List[Dict]:
        """Processes an array of ingested records."""
        return [self.process_record(r) for r in records]


if __name__ == "__main__":
    from ingestion import IngestionPipeline
    pipeline = IngestionPipeline()
    engine = NLPRiskEngine()
    
    stream = pipeline.load_unified_stream()
    signals = engine.process_batch(stream)
    
    print("\n--- SAMPLE EXTRACTED RISK SIGNALS ---")
    for s in signals[:3]:
        print(f"ID: {s['signal_id']} | Sentiment: {s['sentiment_score']} | Cat: {s['event_classification']} | Impact: {s['impact_score']}/10 | Entities: {s['target_entities']}")
