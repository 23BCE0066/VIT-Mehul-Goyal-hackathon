"""
Comprehensive Verification Test Suite
S&P Global & CRISIL Campus Hackathon 2026
Candidate: Mehul Goyal (VIT)
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.engine.ingestion import IngestionPipeline
from src.engine.nlp_processor import NLPRiskEngine
from src.module_a.universe import INDEX_UNIVERSE
from src.module_a.index_rebalancer import TacticalIndexRebalancer
from src.module_b.synthetic_book import WholesalePortfolioBook
from src.module_b.stress_engine import StrategicStressTester


class TestFinancialRiskPlatform(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.pipeline = IngestionPipeline()
        cls.nlp = NLPRiskEngine()
        cls.stream = cls.pipeline.load_unified_stream()
        cls.signals = cls.nlp.process_batch(cls.stream)
        cls.rebalancer = TacticalIndexRebalancer()
        cls.book = WholesalePortfolioBook()
        cls.stress_tester = StrategicStressTester(cls.book)

    def test_01_ingestion_multi_channel(self):
        """Verify multi-channel feed ingestion produces valid records."""
        self.assertGreaterEqual(len(self.stream), 12)
        source_types = {item.get("source_type") for item in self.stream}
        self.assertIn("news", source_types)
        self.assertIn("social", source_types)

    def test_02_nlp_signal_schema_and_bounds(self):
        """Verify NLP engine outputs required fields and strict bounds."""
        valid_categories = {
            "Geopolitical", "Macroeconomic", "Credit Event", 
            "Merger/Acquisition", "Product Launch"
        }
        for sig in self.signals:
            # 1. Sentiment bounded in [-1.0, 1.0]
            self.assertGreaterEqual(sig["sentiment_score"], -1.0)
            self.assertLessEqual(sig["sentiment_score"], 1.0)
            # 2. Event category in 5 taxonomies
            self.assertIn(sig["event_classification"], valid_categories)
            # 3. Impact score in [1, 10]
            self.assertGreaterEqual(sig["impact_score"], 1)
            self.assertLessEqual(sig["impact_score"], 10)

    def test_03_module_a_box_constraints(self):
        """Verify Module A weight allocation sums to 1.0 and satisfies [2%, 22%] box constraints."""
        net_sent = self.rebalancer.aggregate_signals_by_ticker(self.signals)
        weights_df = self.rebalancer.compute_rebalanced_weights(net_sent)
        weights = weights_df["rebalanced_weight"].values
        
        self.assertEqual(len(weights), len(INDEX_UNIVERSE))
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=3)
        self.assertTrue((weights >= 0.02 - 1e-5).all(), "Weight fell below 2% floor")
        self.assertTrue((weights <= 0.22 + 1e-5).all(), "Weight exceeded 22% cap")

    def test_04_module_b_stress_mechanics(self):
        """Verify Module B Basel Expected Loss and drawdown calculation."""
        res = self.stress_tester.run_stress_test("Geopolitical", impact_score=8)
        self.assertGreater(res["total_pre_stress_mv"], 0)
        self.assertGreater(res["total_post_stress_mv"], 0)
        self.assertGreater(res["total_pre_stress_mv"], res["total_post_stress_mv"])
        self.assertGreater(res["post_stress_expected_loss"], res["pre_stress_expected_loss"])
        self.assertGreater(res["expected_loss_surge_pct"], 0.0)

    def test_05_systemic_risk_trigger(self):
        """Verify systemic event trigger correctly trips on impact >= 7."""
        systemic_signals = [s for s in self.signals if s["is_systemic_event"]]
        self.assertGreater(len(systemic_signals), 0)
        for s in systemic_signals:
            self.assertGreaterEqual(s["impact_score"], 7)


if __name__ == "__main__":
    unittest.main()
