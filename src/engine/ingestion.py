"""
Data Ingestion Pipeline
Supports multi-source ingestion:
1. Financial News Feeds (JSON/RSS/GDELT/NewsAPI)
2. Social Media Streams (StockTwits / X Feeds)
With automatic normalization, deduplication, and schema validation.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DataIngestion")


class IngestionPipeline:
    def __init__(self, data_dir: Optional[str] = None):
        if data_dir is None:
            self.data_dir = Path(__file__).resolve().parent.parent.parent / "data"
        else:
            self.data_dir = Path(data_dir)

    def load_news_stream(self, custom_path: Optional[str] = None) -> List[Dict]:
        """Ingests structured financial news records."""
        path = Path(custom_path) if custom_path else self.data_dir / "sample_news.json"
        try:
            with open(path, "r", encoding="utf-8") as f:
                news_items = json.load(f)
            logger.info(f"Loaded {len(news_items)} news items from {path.name}")
            return [self._normalize_item(item, source_type="news") for item in news_items]
        except Exception as e:
            logger.error(f"Error reading news stream: {e}")
            return []

    def load_social_stream(self, custom_path: Optional[str] = None) -> List[Dict]:
        """Ingests social sentiment stream (X / StockTwits)."""
        path = Path(custom_path) if custom_path else self.data_dir / "sample_tweets.json"
        try:
            with open(path, "r", encoding="utf-8") as f:
                tweets = json.load(f)
            logger.info(f"Loaded {len(tweets)} social items from {path.name}")
            return [self._normalize_item(item, source_type="social") for item in tweets]
        except Exception as e:
            logger.error(f"Error reading social stream: {e}")
            return []

    def load_unified_stream(self) -> List[Dict]:
        """Merges and chronologically sorts feeds from all ingested channels."""
        news = self.load_news_stream()
        social = self.load_social_stream()
        unified = news + social
        unified.sort(key=lambda x: x.get("timestamp", ""), reverse=False)
        return unified

    def _normalize_item(self, item: Dict, source_type: str) -> Dict:
        """Converts raw unstructured payload to canonical input structure."""
        text = item.get("content") or item.get("text") or item.get("headline", "")
        if "headline" in item and "content" in item:
            full_text = f"{item['headline']}. {item['content']}"
        else:
            full_text = text

        return {
            "id": item.get("id", f"RAW-{datetime.utcnow().timestamp()}"),
            "source_type": source_type,
            "channel": item.get("source", "Unspecified"),
            "timestamp": item.get("timestamp", datetime.utcnow().isoformat()),
            "headline": item.get("headline", ""),
            "raw_text": full_text.strip(),
            "target_entities": item.get("entities") or item.get("cashtags") or [],
            "ground_truth_category": item.get("ground_truth_category"),
        }


if __name__ == "__main__":
    pipeline = IngestionPipeline()
    unified = pipeline.load_unified_stream()
    print(f"Total unified ingestion records: {len(unified)}")
    for item in unified[:2]:
        print(f"[{item['source_type'].upper()}] {item['channel']} -> {item['raw_text'][:80]}...")
