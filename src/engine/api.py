"""
FastAPI Microservice for AI/NLP Risk Signals
Exposes RESTful endpoints for downstream application consumption:
- GET /health
- GET /signals (Live & Cached structured risk signals)
- POST /analyze (Direct ad-hoc text risk extraction)
- GET /signals/systemic (High-impact events where Impact Score >= 7)
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.engine.ingestion import IngestionPipeline
from src.engine.nlp_processor import NLPRiskEngine

app = FastAPI(
    title="S&P Global & CRISIL AI/NLP Risk Engine API",
    description="Institutional-grade unstructured risk signal processing platform",
    version="1.0.0"
)

pipeline = IngestionPipeline()
nlp_engine = NLPRiskEngine()


class TextAnalysisRequest(BaseModel):
    text: str
    entities: Optional[List[str]] = None
    source: Optional[str] = "Ad-hoc User Submission"


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "RiskEngine_v1", "timestamp": "2026-10-11T14:30:00Z"}


@app.get("/signals")
def get_all_signals():
    """Returns structured risk signals across all ingested streams."""
    stream = pipeline.load_unified_stream()
    signals = nlp_engine.process_batch(stream)
    return {"count": len(signals), "signals": signals}


@app.get("/signals/systemic")
def get_systemic_signals(min_impact: int = 7):
    """Filters signals with severe market impact (Impact >= threshold)."""
    stream = pipeline.load_unified_stream()
    signals = nlp_engine.process_batch(stream)
    filtered = [s for s in signals if s["impact_score"] >= min_impact]
    return {"threshold": min_impact, "count": len(filtered), "systemic_signals": filtered}


@app.post("/analyze")
def analyze_text(request: TextAnalysisRequest):
    """Ad-hoc single text inference endpoint."""
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text field cannot be empty.")
    
    record = {
        "id": "API-ADHOC",
        "source_type": "api_request",
        "channel": request.source,
        "raw_text": request.text,
        "target_entities": request.entities or []
    }
    return nlp_engine.process_record(record)
