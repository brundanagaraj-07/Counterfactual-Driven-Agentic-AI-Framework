"""
layer2_nlp.py — Layer 2: NLP Pre-Processing (HuggingFace Transformers)
=======================================================================
Architecture (from design):
  ┌─────────────────────────────────────────────────────────────┐
  │                  EnrichedRiskInputBundle                    │
  │   sentiment_scores · NER entities · geo_tags · reliability  │
  └─────────────────────────────────────────────────────────────┘
        ▲                  ▲                    ▲
   FinBERT          Twitter-RoBERTa         BERT-NER
   ProsusAI/finbert  cardiffnlp model     dslim/bert-base-NER
   Financial sent.   Social sentiment     Entity extraction
 
Input  : RiskInputBundle  (risk_input_bundle.json from Layer 1)
Output : EnrichedRiskInputBundle (enriched_risk_bundle.json → Layer 3)
 
Pipeline steps:
  1. Load RiskInputBundle from JSON (or accept object directly)
  2. For each text source decide which model(s) to run:
       News / Port / Commodity   → FinBERT (financial sentiment)
       Social signals            → Twitter-RoBERTa (social sentiment)
       All text                  → BERT-NER (entity + geo extraction)
  3. Aggregate scores, attach to enriched schemas
  4. Compute per-item reliability weights (source credibility heuristic)
  5. Persist EnrichedRiskInputBundle to JSON
 
Usage:
    python layer2_nlp.py                        # reads risk_input_bundle.json
    python layer2_nlp.py --input custom.json    # custom input path
    python layer2_nlp.py --output enriched.json # custom output path
"""
 
from __future__ import annotations
 
import os
import json
import argparse
import warnings
import logging
from datetime import datetime
from typing import Optional
 
import torch
from transformers import pipeline, Pipeline
 
from schemas_layer2 import (
    SentimentScore,
    NEREntity,
    EnrichedNewsItem,
    EnrichedSocialSignal,
    EnrichedStockSignal,
    EnrichedPortSignal,
    EnrichedWeatherSignal,
    EnrichedCommoditySignal,
    EnrichedRiskInputBundle,
)
 
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
 
# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
 
FINBERT_MODEL        = "ProsusAI/finbert"
SOCIAL_SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
NER_MODEL            = "dslim/bert-base-NER"
 
DEVICE = 0 if torch.cuda.is_available() else -1   # GPU if available, else CPU
 
# Source credibility weights (heuristic — tunable)
SOURCE_CREDIBILITY: dict[str, float] = {
    "newsapi":        0.85,
    "gnews":          0.80,
    "thenewsapi":     0.80,
    "reuters":        0.95,
    "bloomberg":      0.95,
    "ft":             0.92,
    "wsj":            0.90,
    "reddit":         0.55,
    "rss":            0.70,
    "freightwaves":   0.85,
    "yfinance":       0.90,
    "fred":           0.95,
    "alpha_vantage":  0.88,
    "un comtrade":    0.92,
    "openweather":    0.88,
    "open-meteo":     0.85,
    "default":        0.65,
}
 
MAX_TOKEN_LEN = 512   # HuggingFace token limit safety cap
 
 
# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADER  (lazy singleton — loads once, reused for all items)
# ─────────────────────────────────────────────────────────────────────────────
 
class NLPModels:
    """Singleton wrapper that loads all three HuggingFace pipelines."""
 
    _finbert:  Optional[Pipeline] = None
    _social:   Optional[Pipeline] = None
    _ner:      Optional[Pipeline] = None
 
    @classmethod
    def finbert(cls) -> Pipeline:
        if cls._finbert is None:
            print("  [↓] Loading FinBERT (ProsusAI/finbert) …")
            cls._finbert = pipeline(
                "text-classification",
                model=FINBERT_MODEL,
                device=DEVICE,
                top_k=None,          # return all three class scores
                truncation=True,
                max_length=MAX_TOKEN_LEN,
            )
        return cls._finbert
 
    @classmethod
    def social(cls) -> Pipeline:
        if cls._social is None:
            print("  [↓] Loading Twitter-RoBERTa (cardiffnlp) …")
            cls._social = pipeline(
                "text-classification",
                model=SOCIAL_SENTIMENT_MODEL,
                device=DEVICE,
                top_k=None,
                truncation=True,
                max_length=MAX_TOKEN_LEN,
            )
        return cls._social
 
    @classmethod
    def ner(cls) -> Pipeline:
        if cls._ner is None:
            print("  [↓] Loading BERT-NER (dslim/bert-base-NER) …")
            cls._ner = pipeline(
                "ner",
                model=NER_MODEL,
                device=DEVICE,
                aggregation_strategy="simple",
                truncation=True,
                max_length=MAX_TOKEN_LEN,
            )
        return cls._ner
 
 
# ─────────────────────────────────────────────────────────────────────────────
# HELPER UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
 
def _truncate(text: str, max_chars: int = 1800) -> str:
    """Hard-truncate text before passing to tokeniser (safety guard)."""
    return text[:max_chars] if text else ""
 
 
def _get_reliability(source: str) -> float:
    """Return credibility weight for a source string (case-insensitive)."""
    src = (source or "").lower()
    for key, score in SOURCE_CREDIBILITY.items():
        if key in src:
            return score
    return SOURCE_CREDIBILITY["default"]
 
 
def _run_finbert(text: str) -> SentimentScore:
    """Run FinBERT on text. Returns SentimentScore with all three class probs."""
    text = _truncate(text)
    if not text.strip():
        return SentimentScore(label="neutral", score=0.0, positive=0.0, negative=0.0, neutral=1.0, model=FINBERT_MODEL)
 
    try:
        results = NLPModels.finbert()(text)[0]   # list of {label, score}
        scores  = {r["label"].lower(): r["score"] for r in results}
        dominant = max(scores, key=scores.get)
        return SentimentScore(
            label    = dominant,
            score    = scores[dominant],
            positive = scores.get("positive", 0.0),
            negative = scores.get("negative", 0.0),
            neutral  = scores.get("neutral",  0.0),
            model    = FINBERT_MODEL,
        )
    except Exception as exc:
        return SentimentScore(label="neutral", score=0.0, positive=0.0, negative=0.0, neutral=1.0, model=FINBERT_MODEL, error=str(exc))
 
 
def _run_social_sentiment(text: str) -> SentimentScore:
    """Run Twitter-RoBERTa on text."""
    text = _truncate(text)
    if not text.strip():
        return SentimentScore(label="neutral", score=0.0, positive=0.0, negative=0.0, neutral=1.0, model=SOCIAL_SENTIMENT_MODEL)
 
    try:
        results = NLPModels.social()(text)[0]
        # cardiffnlp labels: Positive / Negative / Neutral (title-cased)
        scores  = {r["label"].lower(): r["score"] for r in results}
        dominant = max(scores, key=scores.get)
        return SentimentScore(
            label    = dominant,
            score    = scores[dominant],
            positive = scores.get("positive", 0.0),
            negative = scores.get("negative", 0.0),
            neutral  = scores.get("neutral",  0.0),
            model    = SOCIAL_SENTIMENT_MODEL,
        )
    except Exception as exc:
        return SentimentScore(label="neutral", score=0.0, positive=0.0, negative=0.0, neutral=1.0, model=SOCIAL_SENTIMENT_MODEL, error=str(exc))
 
 
def _run_ner(text: str) -> list[NEREntity]:
    """Run BERT-NER and return deduplicated entity list."""
    text = _truncate(text)
    if not text.strip():
        return []
 
    try:
        raw_entities = NLPModels.ner()(text)
        seen: set[tuple] = set()
        entities: list[NEREntity] = []
        for ent in raw_entities:
            key = (ent["word"].strip(), ent["entity_group"])
            if key in seen:
                continue
            seen.add(key)
            entities.append(NEREntity(
                text       = ent["word"].strip(),
                entity_type= ent["entity_group"],   # PER, ORG, LOC, MISC
                score      = round(float(ent["score"]), 4),
                model      = NER_MODEL,
            ))
        return entities
    except Exception:
        return []
 
 
def _extract_geo_tags(entities: list[NEREntity]) -> list[str]:
    """Pull LOC/GPE entities as geo_tags (deduplicated)."""
    seen = set()
    tags = []
    for ent in entities:
        if ent.entity_type in ("LOC", "GPE") and ent.text not in seen:
            seen.add(ent.text)
            tags.append(ent.text)
    return tags
 
 
# ─────────────────────────────────────────────────────────────────────────────
# PER-SIGNAL ENRICHMENT FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
 
def _enrich_news(raw: dict) -> EnrichedNewsItem:
    combined_text = f"{raw.get('title', '')} {raw.get('body', '')}".strip()
    sentiment  = _run_finbert(combined_text)
    entities   = _run_ner(combined_text)
    geo_tags   = _extract_geo_tags(entities)
    reliability = _get_reliability(raw.get("source", ""))
 
    return EnrichedNewsItem(
        **raw,
        sentiment   = sentiment,
        entities    = entities,
        geo_tags    = geo_tags,
        reliability = reliability,
    )
 
 
def _enrich_social(raw: dict) -> EnrichedSocialSignal:
    text      = raw.get("text", "")
    sentiment = _run_social_sentiment(text)
    entities  = _run_ner(text)
    geo_tags  = _extract_geo_tags(entities)
    reliability = _get_reliability(raw.get("source", ""))
 
    return EnrichedSocialSignal(
        **raw,
        sentiment   = sentiment,
        entities    = entities,
        geo_tags    = geo_tags,
        reliability = reliability,
    )
 
 
def _enrich_stock(raw: dict) -> EnrichedStockSignal:
    # Build a synthetic sentence for FinBERT
    ticker      = raw.get("ticker", "UNKNOWN")
    change_pct  = raw.get("change_pct", 0.0)
    price       = raw.get("price", 0.0)
    direction   = "up" if change_pct >= 0 else "down"
    synthetic   = f"{ticker} stock moved {direction} {abs(change_pct):.2f}% to ${price:.2f}."
    sentiment   = _run_finbert(synthetic)
    reliability = _get_reliability(raw.get("source", ""))
 
    return EnrichedStockSignal(
        **raw,
        sentiment   = sentiment,
        reliability = reliability,
    )
 
 
def _enrich_port(raw: dict) -> EnrichedPortSignal:
    port_name  = raw.get("port_name", "")
    commodity  = raw.get("commodity", "")
    congestion = raw.get("congestion_flag", False)
    text       = f"Port {port_name} handling {commodity}. Congestion: {congestion}."
    sentiment  = _run_finbert(text)
    entities   = _run_ner(text)
    geo_tags   = _extract_geo_tags(entities)
    reliability = _get_reliability(raw.get("source", ""))
 
    return EnrichedPortSignal(
        **raw,
        sentiment   = sentiment,
        entities    = entities,
        geo_tags    = geo_tags,
        reliability = reliability,
    )
 
 
def _enrich_weather(raw: dict) -> EnrichedWeatherSignal:
    city        = raw.get("city", "")
    desc        = raw.get("description", "")
    disruption  = raw.get("disruption_flag", False)
    text        = f"Weather in {city}: {desc}. Disruption risk: {disruption}."
    sentiment   = _run_finbert(text)
    reliability = _get_reliability(raw.get("source", ""))
 
    return EnrichedWeatherSignal(
        **raw,
        sentiment   = sentiment,
        reliability = reliability,
    )
 
 
def _enrich_commodity(raw: dict) -> EnrichedCommoditySignal:
    commodity   = raw.get("commodity", "")
    price       = raw.get("price", 0.0)
    currency    = raw.get("currency", "USD")
    text        = f"Commodity {commodity} is priced at {price} {currency}."
    sentiment   = _run_finbert(text)
    reliability = _get_reliability(raw.get("source", ""))
 
    return EnrichedCommoditySignal(
        **raw,
        sentiment   = sentiment,
        reliability = reliability,
    )
 
 
# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
 
def run_layer2(
    input_path:  str = "risk_input_bundle.json",
    output_path: str = "enriched_risk_bundle.json",
) -> EnrichedRiskInputBundle:
    """
    Full Layer 2 pipeline.
    Reads RiskInputBundle → enriches with NLP → writes EnrichedRiskInputBundle.
    """
 
    print("\n" + "=" * 65)
    print("  LAYER 2 — NLP PRE-PROCESSING (HuggingFace Transformers)")
    print("=" * 65)
 
    # ── 1. Load Layer 1 output ────────────────────────────────────────────────
    print(f"\n[1/5] Loading RiskInputBundle from '{input_path}' …")
    with open(input_path, "r") as f:
        raw_bundle = json.load(f)
 
    domain      = raw_bundle.get("domain", "supply_chain")
    fetched_at  = raw_bundle.get("fetched_at", datetime.utcnow().isoformat())
 
    news_raw       = raw_bundle.get("news", [])
    social_raw     = raw_bundle.get("social", [])
    stocks_raw     = raw_bundle.get("stocks", [])
    ports_raw      = raw_bundle.get("ports", [])
    weather_raw    = raw_bundle.get("weather", [])
    commodities_raw= raw_bundle.get("commodities", [])
    errors_raw     = raw_bundle.get("errors", [])
 
    total_items = sum(len(x) for x in [news_raw, social_raw, stocks_raw, ports_raw, weather_raw, commodities_raw])
    print(f"    Domain          : {domain}")
    print(f"    Total signals   : {total_items}")
    print(f"    Agent errors    : {len(errors_raw)}")
 
    # ── 2. Load models (lazy — only if data exists) ───────────────────────────
    print("\n[2/5] Pre-loading NLP models …")
    if news_raw or stocks_raw or ports_raw or weather_raw or commodities_raw:
        _ = NLPModels.finbert()
    if social_raw:
        _ = NLPModels.social()
    if total_items > 0:
        _ = NLPModels.ner()
    print("    All models loaded ✓")
 
    # ── 3. Enrich each signal type ────────────────────────────────────────────
    print("\n[3/5] Running NLP enrichment per signal …")
 
    enriched_news: list[EnrichedNewsItem] = []
    for i, item in enumerate(news_raw):
        print(f"    NewsItem {i+1}/{len(news_raw)} …", end="\r")
        enriched_news.append(_enrich_news(item))
    if news_raw: print(f"    ✓ {len(enriched_news)} news items enriched            ")
 
    enriched_social: list[EnrichedSocialSignal] = []
    for i, item in enumerate(social_raw):
        print(f"    SocialSignal {i+1}/{len(social_raw)} …", end="\r")
        enriched_social.append(_enrich_social(item))
    if social_raw: print(f"    ✓ {len(enriched_social)} social signals enriched         ")
 
    enriched_stocks: list[EnrichedStockSignal] = []
    for i, item in enumerate(stocks_raw):
        enriched_stocks.append(_enrich_stock(item))
    if stocks_raw: print(f"    ✓ {len(enriched_stocks)} stock signals enriched")
 
    enriched_ports: list[EnrichedPortSignal] = []
    for i, item in enumerate(ports_raw):
        enriched_ports.append(_enrich_port(item))
    if ports_raw: print(f"    ✓ {len(enriched_ports)} port signals enriched")
 
    enriched_weather: list[EnrichedWeatherSignal] = []
    for i, item in enumerate(weather_raw):
        enriched_weather.append(_enrich_weather(item))
    if weather_raw: print(f"    ✓ {len(enriched_weather)} weather signals enriched")
 
    enriched_commodities: list[EnrichedCommoditySignal] = []
    for i, item in enumerate(commodities_raw):
        enriched_commodities.append(_enrich_commodity(item))
    if commodities_raw: print(f"    ✓ {len(enriched_commodities)} commodity signals enriched")
 
    # ── 4. Build EnrichedRiskInputBundle ─────────────────────────────────────
    print("\n[4/5] Assembling EnrichedRiskInputBundle …")
    enriched_bundle = EnrichedRiskInputBundle(
        domain            = domain,
        fetched_at        = fetched_at,
        enriched_at       = datetime.utcnow().isoformat(),
        news              = enriched_news,
        social            = enriched_social,
        stocks            = enriched_stocks,
        ports             = enriched_ports,
        weather           = enriched_weather,
        commodities       = enriched_commodities,
        errors            = errors_raw,
        layer1_completeness = raw_bundle.get("completeness_score", 0.0),
    )
    enriched_bundle.compute_aggregate_sentiment()
 
    # ── 5. Persist ────────────────────────────────────────────────────────────
    print(f"\n[5/5] Writing enriched bundle to '{output_path}' …")
    with open(output_path, "w") as f:
        json.dump(enriched_bundle.to_dict(), f, indent=2, default=str)
 
    _print_summary(enriched_bundle)
    return enriched_bundle
 
 
# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY PRINTER
# ─────────────────────────────────────────────────────────────────────────────
 
def _print_summary(bundle: EnrichedRiskInputBundle) -> None:
    print("\n" + "=" * 65)
    print("  ENRICHED RISK BUNDLE SUMMARY")
    print("=" * 65)
    print(f"  Enriched at              : {bundle.enriched_at}")
    print(f"  Layer 1 completeness     : {bundle.layer1_completeness * 100:.0f}%")
    print(f"  Aggregate sentiment      : {bundle.aggregate_sentiment}")
    print(f"  Avg reliability          : {bundle.avg_reliability:.2f}")
    print(f"  Total items enriched     : {bundle.total_items}")
    print()
    print(f"  News items               : {len(bundle.news)}")
    print(f"  Social signals           : {len(bundle.social)}")
    print(f"  Stock signals            : {len(bundle.stocks)}")
    print(f"  Port records             : {len(bundle.ports)}")
    print(f"  Weather readings         : {len(bundle.weather)}")
    print(f"  Commodity prices         : {len(bundle.commodities)}")
 
    if bundle.news:
        print("\n  [Top News — Sentiment]")
        for item in bundle.news[:3]:
            lbl = item.sentiment.label.upper() if item.sentiment else "?"
            geo = ", ".join(item.geo_tags[:3]) or "—"
            print(f"    · [{lbl:8s}] {item.title[:55]}")
            print(f"               geo: {geo}")
 
    if bundle.stocks:
        print("\n  [Stock Signals — Sentiment]")
        for s in bundle.stocks:
            lbl = s.sentiment.label.upper() if s.sentiment else "?"
            print(f"    · [{lbl:8s}] {s.ticker:8s}  ${s.price:.2f}  ({s.change_pct:+.2f}%)")
 
    if bundle.top_geo_tags:
        print(f"\n  [Top Geo Tags]  {', '.join(bundle.top_geo_tags[:10])}")
 
    print("=" * 65)
    print("[→] Ready for Layer 3 (Groq LLaMA risk analysis)\n")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Layer 2 — NLP Pre-Processing")
    parser.add_argument("--input",  default="risk_input_bundle.json",   help="Layer 1 output JSON")
    parser.add_argument("--output", default="enriched_risk_bundle.json", help="Layer 2 output JSON")
    args = parser.parse_args()
 
    run_layer2(input_path=args.input, output_path=args.output)
 