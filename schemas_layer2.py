"""
schemas_layer2.py — Pydantic v2 schemas for Layer 2 Enriched Output
====================================================================
Defines all enriched signal types and the EnrichedRiskInputBundle
that is passed to Layer 3 (Groq LLaMA risk analysis).
"""
 
from __future__ import annotations
 
from collections import Counter
from typing import Optional, Any
from pydantic import BaseModel, Field
 
 
# ─────────────────────────────────────────────────────────────────────────────
# NLP RESULT PRIMITIVES
# ─────────────────────────────────────────────────────────────────────────────
 
class SentimentScore(BaseModel):
    """Output from a sentiment classifier (FinBERT or RoBERTa)."""
    label:    str           = "neutral"    # 'positive' | 'negative' | 'neutral'
    score:    float         = 0.0          # confidence of dominant label
    positive: float         = 0.0
    negative: float         = 0.0
    neutral:  float         = 1.0
    model:    str           = ""
    error:    Optional[str] = None         # set if inference failed
 
 
class NEREntity(BaseModel):
    """A single named entity extracted by BERT-NER."""
    text:        str   = ""
    entity_type: str   = ""    # PER, ORG, LOC, MISC
    score:       float = 0.0
    model:       str   = ""
 
 
# ─────────────────────────────────────────────────────────────────────────────
# ENRICHED SIGNAL SCHEMAS  (extend Layer 1 raw fields + NLP outputs)
# ─────────────────────────────────────────────────────────────────────────────
 
class EnrichedNewsItem(BaseModel):
    # ── Layer 1 fields (mirrored) ─────────────────────────────────────────────
    title:        str = ""
    body:         str = ""
    source:       str = ""
    url:          str = ""
    published_at: str = ""
    # ── Layer 2 additions ─────────────────────────────────────────────────────
    sentiment:    Optional[SentimentScore]  = None
    entities:     list[NEREntity]           = Field(default_factory=list)
    geo_tags:     list[str]                 = Field(default_factory=list)
    reliability:  float                     = 0.65    # source credibility weight
 
 
class EnrichedSocialSignal(BaseModel):
    # ── Layer 1 fields ────────────────────────────────────────────────────────
    text:       str            = ""
    source:     str            = ""
    subreddit:  Optional[str]  = None
    score:      Optional[int]  = None
    url:        Optional[str]  = None
    # ── Layer 2 additions ─────────────────────────────────────────────────────
    sentiment:   Optional[SentimentScore] = None
    entities:    list[NEREntity]          = Field(default_factory=list)
    geo_tags:    list[str]                = Field(default_factory=list)
    reliability: float                    = 0.55
 
 
class EnrichedStockSignal(BaseModel):
    # ── Layer 1 fields ────────────────────────────────────────────────────────
    ticker:          str            = ""
    price:           float          = 0.0
    change_pct:      float          = 0.0
    volatility_30d:  Optional[float]= None
    source:          str            = ""
    # ── Layer 2 additions ─────────────────────────────────────────────────────
    sentiment:   Optional[SentimentScore] = None
    reliability: float                    = 0.90
 
 
class EnrichedPortSignal(BaseModel):
    # ── Layer 1 fields ────────────────────────────────────────────────────────
    port_name:       str            = ""
    country:         str            = ""
    trade_value_usd: Optional[float]= None
    commodity:       Optional[str]  = None
    congestion_flag: bool           = False
    source:          str            = ""
    # ── Layer 2 additions ─────────────────────────────────────────────────────
    sentiment:   Optional[SentimentScore] = None
    entities:    list[NEREntity]          = Field(default_factory=list)
    geo_tags:    list[str]                = Field(default_factory=list)
    reliability: float                    = 0.80
 
 
class EnrichedWeatherSignal(BaseModel):
    # ── Layer 1 fields ────────────────────────────────────────────────────────
    city:            str   = ""
    temp_celsius:    float = 0.0
    description:     str   = ""
    humidity:        int   = 0
    wind_speed:      float = 0.0
    disruption_flag: bool  = False
    source:          str   = ""
    # ── Layer 2 additions ─────────────────────────────────────────────────────
    sentiment:   Optional[SentimentScore] = None
    reliability: float                    = 0.88
 
 
class EnrichedCommoditySignal(BaseModel):
    # ── Layer 1 fields ────────────────────────────────────────────────────────
    commodity: str   = ""
    price:     float = 0.0
    currency:  str   = "USD"
    source:    str   = ""
    # ── Layer 2 additions ─────────────────────────────────────────────────────
    sentiment:   Optional[SentimentScore] = None
    reliability: float                    = 0.85
 
 
# ─────────────────────────────────────────────────────────────────────────────
# TOP-LEVEL ENRICHED BUNDLE
# ─────────────────────────────────────────────────────────────────────────────
 
class EnrichedRiskInputBundle(BaseModel):
    """
    Output of Layer 2.  Carries all enriched signals plus aggregate
    NLP metadata that Layer 3 (Groq LLaMA) will consume.
    """
 
    # ── Provenance ────────────────────────────────────────────────────────────
    domain:               str = "supply_chain"
    fetched_at:           str = ""    # timestamp from Layer 1
    enriched_at:          str = ""    # timestamp set by Layer 2
    layer1_completeness:  float = 0.0
 
    # ── Enriched signals ──────────────────────────────────────────────────────
    news:        list[EnrichedNewsItem]       = Field(default_factory=list)
    social:      list[EnrichedSocialSignal]   = Field(default_factory=list)
    stocks:      list[EnrichedStockSignal]    = Field(default_factory=list)
    ports:       list[EnrichedPortSignal]     = Field(default_factory=list)
    weather:     list[EnrichedWeatherSignal]  = Field(default_factory=list)
    commodities: list[EnrichedCommoditySignal]= Field(default_factory=list)
    errors:      list[Any]                    = Field(default_factory=list)
 
    # ── Aggregate NLP metadata (computed by compute_aggregate_sentiment) ──────
    aggregate_sentiment:  str         = "neutral"   # dominant across all items
    avg_reliability:      float       = 0.0
    top_geo_tags:         list[str]   = Field(default_factory=list)
    sentiment_breakdown:  dict        = Field(default_factory=dict)
    total_items:          int         = 0
 
    # ─────────────────────────────────────────────────────────────────────────
    def compute_aggregate_sentiment(self) -> None:
        """
        Walk all enriched signals, tally sentiment labels,
        compute average reliability, extract top geo-tags.
        """
        label_counts: Counter = Counter()
        reliabilities: list[float] = []
        geo_counter:   Counter = Counter()
 
        all_signals: list[Any] = (
            self.news + self.social + self.stocks +    # type: ignore[operator]
            self.ports + self.weather + self.commodities
        )
 
        for sig in all_signals:
            # Sentiment tally
            sent: Optional[SentimentScore] = getattr(sig, "sentiment", None)
            if sent and sent.label:
                label_counts[sent.label] += 1
 
            # Reliability
            rel: float = getattr(sig, "reliability", 0.65)
            reliabilities.append(rel)
 
            # Geo tags
            for tag in getattr(sig, "geo_tags", []):
                geo_counter[tag] += 1
 
        self.total_items = len(all_signals)
 
        # Dominant sentiment
        if label_counts:
            self.aggregate_sentiment = label_counts.most_common(1)[0][0]
            total = sum(label_counts.values())
            self.sentiment_breakdown = {
                lbl: round(cnt / total, 3) for lbl, cnt in label_counts.items()
            }
 
        # Avg reliability
        self.avg_reliability = (
            round(sum(reliabilities) / len(reliabilities), 3)
            if reliabilities else 0.0
        )
 
        # Top geo tags
        self.top_geo_tags = [tag for tag, _ in geo_counter.most_common(15)]
 
    # ─────────────────────────────────────────────────────────────────────────
    def to_dict(self) -> dict:
        """Serialise to plain dict (JSON-safe)."""
        return self.model_dump()
 