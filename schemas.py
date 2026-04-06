"""
schemas.py — Pydantic v2 data models for RiskInputBundle
All agent outputs are validated against these schemas before aggregation.
"""

from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class NewsItem(BaseModel):
    title: str
    body: str
    source: str
    url: str
    published_at: str
    sentiment_score: Optional[float] = None       # filled by Layer 2
    entities: Optional[list[str]] = None           # filled by Layer 2


class StockSignal(BaseModel):
    ticker: str
    price: float
    change_pct: float
    volatility_30d: Optional[float] = None
    rsi: Optional[float] = None
    source: str = "yfinance"


class SocialSignal(BaseModel):
    text: str
    source: str                # "reddit" | "rss"
    subreddit: Optional[str] = None
    score: Optional[int] = None
    url: Optional[str] = None
    sentiment_score: Optional[float] = None


class PortSignal(BaseModel):
    port_name: str
    country: str
    trade_value_usd: Optional[float] = None
    commodity: Optional[str] = None
    congestion_flag: bool = False
    source: str


class WeatherSignal(BaseModel):
    city: str
    temp_celsius: float
    description: str
    humidity: int
    wind_speed: float
    disruption_flag: bool = False     # True if extreme weather
    source: str


class CommoditySignal(BaseModel):
    commodity: str
    price: float
    currency: str
    change_pct: Optional[float] = None
    source: str


class DataUnavailableSignal(BaseModel):
    agent_name: str
    reason: str
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class RiskInputBundle(BaseModel):
    fetched_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    domain: str = "supply_chain"

    news: list[NewsItem] = []
    stocks: list[StockSignal] = []
    social: list[SocialSignal] = []
    ports: list[PortSignal] = []
    weather: list[WeatherSignal] = []
    commodities: list[CommoditySignal] = []

    errors: list[DataUnavailableSignal] = []
    completeness_score: float = 0.0             # 0.0–1.0, computed after assembly

    def compute_completeness(self) -> None:
        """Score how many data categories have at least one record."""
        categories = [self.news, self.stocks, self.social,
                      self.ports, self.weather, self.commodities]
        filled = sum(1 for c in categories if len(c) > 0)
        # Keep full precision so tests comparing fractional completeness (e.g., 1/6) remain stable.
        self.completeness_score = filled / len(categories)
