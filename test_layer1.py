"""
test_layer1.py — pytest suite for Layer 1 agents and schemas
Run with: pytest test_layer1.py -v
"""

import json
import pytest
from unittest.mock import patch, MagicMock
from schemas import (
    NewsItem, StockSignal, SocialSignal, PortSignal,
    WeatherSignal, CommoditySignal, RiskInputBundle, DataUnavailableSignal,
)


# ─────────────────────────────────────────────────────────────────────────────
# Schema tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSchemas:

    def test_news_item_valid(self):
        item = NewsItem(
            title="Port congestion in Singapore",
            body="Ships are queuing outside...",
            source="newsapi",
            url="https://example.com",
            published_at="2024-01-01T10:00:00Z",
        )
        assert item.title == "Port congestion in Singapore"
        assert item.sentiment_score is None

    def test_stock_signal_valid(self):
        sig = StockSignal(ticker="FDX", price=250.5, change_pct=-2.3)
        assert sig.ticker == "FDX"
        assert sig.source == "yfinance"

    def test_weather_disruption_flag(self):
        w = WeatherSignal(
            city="Singapore", temp_celsius=32.0,
            description="typhoon", humidity=90,
            wind_speed=25.0, disruption_flag=True,
            source="openweathermap",
        )
        assert w.disruption_flag is True

    def test_risk_bundle_completeness_full(self):
        bundle = RiskInputBundle(
            news=[NewsItem(title="t", body="b", source="s", url="u", published_at="d")],
            stocks=[StockSignal(ticker="FDX", price=250.0, change_pct=0.5)],
            social=[SocialSignal(text="post", source="reddit")],
            ports=[PortSignal(port_name="Singapore", country="SG", source="comtrade")],
            weather=[WeatherSignal(city="SG", temp_celsius=30.0, description="clear",
                                   humidity=70, wind_speed=5.0, source="openweather")],
            commodities=[CommoditySignal(commodity="WTI", price=80.0, currency="USD", source="FRED")],
        )
        bundle.compute_completeness()
        assert bundle.completeness_score == 1.0

    def test_risk_bundle_completeness_partial(self):
        bundle = RiskInputBundle(
            news=[NewsItem(title="t", body="b", source="s", url="u", published_at="d")],
        )
        bundle.compute_completeness()
        assert bundle.completeness_score == pytest.approx(1 / 6, rel=1e-2)

    def test_data_unavailable_signal(self):
        sig = DataUnavailableSignal(agent_name="NewsAgent", reason="API timeout")
        assert sig.agent_name == "NewsAgent"
        assert "API timeout" in sig.reason

    def test_bundle_json_roundtrip(self):
        bundle = RiskInputBundle(
            news=[NewsItem(title="Test", body="Body", source="newsapi",
                           url="http://x.com", published_at="2024-01-01")],
        )
        dumped  = bundle.model_dump_json()
        loaded  = RiskInputBundle.model_validate_json(dumped)
        assert loaded.news[0].title == "Test"


# ─────────────────────────────────────────────────────────────────────────────
# Tool unit tests (mocked HTTP)
# ─────────────────────────────────────────────────────────────────────────────

class TestTools:

    @patch("tools.requests.get")
    @patch.dict("os.environ", {"NEWSAPI_KEY": "test123"})
    def test_fetch_newsapi_success(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "articles": [{
                "title": "Shipping delay",
                "description": "Ports are congested",
                "source": {"name": "Reuters"},
                "url": "https://reuters.com",
                "publishedAt": "2024-01-01T00:00:00Z",
            }]
        }
        mock_get.return_value = mock_resp
        from tools import fetch_newsapi
        result = fetch_newsapi.invoke({"keywords": "port congestion"})
        data   = json.loads(result)
        assert "articles" in data
        assert data["articles"][0]["title"] == "Shipping delay"

    @patch("tools.requests.get")
    @patch.dict("os.environ", {"OPENWEATHER_API_KEY": "test123"})
    def test_fetch_openweather_disruption(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "main": {"temp": 28.0, "humidity": 85},
            "weather": [{"description": "typhoon"}],
            "wind": {"speed": 20.0},
        }
        mock_get.return_value = mock_resp
        from tools import fetch_openweather
        result = fetch_openweather.invoke({"city": "Shanghai"})
        data   = json.loads(result)
        assert data["disruption_flag"] is True

    @patch("tools.requests.get")
    @patch.dict("os.environ", {"NEWSAPI_KEY": "test123"})
    def test_fetch_newsapi_error_handling(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.json.return_value = {"message": "Invalid API key"}
        mock_get.return_value = mock_resp
        from tools import fetch_newsapi
        result = fetch_newsapi.invoke({"keywords": "test"})
        data   = json.loads(result)
        assert "error" in data

    @patch("tools.yf.Ticker")
    def test_fetch_yfinance(self, mock_ticker_cls):
        import pandas as pd
        mock_ticker = MagicMock()
        mock_ticker.fast_info.last_price = 260.50
        mock_ticker.fast_info.previous_close = 255.00
        mock_ticker.history.return_value = pd.DataFrame({"Close": [250, 255, 260]})
        mock_ticker_cls.return_value = mock_ticker
        from tools import fetch_yfinance
        result = fetch_yfinance.invoke({"tickers": "FDX"})
        data   = json.loads(result)
        assert data["stocks"][0]["ticker"] == "FDX"
        assert data["stocks"][0]["price"] == 260.50

    def test_fetch_openweather_missing_key(self):
        import os
        key_backup = os.environ.pop("OPENWEATHER_API_KEY", None)
        from tools import fetch_openweather
        result = fetch_openweather.invoke({"city": "Singapore"})
        data   = json.loads(result)
        assert "error" in data
        if key_backup:
            os.environ["OPENWEATHER_API_KEY"] = key_backup


# ─────────────────────────────────────────────────────────────────────────────
# Parser unit tests
# ─────────────────────────────────────────────────────────────────────────────

class TestParsers:

    def test_parse_news(self):
        from layer1_agents import _parse_news
        raw = json.dumps({"articles": [
            {"title": "Test", "body": "Body", "source": "newsapi",
             "url": "http://x.com", "published_at": "2024-01-01"},
        ]})
        items = _parse_news(raw, "newsapi")
        assert len(items) == 1
        assert isinstance(items[0], NewsItem)

    def test_parse_stocks(self):
        from layer1_agents import _parse_stocks
        raw = json.dumps({"stocks": [
            {"ticker": "FDX", "price": 260.0, "change_pct": 1.2,
             "volatility_30d": 0.18, "source": "yfinance"},
        ]})
        signals = _parse_stocks(raw)
        assert signals[0].ticker == "FDX"

    def test_parse_weather_disruption(self):
        from layer1_agents import _parse_weather
        raw = json.dumps({
            "city": "Shanghai", "temp_celsius": 20.0,
            "description": "storm", "humidity": 80,
            "wind_speed": 18.0, "disruption_flag": True,
            "source": "openweathermap",
        })
        readings = _parse_weather(raw)
        assert readings[0].disruption_flag is True

    def test_parse_empty_returns_empty_list(self):
        from layer1_agents import _parse_news, _parse_stocks, _parse_social
        assert _parse_news("{}", "test") == []
        assert _parse_stocks("{}") == []
        assert _parse_social("{}") == []
