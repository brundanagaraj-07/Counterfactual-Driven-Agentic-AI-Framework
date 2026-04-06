"""
tools.py — LangChain @tool definitions for every data source.
Each tool is a standalone function that fetches from one API
and returns a JSON string (so the LLM can read it if needed).
"""

from __future__ import annotations
import os
import json
import asyncio
import feedparser
import requests
import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional

from langchain_core.tools import tool
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from aiolimiter import AsyncLimiter

# ── Rate limiters (free-tier safe) ──────────────────────────────────────────
_news_limiter    = AsyncLimiter(max_rate=5,  time_period=60)   # NewsAPI: cautious
_gnews_limiter   = AsyncLimiter(max_rate=5,  time_period=60)
_reddit_limiter  = AsyncLimiter(max_rate=10, time_period=60)
_weather_limiter = AsyncLimiter(max_rate=10, time_period=60)
_fred_limiter    = AsyncLimiter(max_rate=5,  time_period=60)


# ─────────────────────────────────────────────────────────────────────────────
# NEWS TOOLS
# ─────────────────────────────────────────────────────────────────────────────

@tool
def fetch_newsapi(keywords: str) -> str:
    """
    Fetch top supply-chain news articles from NewsAPI.
    Args:
        keywords: Comma-separated search terms, e.g. 'port congestion, shipping delay'
    """
    api_key = os.getenv("NEWSAPI_KEY")
    if not api_key:
        return json.dumps({"error": "NEWSAPI_KEY not set"})

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": keywords,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 5,
        "apiKey": api_key,
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        if resp.status_code != 200:
            return json.dumps({"error": data.get("message", "NewsAPI error")})

        articles = []
        for a in data.get("articles", []):
            articles.append({
                "title":        a.get("title", ""),
                "body":         (a.get("description") or "")[:300],
                "source":       a.get("source", {}).get("name", "newsapi"),
                "url":          a.get("url", ""),
                "published_at": a.get("publishedAt", ""),
            })
        return json.dumps({"articles": articles})
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def fetch_gnews(keywords: str) -> str:
    """
    Fetch top supply-chain news articles from GNews.
    Args:
        keywords: Search query string, e.g. 'supply chain disruption'
    """
    api_key = os.getenv("GNEWS_API_KEY")
    if not api_key:
        return json.dumps({"error": "GNEWS_API_KEY not set"})

    url = "https://gnews.io/api/v4/search"
    params = {
        "q":        keywords,
        "lang":     "en",
        "max":      5,
        "apikey":   api_key,
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        articles = []
        for a in data.get("articles", []):
            articles.append({
                "title":        a.get("title", ""),
                "body":         (a.get("description") or "")[:300],
                "source":       a.get("source", {}).get("name", "gnews"),
                "url":          a.get("url", ""),
                "published_at": a.get("publishedAt", ""),
            })
        return json.dumps({"articles": articles})
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def fetch_thenewsapi(keywords: str) -> str:
    """
    Fetch top supply-chain news articles from TheNewsAPI.
    Args:
        keywords: Search query, e.g. 'freight shipping delay'
    """
    api_key = os.getenv("THENEWSAPI_KEY")
    if not api_key:
        return json.dumps({"error": "THENEWSAPI_KEY not set"})

    url = "https://api.thenewsapi.com/v1/news/all"
    params = {
        "api_token": api_key,
        "search":    keywords,
        "language":  "en",
        "limit":     5,
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        articles = []
        for a in data.get("data", []):
            articles.append({
                "title":        a.get("title", ""),
                "body":         (a.get("description") or "")[:300],
                "source":       a.get("source", "thenewsapi"),
                "url":          a.get("url", ""),
                "published_at": a.get("published_at", ""),
            })
        return json.dumps({"articles": articles})
    except Exception as e:
        return json.dumps({"error": str(e)})


# ─────────────────────────────────────────────────────────────────────────────
# SOCIAL TOOLS
# ─────────────────────────────────────────────────────────────────────────────

@tool
def fetch_reddit(subreddit: str, limit: int = 5) -> str:
    """
    Fetch recent subreddit posts from Pushshift (no Reddit API keys required).
    Args:
        subreddit: Name without 'r/', e.g. 'supplychain' or 'logistics'
        limit: Number of posts to fetch (default 5)
    """
    try:
        safe_limit = max(1, min(limit, 25))
        url = "https://api.pushshift.io/reddit/search/submission/"
        params = {
            "subreddit": subreddit,
            "size": safe_limit,
            "sort": "desc",
            "sort_type": "created_utc",
        }

        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        if resp.status_code != 200:
            return json.dumps({"error": data.get("error", "Pushshift API error")})

        posts = []
        for post in data.get("data", []):
            title = post.get("title", "")
            body = (post.get("selftext") or "")[:200]
            permalink = post.get("permalink", "")
            posts.append({
                "text":      f"{title}. {body}".strip(),
                "source":    "pushshift",
                "subreddit": subreddit,
                "score":     post.get("score"),
                "url":       f"https://reddit.com{permalink}" if permalink else "",
            })
        return json.dumps({"posts": posts})
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def fetch_rss_feed(feed_url: str) -> str:
    """
    Fetch latest entries from an RSS feed (e.g. FreightWaves, Splash247).
    Args:
        feed_url: Full RSS feed URL
    """
    try:
        feed = feedparser.parse(feed_url)
        items = []
        for entry in feed.entries[:5]:
            items.append({
                "text":      f"{entry.get('title', '')}. {entry.get('summary', '')[:200]}",
                "source":    "rss",
                "url":       entry.get("link", ""),
            })
        return json.dumps({"items": items})
    except Exception as e:
        return json.dumps({"error": str(e)})


# ─────────────────────────────────────────────────────────────────────────────
# STOCK MARKET TOOLS
# ─────────────────────────────────────────────────────────────────────────────

@tool
def fetch_yfinance(tickers: str) -> str:
    """
    Fetch current stock price, % change, and 30-day volatility for given tickers.
    Args:
        tickers: Space-separated ticker symbols, e.g. 'FDX UPS MAERSK-B.CO'
    """
    results = []
    for ticker in tickers.strip().split():
        try:
            t = yf.Ticker(ticker)
            info = t.fast_info
            hist = t.history(period="30d")

            price      = round(float(info.last_price), 2)
            prev_close = round(float(info.previous_close), 2)
            change_pct = round(((price - prev_close) / prev_close) * 100, 2) if prev_close else 0.0
            volatility = round(float(hist["Close"].pct_change().std() * (252 ** 0.5) * 100), 2) if len(hist) > 1 else 0.0

            results.append({
                "ticker":        ticker,
                "price":         price,
                "change_pct":    change_pct,
                "volatility_30d": volatility,
                "source":        "yfinance",
            })
        except Exception as e:
            results.append({"ticker": ticker, "error": str(e)})
    return json.dumps({"stocks": results})


@tool
def fetch_fred(series_ids: str) -> str:
    """
    Fetch macroeconomic series from FRED (Federal Reserve Economic Data).
    Args:
        series_ids: Space-separated FRED series IDs, e.g. 'WTISPLC DCOILWTICO'
                    Useful ones: WTISPLC (WTI oil), DCOILWTICO (crude), BOPGSTB (trade balance)
    """
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        return json.dumps({"error": "FRED_API_KEY not set"})

    results = []
    for sid in series_ids.strip().split():
        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id":     sid,
            "api_key":       api_key,
            "file_type":     "json",
            "sort_order":    "desc",
            "limit":         1,
        }
        try:
            resp  = requests.get(url, params=params, timeout=10)
            data  = resp.json()
            obs   = data.get("observations", [{}])[0]
            results.append({
                "series_id": sid,
                "value":     obs.get("value", "N/A"),
                "date":      obs.get("date", ""),
                "source":    "FRED",
            })
        except Exception as e:
            results.append({"series_id": sid, "error": str(e)})
    return json.dumps({"fred_data": results})


@tool
def fetch_alpha_vantage(symbol: str, function: str = "TIME_SERIES_DAILY") -> str:
    """
    Fetch stock data from Alpha Vantage as a cross-check for yfinance.
    Args:
        symbol: Stock ticker symbol, e.g. 'FDX'
        function: Alpha Vantage function (default: TIME_SERIES_DAILY)
    """
    api_key = os.getenv("ALPHA_VANTAGE_KEY")
    if not api_key:
        return json.dumps({"error": "ALPHA_VANTAGE_KEY not set"})

    url = "https://www.alphavantage.co/query"
    params = {"function": function, "symbol": symbol, "apikey": api_key, "outputsize": "compact"}
    try:
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        ts   = data.get("Time Series (Daily)", {})
        latest_date = max(ts.keys()) if ts else None
        if not latest_date:
            return json.dumps({"error": "No data returned", "raw": str(data)[:200]})
        record = ts[latest_date]
        return json.dumps({
            "ticker":  symbol,
            "date":    latest_date,
            "open":    record.get("1. open"),
            "close":   record.get("4. close"),
            "volume":  record.get("5. volume"),
            "source":  "alpha_vantage",
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


# ─────────────────────────────────────────────────────────────────────────────
# PORT / TRADE TOOLS
# ─────────────────────────────────────────────────────────────────────────────

@tool
def fetch_un_comtrade(reporter_code: str = "842", commodity_code: str = "TOTAL") -> str:
    """
    Fetch trade flow data from UN Comtrade (free, no key needed for basic).
    Args:
        reporter_code: UN country code. 842=USA, 156=China, 276=Germany, 356=India
        commodity_code: HS code or 'TOTAL' for aggregate trade
    """
    url = "https://comtradeapi.un.org/public/v1/preview/C/A/HS"
    params = {
        "reporterCode": reporter_code,
        "period":       str((datetime.utcnow().year - 1)),   # previous year (annual)
        "cmdCode":      commodity_code,
        "flowCode":     "M",                                  # M=imports
        "maxRecords":   5,
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        data = resp.json()
        records = []
        for r in data.get("data", [])[:5]:
            records.append({
                "port_name":       r.get("reporterDesc", "Unknown"),
                "country":         r.get("reporterDesc", ""),
                "trade_value_usd": r.get("primaryValue", 0),
                "commodity":       r.get("cmdDesc", commodity_code),
                "source":          "UN Comtrade",
            })
        return json.dumps({"trade_records": records})
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def fetch_vessel_finder(port_name: str) -> str:
    """
    Fetch basic port vessel activity from VesselFinder free tier.
    NOTE: Full AIS data requires paid plan. Free tier returns limited info.
    Args:
        port_name: Port name to query, e.g. 'Singapore', 'Rotterdam'
    """
    api_key = os.getenv("VESSELFINDER_API_KEY", "")
    if not api_key:
        # Graceful fallback — return a stub so pipeline doesn't break
        return json.dumps({
            "port_name":      port_name,
            "note":           "VesselFinder key not set. Using stub data.",
            "congestion_flag": False,
            "source":         "stub",
        })

    url = f"https://api.vesselfinder.com/vessels"
    params = {"userkey": api_key, "port": port_name, "limit": 5}
    try:
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        return json.dumps({
            "port_name":      port_name,
            "vessel_count":   len(data.get("vessels", [])),
            "congestion_flag": len(data.get("vessels", [])) > 20,
            "source":         "vesselfinder",
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


# ─────────────────────────────────────────────────────────────────────────────
# WEATHER TOOLS
# ─────────────────────────────────────────────────────────────────────────────

@tool
def fetch_openweather(city: str) -> str:
    """
    Fetch current weather for a port city. Flags disruption if extreme conditions.
    Args:
        city: City name, e.g. 'Singapore', 'Rotterdam', 'Shanghai'
    """
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return json.dumps({"error": "OPENWEATHER_API_KEY not set"})

    url = "http://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "appid": api_key, "units": "metric"}
    try:
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        if resp.status_code != 200:
            return json.dumps({"error": data.get("message", "OpenWeather error")})

        wind_speed = data["wind"]["speed"]
        desc       = data["weather"][0]["description"]
        disruption = wind_speed > 15 or any(
            kw in desc.lower()
            for kw in ["storm", "typhoon", "hurricane", "blizzard", "tornado", "fog"]
        )
        return json.dumps({
            "city":            city,
            "temp_celsius":    data["main"]["temp"],
            "description":     desc,
            "humidity":        data["main"]["humidity"],
            "wind_speed":      wind_speed,
            "disruption_flag": disruption,
            "source":          "openweathermap",
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def fetch_open_meteo(latitude: float, longitude: float, location_name: str) -> str:
    """
    Fetch weather forecast from Open-Meteo (no API key needed, fully free).
    Args:
        latitude: Latitude of the location
        longitude: Longitude of the location
        location_name: Human-readable name for labeling
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude":   latitude,
        "longitude":  longitude,
        "current":    "temperature_2m,wind_speed_10m,precipitation,weather_code",
        "forecast_days": 1,
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        current    = data.get("current", {})
        wind_speed = current.get("wind_speed_10m", 0)
        disruption = wind_speed > 15 or current.get("precipitation", 0) > 10
        return json.dumps({
            "city":            location_name,
            "temp_celsius":    current.get("temperature_2m", 0),
            "description":     f"WMO code {current.get('weather_code', 0)}",
            "humidity":        0,         # not in free endpoint
            "wind_speed":      wind_speed,
            "disruption_flag": disruption,
            "source":          "open-meteo",
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


# ─────────────────────────────────────────────────────────────────────────────
# COMMODITY TOOLS
# ─────────────────────────────────────────────────────────────────────────────

@tool
def fetch_commodities_api(commodities: str) -> str:
    """
    Fetch commodity spot prices from Commodities-API (free tier).
    Args:
        commodities: Comma-separated commodity codes, e.g. 'BRENT,WTI,WHEAT,CORN'
    """
    api_key = os.getenv("COMMODITIES_API_KEY", "")
    if not api_key:
        # FRED fallback message
        return json.dumps({"error": "COMMODITIES_API_KEY not set. Use fetch_fred for oil prices."})

    url = "https://commodities-api.com/api/latest"
    params = {"access_key": api_key, "base": "USD", "symbols": commodities}
    try:
        resp  = requests.get(url, params=params, timeout=10)
        data  = resp.json()
        rates = data.get("data", {}).get("rates", {})
        result = []
        for name, price in rates.items():
            result.append({
                "commodity": name,
                "price":     round(1 / price, 4) if price else 0,   # API returns per-USD rate
                "currency":  "USD",
                "source":    "commodities-api",
            })
        return json.dumps({"commodities": result})
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def fetch_fred_commodities(series_ids: str = "DCOILWTICO GASDESW WHEAT") -> str:
    """
    Fetch commodity-related series from FRED as a free alternative.
    Args:
        series_ids: Space-separated FRED series IDs.
                    DCOILWTICO=WTI crude oil, GASDESW=diesel, WHEAT=wheat price index
    """
    return fetch_fred.invoke({"series_ids": series_ids})
