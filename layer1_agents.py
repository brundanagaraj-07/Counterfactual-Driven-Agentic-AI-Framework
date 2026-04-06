"""
layer1_agents.py — Layer 1 Multi-Agent Data Ingestion
=====================================================
Architecture:
  CrewAI Crew  (orchestration)
    ├── NewsAgent      → fetch_newsapi, fetch_gnews, fetch_thenewsapi
    ├── SocialAgent    → fetch_reddit, fetch_rss_feed
    ├── StockAgent     → fetch_yfinance, fetch_fred, fetch_alpha_vantage
    ├── PortAgent      → fetch_un_comtrade, fetch_vessel_finder
    ├── WeatherAgent   → fetch_openweather, fetch_open_meteo
    └── CommodityAgent → fetch_commodities_api, fetch_fred_commodities

All agents run concurrently via asyncio.gather().
Results are pushed onto an asyncio.Queue and consumed by the aggregator.
Final output: RiskInputBundle (Pydantic v2 validated).
"""

from __future__ import annotations
import os
import json
import asyncio
import warnings
from datetime import datetime
from typing import Any

warnings.filterwarnings("ignore", category=DeprecationWarning)

from dotenv import load_dotenv
load_dotenv()

# ── LangChain / LangGraph ────────────────────────────────────────────────────
from langchain_groq import ChatGroq

# ── CrewAI ────────────────────────────────────────────────────────────────────
from crewai import Agent, Task, Crew, Process

# ── Local ─────────────────────────────────────────────────────────────────────
from schemas import (
    NewsItem, StockSignal, SocialSignal, PortSignal,
    WeatherSignal, CommoditySignal, DataUnavailableSignal, RiskInputBundle,
)
from tools import (
    fetch_newsapi, fetch_gnews, fetch_thenewsapi,
    fetch_reddit, fetch_rss_feed,
    fetch_yfinance, fetch_fred, fetch_alpha_vantage,
    fetch_un_comtrade, fetch_vessel_finder,
    fetch_openweather, fetch_open_meteo,
    fetch_commodities_api, fetch_fred_commodities,
)

# ── LLM (shared across all agents) ───────────────────────────────────────────
LLM = ChatGroq(
    model="groq/llama-3.3-70b-versatile",  # Use groq/ prefix for LiteLLM compatibility
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
)

# Supply chain domain keywords
SUPPLY_CHAIN_KEYWORDS = "port congestion shipping delay supply chain disruption freight"
PORT_CITIES = ["Singapore", "Rotterdam", "Shanghai", "Los Angeles"]
STOCK_TICKERS = "AAPL MSFT TSM FDX"   # Apple, Microsoft, TSMC, FedEx


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS — parse raw tool JSON → typed Pydantic objects
# ─────────────────────────────────────────────────────────────────────────────

def _safe_json(raw: str) -> dict | list:
    try:
        return json.loads(raw)
    except Exception:
        return {}


def _parse_news(raw: str, source: str) -> list[NewsItem]:
    data  = _safe_json(raw)
    items = []
    for a in data.get("articles", data.get("items", [])):
        if "error" in a:
            continue
        try:
            items.append(NewsItem(
                title=a.get("title", ""),
                body=a.get("body", a.get("text", ""))[:400],
                source=a.get("source", source),
                url=a.get("url", ""),
                published_at=a.get("published_at", a.get("publishedAt", "")),
            ))
        except Exception:
            pass
    return items


def _parse_stocks(raw: str) -> list[StockSignal]:
    data   = _safe_json(raw)
    result = []
    for s in data.get("stocks", []):
        if "error" in s:
            continue
        try:
            result.append(StockSignal(
                ticker=s["ticker"],
                price=float(s.get("price", 0)),
                change_pct=float(s.get("change_pct", 0)),
                volatility_30d=s.get("volatility_30d"),
                source=s.get("source", "yfinance"),
            ))
        except Exception:
            pass
    return result


def _parse_social(raw: str) -> list[SocialSignal]:
    data   = _safe_json(raw)
    result = []
    for key in ("posts", "items"):
        for p in data.get(key, []):
            try:
                result.append(SocialSignal(
                    text=p.get("text", ""),
                    source=p.get("source", "reddit"),
                    subreddit=p.get("subreddit"),
                    score=p.get("score"),
                    url=p.get("url"),
                ))
            except Exception:
                pass
    return result


def _parse_ports(raw: str) -> list[PortSignal]:
    data   = _safe_json(raw)
    result = []
    for r in data.get("trade_records", []):
        try:
            result.append(PortSignal(
                port_name=r.get("port_name", "Unknown"),
                country=r.get("country", ""),
                trade_value_usd=r.get("trade_value_usd"),
                commodity=r.get("commodity"),
                congestion_flag=r.get("congestion_flag", False),
                source=r.get("source", "UN Comtrade"),
            ))
        except Exception:
            pass
    # Also handle single vessel-finder record
    if "port_name" in data:
        try:
            result.append(PortSignal(
                port_name=data["port_name"],
                country="",
                congestion_flag=data.get("congestion_flag", False),
                source=data.get("source", "vesselfinder"),
            ))
        except Exception:
            pass
    return result


def _parse_weather(raw: str) -> list[WeatherSignal]:
    data = _safe_json(raw)
    if "error" in data:
        return []
    try:
        return [WeatherSignal(
            city=data.get("city", "Unknown"),
            temp_celsius=float(data.get("temp_celsius", 0)),
            description=data.get("description", ""),
            humidity=int(data.get("humidity", 0)),
            wind_speed=float(data.get("wind_speed", 0)),
            disruption_flag=data.get("disruption_flag", False),
            source=data.get("source", "weather"),
        )]
    except Exception:
        return []


def _parse_commodities(raw: str) -> list[CommoditySignal]:
    data   = _safe_json(raw)
    result = []
    for item in data.get("commodities", data.get("fred_data", [])):
        try:
            result.append(CommoditySignal(
                commodity=item.get("commodity", item.get("series_id", "")),
                price=float(item.get("price", item.get("value", 0)) or 0),
                currency=item.get("currency", "USD"),
                source=item.get("source", ""),
            ))
        except Exception:
            pass
    return result


# ─────────────────────────────────────────────────────────────────────────────
# CREWAI AGENTS — each wraps its LangChain tools
# ─────────────────────────────────────────────────────────────────────────────

news_agent = Agent(
    role="Supply Chain News Intelligence Analyst",
    goal=(
        "Fetch the latest news about supply chain disruptions, port congestion, "
        "shipping delays, and freight market movements from multiple news APIs."
    ),
    backstory=(
        "You are a specialist news analyst monitoring global supply chains 24/7. "
        "You aggregate headlines from NewsAPI, GNews, and TheNewsAPI to surface "
        "the most relevant risk signals for logistics operators."
    ),
    tools=[fetch_newsapi, fetch_gnews, fetch_thenewsapi],
    llm=LLM,
    verbose=True,
    allow_delegation=False,
    max_iter=3,
)

social_agent = Agent(
    role="Social Media and Industry Feed Monitor",
    goal=(
        "Monitor Reddit communities and industry RSS feeds for grassroots signals "
        "about supply chain stress, port slowdowns, and logistics disruptions."
    ),
    backstory=(
        "You track real-time sentiment from practitioners in r/supplychain, "
        "r/logistics, and r/economics, plus industry RSS feeds from FreightWaves. "
        "Your job is to surface weak signals before they become mainstream news."
    ),
    tools=[fetch_reddit, fetch_rss_feed],
    llm=LLM,
    verbose=True,
    allow_delegation=False,
    max_iter=3,
)

stock_agent = Agent(
    role="Logistics and Freight Stock Market Analyst",
    goal=(
        "Fetch current prices, volatility, and macro indicators for key logistics "
        "and shipping companies. Flag unusual price movements as potential risk signals."
    ),
    backstory=(
        "You analyse stock market data for major logistics players (FedEx, UPS, ZIM) "
        "and macro FRED indicators. A 5%+ daily move in shipping stocks often precedes "
        "a supply chain disruption event by 24-48 hours."
    ),
    tools=[fetch_yfinance, fetch_fred, fetch_alpha_vantage],
    llm=LLM,
    verbose=True,
    allow_delegation=False,
    max_iter=3,
)

port_agent = Agent(
    role="Global Port and Trade Flow Monitor",
    goal=(
        "Retrieve current trade flow volumes from UN Comtrade and vessel activity "
        "from VesselFinder for key global ports. Identify anomalies in import/export volumes."
    ),
    backstory=(
        "You specialise in port intelligence and trade statistics. Drops in trade "
        "volumes at major hubs (Singapore, Rotterdam, Shanghai) are early indicators "
        "of global supply chain contractions."
    ),
    tools=[fetch_un_comtrade, fetch_vessel_finder],
    llm=LLM,
    verbose=True,
    allow_delegation=False,
    max_iter=3,
)

weather_agent = Agent(
    role="Meteorological Risk Assessment Agent",
    goal=(
        "Monitor weather conditions at key global port cities. Flag storm, typhoon, "
        "or high-wind conditions that could cause port closures or shipping delays."
    ),
    backstory=(
        "You track real-time weather at Singapore, Rotterdam, Shanghai, and Los Angeles. "
        "Wind speeds above 15 m/s, tropical storms, or dense fog are immediate disruption "
        "signals for maritime operations."
    ),
    tools=[fetch_openweather, fetch_open_meteo],
    llm=LLM,
    verbose=True,
    allow_delegation=False,
    max_iter=3,
)

commodity_agent = Agent(
    role="Commodity Price and Supply Pressure Analyst",
    goal=(
        "Fetch current prices for oil, diesel, wheat, and corn. Elevated fuel prices "
        "directly increase shipping costs and are a leading indicator of logistics inflation."
    ),
    backstory=(
        "You monitor commodity markets that directly feed into supply chain costs. "
        "WTI crude oil, diesel prices, and agricultural commodities are key inputs "
        "for freight rate forecasting."
    ),
    tools=[fetch_commodities_api, fetch_fred_commodities],
    llm=LLM,
    verbose=True,
    allow_delegation=False,
    max_iter=3,
)


# ─────────────────────────────────────────────────────────────────────────────
# CREWAI TASKS — what each agent must do
# ─────────────────────────────────────────────────────────────────────────────

news_task = Task(
    description=(
        f"Fetch the latest 5 articles from each of the three news APIs "
        f"(NewsAPI, GNews, TheNewsAPI) using keywords: '{SUPPLY_CHAIN_KEYWORDS}'. "
        f"Return a combined list of news articles in JSON format."
    ),
    expected_output="JSON list of news articles with title, body, source, url, published_at fields.",
    agent=news_agent,
)

social_task = Task(
    description=(
        "Fetch 5 hot posts from each of r/supplychain, r/logistics, and r/economics. "
        "Also fetch the FreightWaves RSS feed: https://www.freightwaves.com/feed. "
        "Return all posts and feed items as JSON."
    ),
    expected_output="JSON list of social posts and RSS items with text, source, url fields.",
    agent=social_agent,
)

stock_task = Task(
    description=(
        f"Fetch current stock data for tickers: {STOCK_TICKERS} using yfinance. "
        "Also fetch FRED series: DCOILWTICO BOPGSTB (oil price, trade balance). "
        "Return all stock and macro data as JSON."
    ),
    expected_output="JSON with stock prices, % changes, volatility and FRED macro values.",
    agent=stock_agent,
)

port_task = Task(
    description=(
        "Fetch recent import trade flows from UN Comtrade for USA (code 842) and China (code 156). "
        "Also check VesselFinder for vessel activity at Singapore and Rotterdam ports. "
        "Return trade records and port activity as JSON."
    ),
    expected_output="JSON with trade flow records and port congestion status.",
    agent=port_agent,
)

weather_task = Task(
    description=(
        f"Fetch current weather for these port cities using OpenWeatherMap: {PORT_CITIES}. "
        "Flag any city with wind speed > 15 m/s or storm conditions as disruption risk. "
        "Return all weather readings as JSON."
    ),
    expected_output="JSON list of weather readings with city, temp, description, wind_speed, disruption_flag.",
    agent=weather_agent,
)

commodity_task = Task(
    description=(
        "Fetch current commodity prices for oil and key agricultural goods. "
        "Try Commodities-API first; if unavailable, use FRED series: DCOILWTICO GASDESW. "
        "Return commodity prices as JSON."
    ),
    expected_output="JSON list of commodity prices with commodity name, price, currency, source.",
    agent=commodity_agent,
)


# ─────────────────────────────────────────────────────────────────────────────
# ASYNC RUNNER — wraps each CrewAI task in its own crew, runs all in parallel
# ─────────────────────────────────────────────────────────────────────────────

async def _run_single_crew(
    agent: Agent,
    task:  Task,
    queue: asyncio.Queue,
    agent_name: str,
) -> None:
    """
    Runs one agent+task as an isolated single-agent Crew.
    Pushes (agent_name, raw_output) onto the shared queue when done.
    On failure, pushes a DataUnavailableSignal.
    """
    try:
        crew   = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
        result = await asyncio.to_thread(crew.kickoff)           # CrewAI is sync → run in thread
        await queue.put((agent_name, str(result)))
        print(f"  [✓] {agent_name} completed")
    except Exception as e:
        signal = DataUnavailableSignal(agent_name=agent_name, reason=str(e))
        await queue.put((agent_name, signal))
        print(f"  [✗] {agent_name} failed: {e}")


async def run_all_agents() -> RiskInputBundle:
    """
    Launch all 6 agents concurrently via asyncio.gather().
    Consume the shared queue and assemble the final RiskInputBundle.
    """
    queue = asyncio.Queue()
    print("\n[Layer 1] Launching all agents in parallel...\n")

    # Kick off all agents simultaneously
    await asyncio.gather(
        _run_single_crew(news_agent,      news_task,      queue, "NewsAgent"),
        _run_single_crew(social_agent,    social_task,    queue, "SocialAgent"),
        _run_single_crew(stock_agent,     stock_task,     queue, "StockAgent"),
        _run_single_crew(port_agent,      port_task,      queue, "PortAgent"),
        _run_single_crew(weather_agent,   weather_task,   queue, "WeatherAgent"),
        _run_single_crew(commodity_agent, commodity_task, queue, "CommodityAgent"),
    )

    # ── Consume queue and build bundle ─────────────────────────────────────────
    bundle = RiskInputBundle()

    while not queue.empty():
        agent_name, payload = await queue.get()

        # Handle failure signals
        if isinstance(payload, DataUnavailableSignal):
            bundle.errors.append(payload)
            continue

        raw: str = payload

        if agent_name == "NewsAgent":
            bundle.news.extend(_parse_news(raw, "news_agent"))

        elif agent_name == "SocialAgent":
            bundle.social.extend(_parse_social(raw))

        elif agent_name == "StockAgent":
            bundle.stocks.extend(_parse_stocks(raw))

        elif agent_name == "PortAgent":
            bundle.ports.extend(_parse_ports(raw))

        elif agent_name == "WeatherAgent":
            bundle.weather.extend(_parse_weather(raw))

        elif agent_name == "CommodityAgent":
            bundle.commodities.extend(_parse_commodities(raw))

    bundle.compute_completeness()
    return bundle


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def run_layer1() -> RiskInputBundle:
    """Synchronous entry point — runs the full async pipeline and returns bundle."""
    return asyncio.run(run_all_agents())


def print_bundle_summary(bundle: RiskInputBundle) -> None:
    """Print a human-readable summary of what was fetched."""
    print("\n" + "=" * 60)
    print("  RISK INPUT BUNDLE SUMMARY")
    print("=" * 60)
    print(f"  Fetched at        : {bundle.fetched_at}")
    print(f"  Domain            : {bundle.domain}")
    print(f"  Completeness      : {bundle.completeness_score * 100:.0f}%")
    print(f"  News items        : {len(bundle.news)}")
    print(f"  Social signals    : {len(bundle.social)}")
    print(f"  Stock signals     : {len(bundle.stocks)}")
    print(f"  Port records      : {len(bundle.ports)}")
    print(f"  Weather readings  : {len(bundle.weather)}")
    print(f"  Commodity prices  : {len(bundle.commodities)}")
    if bundle.errors:
        print(f"  Agent errors      : {len(bundle.errors)}")
        for e in bundle.errors:
            print(f"    - {e.agent_name}: {e.reason[:80]}")
    print("=" * 60)

    if bundle.news:
        print("\n  [Top News Headlines]")
        for item in bundle.news[:3]:
            print(f"    · {item.title[:80]}")

    if bundle.stocks:
        print("\n  [Stock Signals]")
        for s in bundle.stocks:
            direction = "▲" if s.change_pct >= 0 else "▼"
            print(f"    · {s.ticker:12s} ${s.price:8.2f}  {direction} {abs(s.change_pct):.2f}%")

    if bundle.weather:
        print("\n  [Weather — Disruption Flags]")
        for w in bundle.weather:
            flag = " ⚠ DISRUPTION" if w.disruption_flag else ""
            print(f"    · {w.city:15s} {w.temp_celsius:5.1f}°C  {w.description}{flag}")

    if bundle.commodities:
        print("\n  [Commodity Prices]")
        for c in bundle.commodities:
            print(f"    · {c.commodity:15s} {c.price} {c.currency}")

    print()


if __name__ == "__main__":
    print("=" * 60)
    print("  LAYER 1 — MULTI-AGENT DATA INGESTION")
    print("  Domain: Supply Chain & Logistics")
    print("=" * 60)

    bundle = run_layer1()
    print_bundle_summary(bundle)

    # Save bundle to JSON for Layer 2 consumption
    output_path = "risk_input_bundle.json"
    with open(output_path, "w") as f:
        f.write(bundle.model_dump_json(indent=2))
    print(f"\n[✓] Bundle saved to {output_path}")
    print("[→] Ready for Layer 2 (HuggingFace NLP enrichment)")
