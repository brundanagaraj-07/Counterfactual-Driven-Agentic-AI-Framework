# Counterfactual-Driven Agentic AI Framework — Layer 1

**Multi-Agent Supply Chain Data Ingestion Pipeline**

A concurrent multi-agent system for real-time supply chain risk intelligence, powered by CrewAI, Groq LLM, and LangGraph.

## Overview

Layer 1 launches **6 autonomous agents in parallel** to ingest supply chain data from diverse sources:

- **NewsAgent** → NewsAPI (news articles)
- **SocialAgent** → Pushshift (Reddit posts) + RSS feeds
- **StockAgent** → yfinance + Alpha Vantage + FRED (stock prices & macro indicators)
- **PortAgent** → UN Comtrade + VesselFinder (trade flows & vessel activity)
- **WeatherAgent** → OpenWeather + Open-Meteo (port city weather)
- **CommodityAgent** → Commodities-API + FRED (oil, diesel, agricultural prices)

All agents orchestrated via **CrewAI**, validated with **Pydantic v2**, and output a `RiskInputBundle` JSON for Layer 2 (NLP enrichment).

---

## Installation

### Prerequisites
- **Python 3.11+**
- **Conda** or **venv** for environment management
- **Git** for version control

### Step 1: Clone the Repository

```bash
git clone https://github.com/brundanagaraj-07/Counterfactual-Driven-Agentic-AI-Framework.git
cd Counterfactual-Driven-Agentic-AI-Framework
```

### Step 2: Create and Activate Conda Environment

```bash
# Create environment
conda create -n Counterfactual python=3.11 -y

# Activate environment
conda activate Counterfactual
```

Or with **venv**:
```bash
python -m venv venv
source venv/Scripts/activate  # Windows
# or
source venv/bin/activate      # macOS/Linux
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- **CrewAI** (agent orchestration)
- **LangChain** + **LangGraph** (LLM integration)
- **Pydantic v2** (data validation)
- **yfinance**, **feedparser**, **requests** (data APIs)
- **pytest** (testing)

---

## API Keys Setup

### Step 1: Copy `.env.example` to `.env`

```bash
cp .env.example .env
```

### Step 2: Add Your API Keys

Edit `.env` and fill in the following (required keys marked with ⭐):

| Service | Key | Source | Required |
|---------|-----|--------|----------|
| **Groq LLM** ⭐ | `GROQ_API_KEY` | https://console.groq.com | Yes |
| **NewsAPI** | `NEWSAPI_KEY` | https://newsapi.org/register | Optional |
| **FRED (Fed Reserve Data)** | `FRED_API_KEY` | https://fred.stlouisfed.org/docs/api | Optional |
| **Alpha Vantage** | `ALPHA_VANTAGE_KEY` | https://www.alphavantage.co | Optional |
| **OpenWeather** | `OPENWEATHER_API_KEY` | https://openweathermap.org/api | Optional |
| **Commodities-API** | `COMMODITIES_API_KEY` | https://commodities-api.com | Optional |
| **VesselFinder** | `VESSELFINDER_API_KEY` | https://www.vesselfinder.com/api | Optional |

**Free-tier alternatives (no key required):**
- ✓ **Pushshift** (Reddit data) — no key needed
- ✓ **yfinance** (stocks) — no key needed
- ✓ **UN Comtrade** (trade data) — no key needed
- ✓ **Open-Meteo** (weather forecast) — no key needed

**Example `.env`:**
```bash
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxx
NEWSAPI_KEY=3eca2036c90a426e9513d8b04xxxxx
FRED_API_KEY=2027ed85461b99f7e41641faxxx
```

### ⚠️ Security Note
**Never commit `.env` to version control.** The `.gitignore` excludes it automatically.

---

## Running the Pipeline

### Quick Start

```bash
python layer1_agents.py
```

**Output:**
- Console: Summary of fetched data (weather, stocks, commodities, etc.)
- File: `risk_input_bundle.json` — Structured data ready for Layer 2

### Example Output

```
============================================================
  RISK INPUT BUNDLE SUMMARY
============================================================
  Fetched at        : 2026-04-07T10:15:30.123456
  Domain            : supply_chain
  Completeness      : 67%
  News items        : 5
  Social signals    : 12
  Stock signals     : 4
  Port records      : 3
  Weather readings  : 4
  Commodity prices  : 2
============================================================

  [Stock Signals]
    · AAPL         $  175.67  ▲ 0.00%
    · MSFT         $  325.21  ▲ 0.00%
    · TSM          $  115.32  ▲ 0.00%
    · FDX          $  225.11  ▲ 0.00%
```

---

## Project Structure

```
.
├── layer1_agents.py           # Main pipeline + 6 agent definitions
├── tools.py                   # 14 data fetching tools (wrapped with @tool)
├── schemas.py                 # Pydantic v2 models for validation
├── test_layer1.py             # Pytest suite (16 tests, all passing)
├── requirements.txt           # Python dependencies
├── .env.example              # Template for API keys
├── .gitignore                # Excludes .env, __pycache__, etc.
└── README.md                 # This file
```

---

## Testing

**Run the full test suite:**

```bash
pytest test_layer1.py -v
```

**Expected output:**
```
test_layer1.py::TestSchemas::test_news_item_valid PASSED
test_layer1.py::TestSchemas::test_stock_signal_valid PASSED
...
test_layer1.py::TestParsers::test_parse_news PASSED
========== 16 passed in 27.20s ==========
```

**Coverage:**
- ✓ Pydantic schema validation
- ✓ Tool unit tests (mocked HTTP)
- ✓ Parser logic tests
- ✓ Bundle completeness scoring

---

## Architecture

### Agent Orchestration
```
CrewAI Crew (6 agents in parallel)
    ├─ NewsAgent     [fetch_newsapi, fetch_gnews, fetch_thenewsapi]
    ├─ SocialAgent   [fetch_reddit (Pushshift), fetch_rss_feed]
    ├─ StockAgent    [fetch_yfinance, fetch_fred, fetch_alpha_vantage]
    ├─ PortAgent     [fetch_un_comtrade, fetch_vessel_finder]
    ├─ WeatherAgent  [fetch_openweather, fetch_open_meteo]
    └─ CommodityAgent [fetch_commodities_api, fetch_fred_commodities]
         ↓ (asyncio.gather)
    Aggregator → RiskInputBundle (Pydantic validated)
         ↓
    risk_input_bundle.json (Layer 2 input)
```

### Data Flow
1. **Tools** execute API calls (net → JSON strings)
2. **Parsers** convert JSON → Pydantic models
3. **Bundle** aggregates all signals
4. **Output** saved as `risk_input_bundle.json`

---

## Rate Limiting & Quotas

**Free-tier constraints:**
- **Groq**: 12,000 TPM (tokens/min) — may trigger rate limits on large runs
- **NewsAPI**: 100 requests/day
- **FRED**: Unlimited
- **yfinance**: Unlimited
- **OpenWeather**: 60 calls/min

**Recommended for production:**
- Upgrade Groq to paid tier (higher TPM limits)
- Use alternative LLMs (OpenAI, Claude) if needed
- Cache agent outputs to reduce API calls

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'pkg_resources'`
```bash
pip install "setuptools<81"
```

### `LiteLLM RateLimitError`
- Groq free tier TPM exceeded → Wait ~1-2 seconds or upgrade plan
- Check `GROQ_API_KEY` is valid

### `fetch_reddit` returns empty
- Pushshift API temporarily down → Check https://pushshift.io/status
- Fallback: Implement Reddit's JSON endpoint (no auth needed)

### Stock prices showing `$0.00`
- yfinance ticker not found → Verify ticker symbol (e.g., `AAPL`, `MSFT`)
- Alternative: Use Alpha Vantage as fallback (requires `ALPHA_VANTAGE_KEY`)

---

## Next Steps (Layer 2+)

After running Layer 1:

1. **Layer 2 (NLP Enrichment)** → HuggingFace transformers
   - Sentiment analysis on news/social
   - NER (Named Entity Recognition) for companies/ports
   - Aspect-based sentiment

2. **Layer 3 (Risk Scoring)**
   - Aggregate signals into risk indices
   - Forecast supply chain disruptions

3. **Layer 4 (Visualization)**
   - Real-time dashboards
   - Risk heatmaps by geography/company

---

## Contributing

1. Create a new branch: `git checkout -b feature/your-feature`
2. Make changes and test: `pytest test_layer1.py -v`
3. Commit: `git commit -m "Add your feature"`
4. Push: `git push origin feature/your-feature`
5. Open a Pull Request

---

## License

MIT (see LICENSE file, if present)

---

## Contact

**Repository**: https://github.com/brundanagaraj-07/Counterfactual-Driven-Agentic-AI-Framework

For issues or questions, open a GitHub issue.

---

**Last updated**: April 7, 2026
