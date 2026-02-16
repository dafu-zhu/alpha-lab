# US Equity Data Lake

[![Tests](https://github.com/dafu-zhu/us-equity-datalake/actions/workflows/tests.yml/badge.svg)](https://github.com/dafu-zhu/us-equity-datalake/actions/workflows/tests.yml)
[![Coverage](https://codecov.io/gh/dafu-zhu/us-equity-datalake/branch/main/graph/badge.svg)](https://codecov.io/gh/dafu-zhu/us-equity-datalake)

A local-first alpha research platform inspired by [WorldQuant BRAIN](https://platform.worldquantbrain.com/). Build and test alpha factors using a familiar expression syntax — with 80+ operators, 48 data fields, and your own data.

```python
from quantdl.api.client import QuantDLClient

client = QuantDLClient(data_path="/path/to/your/data")

# Write alphas just like on WorldQuant BRAIN
alpha = client.query("""
regime = ts_rank(ts_sum(volume, 5)/ts_sum(volume, 60), 60) > 0.5;
fundamental = group_rank(ts_delta(income/sharesout, 21), subindustry);
timing = group_rank(-ts_delta(vwap, 5), subindustry);
trade_when(regime, 2*fundamental + timing, -1)
""")
```

## What It Does

The platform has two layers — a **data engine** that builds a survivorship-bias-free data lake from official sources, and an **alpha engine** that lets you express and evaluate trading signals using a WorldQuant-style formula language.

### Data Engine

| Data | Source | Coverage | Update Frequency |
|------|--------|----------|-----------------|
| **Daily prices** (OHLCV) | [Alpaca](https://alpaca.markets/) | 2017 -- present | On demand |
| **Fundamentals** (income, balance sheet, cash flow) | [SEC EDGAR](https://www.sec.gov/edgar) | 2009 -- present | On demand |
| **Security master** (symbol changes, delistings, GICS classification) | CRSP + SEC + Nasdaq + yfinance | 1986 -- present | On demand |
| **Universe snapshots** (top 3000 by liquidity, monthly) | Alpaca + SecurityMaster | 2017 -- present | On demand |
| **Feature wide tables** (time x security matrices for alpha research) | Derived from above | 2017 -- present | On demand |

All data lives on your local filesystem — no cloud services required.

### Alpha Engine

| | Count | Examples |
|--|-------|---------|
| **Time-series operators** | 27 | `ts_rank`, `ts_sum`, `ts_delta`, `ts_regression`, `ts_corr`, `ts_decay_linear` |
| **Cross-sectional operators** | 7 | `rank`, `zscore`, `scale`, `quantile`, `winsorize`, `normalize` |
| **Group operators** | 6 | `group_rank`, `group_neutralize`, `group_zscore`, `group_backfill` |
| **Arithmetic operators** | 17 | `abs`, `log`, `sqrt`, `power`, `signed_power`, `densify` |
| **Logical operators** | 11 | `if_else`, `and_`, `or_`, `lt`, `gt`, `eq`, `is_nan` |
| **Transformational** | 1 | `trade_when` (entry/exit/carry-forward) |
| **Vector** | 2 | `vec_avg`, `vec_sum` |
| **Data fields** | 48 | price, volume, 30+ fundamental, derived ratios, GICS groups |

> **This project is under active development.** More operators, fields, and simulation tools are on the way.

## Why

- **WorldQuant-style expression language** — write alphas as formula strings (`rank(-ts_delta(close, 5))`), not boilerplate code
- **No survivorship bias** — delisted and inactive stocks are retained; the security master tracks 20,000+ securities back to 1986
- **Security master tracks corporate actions** — symbol changes, mergers, and delistings are handled automatically via `security_id` (a stable identifier that follows each company through ticker changes)
- **48 data fields, 80+ operators** — price/volume, 30+ fundamental items, derived ratios, and GICS group classifications, all aligned as wide matrices ready for cross-sectional analysis
- **Official sources only** — SEC EDGAR for fundamentals, Alpaca SIP feed for prices
- **Flat-file storage** — everything is Parquet/Arrow, no database server needed

## Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- Free [Alpaca](https://app.alpaca.markets/signup) account (for price data + trading calendar)
- Email address (for SEC EDGAR User-Agent, [required by SEC](https://www.sec.gov/os/webmaster-faq#code-support))

### 1. Clone and install

```bash
git clone https://github.com/dafu-zhu/us-equity-datalake.git
cd us-equity-datalake
uv sync
```

### 2. Configure environment

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

```env
# Where to store all downloaded data (required)
LOCAL_STORAGE_PATH=/path/to/your/data

# Alpaca API (free account, required for prices + calendar)
ALPACA_API_KEY=your_key
ALPACA_API_SECRET=your_secret

# SEC EDGAR (just your email, no account needed)
SEC_USER_AGENT=your_name@example.com

# OpenFIGI (optional, for symbol rebrand detection)
OpenFIGI_API_KEY=your_key
```

### 3. Build the security master

The security master is the foundation -- it maps every stock to a stable `security_id` that persists across ticker changes. A pre-built copy ships with the package, but you need to create a working copy and update it with current data:

```bash
uv run qdl --master
```

This will:
1. Copy the bundled `security_master.parquet` to your `LOCAL_STORAGE_PATH`
2. Update it with current exchange listings from SEC
3. Detect new IPOs from Nasdaq
4. Classify sectors/industries via yfinance
5. Build the NYSE trading calendar from Alpaca

### 4. Download data

```bash
# Download everything (ticks + fundamentals + universe + features)
uv run qdl --all --start 2017 --end 2025

# Or download specific data types:
uv run qdl --ticks                          # Daily OHLCV prices
uv run qdl --fundamental                    # SEC financial statements
uv run qdl --top-3000                       # Monthly universe snapshots
uv run qdl --features                       # Build feature wide tables
```

The first full download (2017--2025) takes roughly 10--15 minutes for prices and 30--60 minutes for fundamentals.

### 5. Research

```python
from quantdl.api.client import QuantDLClient

client = QuantDLClient(data_path="/path/to/your/data")

# --- Data access ---

info = client.lookup("AAPL")
# SecurityInfo(security_id=14593, symbol='AAPL', company='APPLE INC', ...)

prices = client.get("close", symbols=["AAPL", "MSFT"], start="2024-01-01")
returns = client.get("returns")

# --- Alpha expressions (WorldQuant BRAIN style) ---

# Simple momentum alpha
alpha = client.query("rank(-ts_delta(close, 5))")

# Mean reversion with volatility scaling
alpha = client.query("rank(-(close - ts_mean(close, 20)) / ts_std(close, 20))")

# Multi-line with sector neutralization
alpha = client.query("""
raw = ts_delta(close, 10) / ts_std(returns, 20);
group_neutralize(rank(raw), sector)
""")

# Fundamental + price composite
alpha = client.query("""
value = group_rank(ts_delta(income/sharesout, 63), subindustry);
momentum = group_rank(-ts_delta(vwap, 5), subindustry);
scale(value + momentum)
""")
```

## CLI Reference

```
uv run qdl [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--master` | | Build security master + trading calendar |
| `--all` | | Master build + download all data types |
| `--ticks` | | Download daily OHLCV prices |
| `--fundamental` | | Download SEC fundamental data |
| `--top-3000` | | Download monthly universe snapshots |
| `--features` | | Build feature wide tables from raw data |
| `--start` | 2017 | Start year |
| `--end` | 2025 | End year |
| `--overwrite` | false | Re-download even if data exists |
| `--daily-chunk-size` | 200 | Symbols per Alpaca API batch |
| `--daily-sleep-time` | 0.2 | Seconds between API batches |
| `--max-workers` | 50 | Parallel workers for fundamentals |

## Data Layout

All data is stored under `$LOCAL_STORAGE_PATH`:

```
$LOCAL_STORAGE_PATH/
├── data/
│   ├── meta/
│   │   ├── master/
│   │   │   ├── security_master.parquet    # Symbol-to-security_id mapping
│   │   │   ├── calendar_master.parquet    # NYSE trading days
│   │   │   └── prev_universe.json         # Previous universe snapshot
│   │   └── universe/{YYYY}/{MM}/
│   │       └── top3000.txt                # Monthly top 3000 symbols
│   ├── raw/
│   │   ├── ticks/daily/{security_id}/
│   │   │   └── ticks.parquet              # OHLCV: timestamp, close, volume, num_trades, vwap
│   │   └── fundamental/{security_id}/
│   │       └── fundamental.parquet        # Long table: concept, value, as_of_date, ...
│   └── features/
│       ├── close.arrow                    # Wide table: Date x security_ids
│       ├── volume.arrow
│       ├── returns.arrow
│       ├── sales.arrow
│       ├── sector.arrow                   # GICS sector labels
│       └── ...                            # 48 feature fields total
```

**Key design decisions:**

- **One file per security** for raw data -- simple, no database needed, easy to inspect
- **security_id-based paths** -- stable across ticker changes (e.g., FB and META both map to the same security_id)
- **Arrow IPC for features** -- columnar format optimized for time-series slicing
- **Parquet for raw data** -- compact, widely supported, self-describing schema

## Data Fields (48)

All fields are pre-built as wide matrices (Date x security_id) in Arrow IPC format — ready to use directly in alpha expressions.

| Category | Count | Fields |
|----------|-------|--------|
| **Price/Volume** | 10 | `close`, `open`, `high`, `low`, `volume`, `vwap`, `returns`, `adv20`, `cap`, `split` |
| **Fundamental** | 20+ | `sales`, `income`, `assets`, `equity`, `liabilities`, `cash`, `sharesout`, `operating_income`, `cogs`, `sga_expense`, `depre_amort`, `inventory`, `debt_lt`, `debt_st`, `assets_curr`, `liabilities_curr`, `cashflow_op`, `capex`, `pretax_income`, ... |
| **Derived ratios** | 14 | `ebitda`, `eps`, `bookvalue_ps`, `sales_ps`, `current_ratio`, `return_equity`, `return_assets`, `debt`, `working_capital`, `enterprise_value`, `invested_capital`, `operating_expense`, `inventory_turnover`, `sales_growth` |
| **Group** | 4 | `sector`, `industry`, `subindustry`, `exchange` |

> Field names are aligned with WorldQuant BRAIN conventions where possible. See the [feature registry](src/quantdl/features/registry.py) for the complete list with XBRL mappings.

## Project Structure

```
us-equity-datalake/
├── src/quantdl/
│   ├── alpha/              # Expression parser + Alpha class (WQ-style formulas)
│   ├── api/                # QuantDLClient + 80 operators (ts_*, rank, group_*, ...)
│   │   └── operators/      # Time-series, cross-sectional, group, arithmetic, logical
│   ├── collection/         # Data collectors (Alpaca prices, SEC fundamentals)
│   ├── features/           # Feature wide table builders + field registry (48 fields)
│   ├── master/             # Security master (tracks 20k+ symbols across changes)
│   ├── storage/            # Upload pipeline (collect → validate → publish)
│   ├── universe/           # Universe management (current + historical)
│   ├── data/               # Bundled data (source security_master.parquet)
│   └── utils/              # Logging, rate limiting, mapping
├── scripts/                # Utility scripts (WRDS build, XBRL harvesting)
├── configs/                # SEC field mappings, GICS classification
└── tests/                  # Unit + integration tests (870+ tests)
```

## Development

```bash
# Run all tests
uv run pytest

# Run unit tests only (fast, ~14s)
uv run pytest -m unit

# Run a single test file
uv run pytest tests/unit/collection/test_alpaca_ticks.py

# Run with coverage
uv run pytest --cov=src/quantdl
```

## Rebuilding the Security Master from WRDS

The bundled `security_master.parquet` is pre-built from CRSP via WRDS. To rebuild it from scratch (requires a [WRDS](https://wrds-www.wharton.upenn.edu/) account):

```bash
# Set WRDS credentials
export WRDS_USERNAME=your_username
export WRDS_PASSWORD=your_password

# Install wrds (not a runtime dependency)
uv add wrds

# Rebuild
uv run python scripts/build_security_master.py

# Remove wrds when done
uv remove wrds
```

## Roadmap

This project is under active development. Planned additions include:

- More data fields (short interest, analyst estimates, ETF holdings)
- Backtest simulation engine with PnL, turnover, and drawdown analysis
- Alpha decay and correlation diagnostics
- Multi-region support

## Acknowledgments

- [SEC EDGAR](https://www.sec.gov/edgar) for free access to financial statements
- [Alpaca](https://alpaca.markets/) for free market data API
- [Nasdaq](https://www.nasdaqtrader.com/) for reference data
- [WRDS/CRSP](https://wrds-www.wharton.upenn.edu/) for historical security master data
