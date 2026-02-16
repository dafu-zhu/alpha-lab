# US Equity Data Lake

[![Tests](https://github.com/dafu-zhu/us-equity-datalake/actions/workflows/tests.yml/badge.svg)](https://github.com/dafu-zhu/us-equity-datalake/actions/workflows/tests.yml)
[![Coverage](https://codecov.io/gh/dafu-zhu/us-equity-datalake/branch/main/graph/badge.svg)](https://codecov.io/gh/dafu-zhu/us-equity-datalake)

A self-hosted market data infrastructure for US equities. Downloads price data and financial statements from official sources, stores everything locally as Parquet files, and provides a Python API for quantitative research.

## What It Does

This project builds a local data lake containing:

| Data | Source | Coverage | Update Frequency |
|------|--------|----------|-----------------|
| **Daily prices** (OHLCV) | [Alpaca](https://alpaca.markets/) | 2017 -- present | On demand |
| **Fundamentals** (income, balance sheet, cash flow) | [SEC EDGAR](https://www.sec.gov/edgar) | 2009 -- present | On demand |
| **Security master** (symbol changes, delistings, GICS classification) | CRSP + SEC + Nasdaq + yfinance | 1986 -- present | On demand |
| **Universe snapshots** (top 3000 by liquidity, monthly) | Alpaca + SecurityMaster | 2017 -- present | On demand |
| **Feature wide tables** (time x security matrices for alpha research) | Derived from above | 2017 -- present | On demand |

All data lives on your local filesystem -- no cloud services required.

## Why

- **No survivorship bias** -- delisted and inactive stocks are retained
- **Security master tracks corporate actions** -- symbol changes, mergers, and delistings are handled automatically via `security_id` (a stable identifier that follows each company through ticker changes)
- **Structured for alpha research** -- pre-built wide tables (timestamp x security_id) for prices, volumes, fundamentals, and derived metrics
- **Official sources only** -- SEC EDGAR for fundamentals, Alpaca SIP feed for prices
- **Flat-file storage** -- everything is Parquet/Arrow, no database server needed

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

### 5. Use the data

```python
from quantdl.api.client import QuantDLClient

client = QuantDLClient(data_path="/path/to/your/data")

# Look up a symbol
info = client.lookup("AAPL")
# SecurityInfo(security_id=14593, symbol='AAPL', company='APPLE INC', ...)

# Get closing prices for specific stocks
df = client.get("close", symbols=["AAPL", "MSFT"], start="2024-01-01")

# Get all available features for the full universe
df = client.get("returns")
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
│       └── ...                            # ~30 feature fields total
```

**Key design decisions:**

- **One file per security** for raw data -- simple, no database needed, easy to inspect
- **security_id-based paths** -- stable across ticker changes (e.g., FB and META both map to the same security_id)
- **Arrow IPC for features** -- columnar format optimized for time-series slicing
- **Parquet for raw data** -- compact, widely supported, self-describing schema

## Feature Fields

Feature wide tables are time x security_id matrices stored as Arrow IPC files. Available fields:

| Category | Fields |
|----------|--------|
| **Price/Volume** | `close`, `volume`, `vwap`, `num_trades`, `returns`, `adv20`, `cap`, `split` |
| **Fundamental (raw)** | `sales`, `income`, `assets`, `equity`, `liabilities`, `cash`, `debt_lt`, `debt_st`, `cfo`, `sharesout`, `operating_income`, `cogs`, `sga_expense`, `depre_amort`, `inventory`, `assets_curr`, `liabilities_curr` |
| **Fundamental (derived)** | `ebitda`, `eps`, `bookvalue_ps`, `sales_ps`, `current_ratio`, `return_equity`, `return_assets`, `debt`, `working_capital`, `enterprise_value`, `invested_capital`, `operating_expense`, `inventory_turnover`, `sales_growth` |
| **Group** | `sector`, `industry`, `subindustry`, `exchange` |

## Project Structure

```
us-equity-datalake/
├── src/quantdl/
│   ├── api/                # Python API for querying data (QuantDLClient)
│   ├── collection/         # Data collectors (Alpaca prices, SEC fundamentals)
│   ├── features/           # Feature wide table builders
│   ├── master/             # Security master (tracks symbols across changes)
│   ├── storage/            # Upload pipeline (collect → validate → publish)
│   ├── universe/           # Universe management (current + historical)
│   ├── data/               # Bundled data (source security_master.parquet)
│   └── utils/              # Logging, rate limiting, mapping
├── scripts/                # Utility scripts (WRDS build, XBRL harvesting)
├── configs/                # SEC field mappings, GICS classification
└── tests/                  # Unit + integration tests (854 tests)
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

## Acknowledgments

- [SEC EDGAR](https://www.sec.gov/edgar) for free access to financial statements
- [Alpaca](https://alpaca.markets/) for free market data API
- [Nasdaq](https://www.nasdaqtrader.com/) for reference data
- [WRDS/CRSP](https://wrds-www.wharton.upenn.edu/) for historical security master data
