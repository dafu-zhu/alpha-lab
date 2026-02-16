# US Equity Data Lake

[![Tests](https://github.com/dafu-zhu/us-equity-datalake/actions/workflows/tests.yml/badge.svg)](https://github.com/dafu-zhu/us-equity-datalake/actions/workflows/tests.yml)
[![Coverage](https://codecov.io/gh/dafu-zhu/us-equity-datalake/branch/main/graph/badge.svg)](https://codecov.io/gh/dafu-zhu/us-equity-datalake)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

A local-first alpha research platform that brings [WorldQuant BRAIN](https://platform.worldquantbrain.com/)-style alpha development to your own machine. Write alpha expressions, evaluate them against 48 data fields using 80+ operators, and iterate without latency, transfer costs, or platform limits.

The long-term goal is a **full local proxy of WorldQuant BRAIN** — data, operators, simulation, and backtesting — to support rapid alpha research and development.

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

## Why Local

- **Zero latency** — data lives on disk, not across a wire. Expression evaluation runs in-process on Arrow columnar memory.
- **Zero cost** — no API metering, no compute credits, no data transfer fees. All sources are free (Alpaca, SEC EDGAR).
- **No platform limits** — unlimited simulations, no queue, no daily submission caps. Iterate as fast as your machine allows.
- **Survivorship-bias-free** — 20,000+ securities tracked back to 1986, including delisted stocks, symbol changes, and mergers.

## Overview

| Layer | What | Details |
|-------|------|---------|
| **Data engine** | Builds a local data lake from official sources | [Data fields](docs/FIELDS.md) · [CLI reference](docs/CLI.md) · [Storage layout](docs/STORAGE.md) |
| **Alpha engine** | WQ BRAIN-style expression language | [Operators](docs/OPERATORS.md) · [API reference](docs/API.md) · [Expression guide](docs/ALPHA-GUIDE.md) |
| **Backtest engine** | Simulate, evaluate, and compare alphas | _Coming soon_ — [Roadmap](#roadmap) |

### Data Sources

| Data | Source | Coverage |
|------|--------|----------|
| Daily prices (OHLCV) | [Alpaca](https://alpaca.markets/) | 2017 -- present |
| Fundamentals (income, balance sheet, cash flow) | [SEC EDGAR](https://www.sec.gov/edgar) | 2009 -- present |
| Security master (symbol changes, delistings, GICS) | CRSP + SEC + Nasdaq + yfinance | 1986 -- present |
| Universe snapshots (top 3000 by liquidity) | Alpaca + SecurityMaster | 2017 -- present |

### Operators (80+)

| Category | Count | Examples |
|----------|-------|---------|
| Time-series | 27 | `ts_rank`, `ts_sum`, `ts_delta`, `ts_regression`, `ts_corr`, `ts_decay_linear` |
| Cross-sectional | 7 | `rank`, `zscore`, `scale`, `quantile`, `winsorize`, `normalize` |
| Group | 6 | `group_rank`, `group_neutralize`, `group_zscore`, `group_backfill` |
| Arithmetic | 17 | `abs`, `log`, `sqrt`, `power`, `signed_power`, `densify` |
| Logical | 11 | `if_else`, `and_`, `or_`, `lt`, `gt`, `eq`, `is_nan` |
| Transformational | 1 | `trade_when` (entry/exit/carry-forward) |
| Vector | 2 | `vec_avg`, `vec_sum` |

Operator signatures follow [WQ BRAIN conventions](https://platform.worldquantbrain.com/learn/documentation/discover-brain/operators-702). Full reference: [docs/OPERATORS.md](docs/OPERATORS.md)

### Data Fields (48)

| Category | Count | Examples |
|----------|-------|---------|
| Price/Volume | 10 | `close`, `open`, `high`, `low`, `volume`, `vwap`, `returns`, `adv20`, `cap` |
| Fundamental | 20+ | `sales`, `income`, `assets`, `equity`, `sharesout`, `operating_income`, `cogs`, `debt_lt`, `cashflow_op` |
| Derived | 14 | `ebitda`, `eps`, `return_equity`, `enterprise_value`, `current_ratio`, `sales_growth` |
| Group | 4 | `sector`, `industry`, `subindustry`, `exchange` |

Field names align with [WQ BRAIN data conventions](https://platform.worldquantbrain.com/learn/documentation/discover-brain/data) where possible. Full reference with XBRL mappings: [docs/FIELDS.md](docs/FIELDS.md)

## Quick Start

### Prerequisites

- Python 3.12+, [uv](https://docs.astral.sh/uv/)
- Free [Alpaca](https://app.alpaca.markets/signup) account (prices + calendar)
- Email address (for [SEC EDGAR](https://www.sec.gov/os/webmaster-faq#code-support) User-Agent)

### Install

```bash
git clone https://github.com/dafu-zhu/us-equity-datalake.git
cd us-equity-datalake
uv sync
cp .env.example .env   # then fill in credentials
```

### Build & Download

```bash
uv run qdl --master                         # Build security master + calendar
uv run qdl --all --start 2017 --end 2025    # Download everything (~45 min first run)
```

Or pick what you need:

```bash
uv run qdl --ticks          # Daily OHLCV prices
uv run qdl --fundamental    # SEC financial statements
uv run qdl --top-3000       # Monthly universe snapshots
uv run qdl --features       # Build feature wide tables
```

### Research

```python
from quantdl.api.client import QuantDLClient

client = QuantDLClient(data_path="/path/to/your/data")

# Look up any symbol (resolves across ticker changes)
client.lookup("AAPL")

# Load pre-built feature tables
prices = client.get("close", symbols=["AAPL", "MSFT"], start="2024-01-01")

# Alpha expressions — same syntax as WorldQuant BRAIN
client.query("rank(-ts_delta(close, 5))")

client.query("""
raw = ts_delta(close, 10) / ts_std(returns, 20);
group_neutralize(rank(raw), sector)
""")
```

See the [API reference](docs/API.md) and [expression guide](docs/ALPHA-GUIDE.md) for more.

## Documentation

| Doc | Description |
|-----|-------------|
| [API Reference](docs/API.md) | `QuantDLClient` methods — `get()`, `query()`, `lookup()`, `universe()` |
| [Operators](docs/OPERATORS.md) | Full operator reference with signatures and examples |
| [Data Fields](docs/FIELDS.md) | 48 fields with categories, XBRL mappings, and WQ BRAIN equivalents |
| [Expression Guide](docs/ALPHA-GUIDE.md) | How to write alpha expressions, multi-line syntax, auto-field loading |
| [CLI Reference](docs/CLI.md) | All `qdl` command options |
| [Storage Layout](docs/STORAGE.md) | Directory structure and design decisions |
| [Testing](tests/README.md) | Test structure, markers, and fixtures |
| [Feature Registry](src/quantdl/features/registry.py) | Source of truth for all field definitions |
| [Operator Source](src/quantdl/api/operators/) | Operator implementations with docstrings |

## Project Structure

```
src/quantdl/
├── alpha/          Expression parser + Alpha class (WQ-style formulas)
├── api/            QuantDLClient + 80 operators (ts_*, rank, group_*, ...)
│   └── operators/  Time-series, cross-sectional, group, arithmetic, logical
├── collection/     Data collectors (Alpaca prices, SEC fundamentals)
├── features/       Feature wide table builders + field registry (48 fields)
├── master/         Security master (tracks 20k+ symbols across changes)
├── storage/        Upload pipeline (collect → validate → publish)
├── universe/       Universe management (current + historical)
└── data/           Bundled data (source security_master.parquet)
```

## Roadmap

This project is under active development toward a full local WorldQuant BRAIN experience.

- [x] **Data engine** — 48 fields from Alpaca + SEC EDGAR, survivorship-bias-free
- [x] **Alpha engine** — 80+ operators, WQ-style expression parser, auto-field loading
- [ ] **Backtest engine** — Simulate alpha PnL, Sharpe, turnover, drawdown, and fitness metrics ([WQ reference](https://platform.worldquantbrain.com/learn/documentation/interpret-results/alpha-submission))
- [ ] **Alpha diagnostics** — Decay analysis, correlation matrix, self-correlation
- [ ] **WQ field mapping API** — Programmatic mapping between local field names and WQ BRAIN field IDs
- [ ] **More data fields** — Short interest, analyst estimates, ETF holdings, options-derived
- [ ] **Multi-region support** — Extend beyond US equities

## Development

```bash
uv run pytest                 # All tests
uv run pytest -m unit         # Unit tests only (~14s)
uv run pytest --cov=src/quantdl  # With coverage
```

<details>
<summary>Rebuilding the security master from WRDS</summary>

The bundled `security_master.parquet` is pre-built from CRSP via WRDS. To rebuild from scratch (requires [WRDS](https://wrds-www.wharton.upenn.edu/) account):

```bash
export WRDS_USERNAME=your_username
export WRDS_PASSWORD=your_password
uv add wrds
uv run python scripts/build_security_master.py
uv remove wrds
```

</details>

## Acknowledgments

[SEC EDGAR](https://www.sec.gov/edgar) · [Alpaca](https://alpaca.markets/) · [Nasdaq](https://www.nasdaqtrader.com/) · [WRDS/CRSP](https://wrds-www.wharton.upenn.edu/) · [WorldQuant BRAIN](https://platform.worldquantbrain.com/) (inspiration)
