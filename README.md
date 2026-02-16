# US Equity Data Lake

[![Tests](https://github.com/dafu-zhu/us-equity-datalake/actions/workflows/tests.yml/badge.svg)](https://github.com/dafu-zhu/us-equity-datalake/actions/workflows/tests.yml)
[![Coverage](https://codecov.io/gh/dafu-zhu/us-equity-datalake/branch/main/graph/badge.svg)](https://codecov.io/gh/dafu-zhu/us-equity-datalake)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

A local-first alpha research platform that brings [WorldQuant BRAIN](https://platform.worldquantbrain.com/)-style development to your own machine — same expression syntax, same operators, no latency, no transfer costs, no platform limits.

The long-term goal is a **full local proxy of WorldQuant BRAIN** — data, operators, simulation, and backtesting — to support rapid alpha research and development.

Get started with `uv sync`:

```python
from quantdl.api.client import QuantDLClient

client = QuantDLClient(data_path="/path/to/your/data")

alpha = client.query("""
regime = ts_rank(ts_sum(volume, 5)/ts_sum(volume, 60), 60) > 0.5;
fundamental = group_rank(ts_delta(income/sharesout, 21), subindustry);
timing = group_rank(-ts_delta(vwap, 5), subindustry);
trade_when(regime, 2*fundamental + timing, -1)
""")
```

You can find more about the API in the [API reference](docs/API.md), and the full expression syntax in the [expression guide](docs/ALPHA-GUIDE.md).

## Why Local

- **Zero latency** — data lives on disk, not across a wire. Expression evaluation runs in-process on Arrow columnar memory.
- **Zero cost** — no API metering, no compute credits, no data transfer fees. All sources are free ([Alpaca](https://alpaca.markets/), [SEC EDGAR](https://www.sec.gov/edgar)).
- **No platform limits** — unlimited simulations, no queue, no daily submission caps. Iterate as fast as your machine allows.
- **Survivorship-bias-free** — 20,000+ securities tracked back to 1986, including delisted stocks, symbol changes, and mergers.

## What's Inside

The platform provides **48 data fields** (price/volume, 30+ fundamentals, derived ratios, GICS classifications) and **80+ operators** across time-series, cross-sectional, group, arithmetic, logical, and transformational categories — aligned with [WQ BRAIN operator conventions](https://platform.worldquantbrain.com/learn/documentation/discover-brain/operators-702).

You can find the complete list of available data fields in [docs/FIELDS.md](docs/FIELDS.md), and the full operator reference in [docs/OPERATORS.md](docs/OPERATORS.md).

Data is sourced from [Alpaca](https://alpaca.markets/) (daily OHLCV, 2017+), [SEC EDGAR](https://www.sec.gov/edgar) (fundamentals, 2009+), and a built-in security master derived from CRSP, SEC, Nasdaq, and yfinance (1986+). All data is stored locally as Parquet/Arrow files — no database server needed.

You can find more about the storage layout in [docs/STORAGE.md](docs/STORAGE.md).

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

Please find more about the environment variables in [`.env.example`](.env.example).

### Build & Download

```bash
uv run qdl --master                         # Build security master + calendar
uv run qdl --all --start 2017 --end 2025    # Download everything (~45 min first run)
```

Or pick what you need: `--ticks`, `--fundamental`, `--top-3000`, `--features`. See the [CLI reference](docs/CLI.md) for all options.

### Research

```python
client = QuantDLClient(data_path="/path/to/your/data")

# Look up any symbol (resolves across ticker changes)
client.lookup("AAPL")

# Load pre-built feature tables
client.get("close", symbols=["AAPL", "MSFT"], start="2024-01-01")

# Alpha expressions — same syntax as WorldQuant BRAIN
client.query("rank(-ts_delta(close, 5))")
```

## Documentation

- [API Reference](docs/API.md) — `QuantDLClient` methods: `get()`, `query()`, `lookup()`, `universe()`
- [Operators](docs/OPERATORS.md) — Full reference for all 80+ operators with signatures and examples
- [Data Fields](docs/FIELDS.md) — 48 fields with categories, XBRL mappings, and WQ BRAIN equivalents
- [Expression Guide](docs/ALPHA-GUIDE.md) — How to write alpha expressions and multi-line queries
- [CLI Reference](docs/CLI.md) — All `qdl` command options
- [Storage Layout](docs/STORAGE.md) — Directory structure and design decisions
- [Testing](tests/README.md) — Test structure, markers, and fixtures

## Roadmap

This project is under active development toward a full local WorldQuant BRAIN experience.

- [x] **Data engine** — 48 fields from Alpaca + SEC EDGAR, survivorship-bias-free
- [x] **Alpha engine** — 80+ operators, WQ-style expression parser, auto-field loading
- [ ] **Backtest engine** — Simulate alpha PnL, Sharpe, turnover, drawdown, and fitness metrics ([WQ reference](https://platform.worldquantbrain.com/learn/documentation/interpret-results/alpha-submission))
- [ ] **Alpha diagnostics** — Decay analysis, correlation matrix, self-correlation
- [ ] **WQ field translation** — Auto-translate local expressions to WQ BRAIN field names for direct copy-paste submission
- [ ] **More data fields** — Short interest, analyst estimates, ETF holdings, options-derived
- [ ] **Multi-region support** — Extend beyond US equities

## Development

```bash
uv run pytest                    # All tests
uv run pytest -m unit            # Unit tests only (~14s)
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
