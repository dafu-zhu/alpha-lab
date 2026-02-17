# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AlphaLab: Local-first alpha research platform inspired by WorldQuant BRAIN. Self-hosted market data infrastructure for US equities using official/authoritative sources. Data stored in flat-file structure on local filesystem.

**Data Coverage:**
- Daily ticks (OHLCV): Alpaca API (2017+)
- Fundamentals: SEC EDGAR JSON API (2009+)
- Security master: Local parquet (data/meta/master/security_master.parquet, checked into repo)

## Commands

### Environment Setup
```bash
uv sync

# Required .env variables (see .env.example):
# ALPACA_API_KEY, ALPACA_API_SECRET (for ticks)
# SEC_USER_AGENT (for EDGAR API, e.g., your_name@example.com)
# LOCAL_STORAGE_PATH=/path/to/data (required)
```

**No linting/formatting tools configured.** Style enforcement is manual.
Code quality tools: `uv`, `pytest`. Avoid: `pip`, `black`, `flake8`.

### Testing
```bash
uv run pytest --cov=src/alphalab     # All tests with coverage
uv run pytest -m unit               # Unit tests only (fast)
uv run pytest -m integration        # Integration tests only
uv run pytest tests/unit/collection/test_fundamental.py  # Single file
uv run pytest -n auto               # Parallel execution
```

### Data Operations
```bash
# Build security master + calendar
alab --master

# Full backfill (master + all data types)
alab --all --start 2017 --end 2025

# Download specific data types
alab --ticks
alab --fundamental
alab --top-3000
alab --features

# With options
alab --ticks --overwrite --daily-chunk-size 100 --daily-sleep-time 0.5
alab --fundamental --max-workers 25
```

## Architecture

### Core Components

**1. Collection Layer** (`src/alphalab/collection/`)
- `alpaca_ticks.py`: Fetches daily OHLCV data from Alpaca API
- `fundamental.py`: Fetches SEC EDGAR XBRL data (JSON API)
- `models.py`: Data models (TickField, FndDataPoint, DataSource)

**2. Storage Layer** (`src/alphalab/storage/`)
- `pipeline/collectors.py`: Collects data from sources, handles rate limiting
- `pipeline/publishers.py`: Publishes collected data to local storage in Parquet format
- `pipeline/validation.py`: Validates uploaded data completeness
- `clients/local.py`: `LocalStorageClient` (aliased as `StorageClient`) — duck-typed boto3 S3 client for local filesystem (metadata in sidecar `.{filename}.metadata.json`)
- `clients/ticks.py`: `TicksClient` — symbol-based query API with transparent security_id resolution
- `handlers/ticks.py`: `DailyTicksHandler` — orchestrates daily tick uploads per year
- `handlers/fundamental.py`: `FundamentalHandler` — orchestrates fundamental uploads with CIK resolution
- `handlers/top3000.py`: `Top3000Handler` — orchestrates monthly universe uploads
- `utils/cik_resolver.py`: Maps tickers to SEC CIK codes
- `utils/rate_limiter.py`: Rate limiting for API calls

**3. Universe Management** (`src/alphalab/universe/`)
- `current.py`: Fetches current stock universe from Nasdaq Trader
- `historical.py`: Historical universe from local security master parquet
- `manager.py`: Manages universe state, handles symbol changes

**4. Security Master** (`src/alphalab/master/`)
- `security_master.py`: Tracks stocks across symbol changes, mergers, delistings
  - Loads from local parquet only (no WRDS dependency at runtime)
  - Source parquet at `data/meta/master/security_master.parquet` (git-tracked)
  - Working copy at `$LOCAL_STORAGE_PATH/data/meta/master/security_master.parquet`
  - Constructor: `SecurityMaster(local_path=None)` — raises FileNotFoundError if missing
  - Single `update()` method: SEC exchange mapping → Nasdaq universe → OpenFIGI rebrand detection → yfinance GICS classification
  - Schema: security_id, permno, symbol, company, cik, start_date, end_date, exchange, sector, industry, subindustry
- Contains `SymbolNormalizer` for deterministic ticker format conversion
- `configs/morningstar_to_gics.yaml`: Maps yfinance (Morningstar) sector/industry → GICS classification
- `scripts/build_security_master.py` (gitignored): Standalone WRDS build with Compustat GICS

**5. Feature Pipeline** (`src/alphalab/features/`)
- `registry.py`: Central field registry (FieldSpec, ALL_FIELDS, VALID_FIELD_NAMES, get_build_order)
- `builder.py`: `FeatureBuilder` orchestrator — builds all features in dependency order
- `builders/ticks.py`: `TicksFeatureBuilder` — wide tables from raw ticks
- `builders/fundamental.py`: `FundamentalFeatureBuilder` — wide tables with forward-fill
- `builders/groups.py`: `GroupFeatureBuilder` — GICS/exchange group masks
- Output: `data/features/{field}.arrow` (Arrow IPC, cols: timestamp + security_ids)

**6. Upload App** (`src/alphalab/storage/`)
- `app.py`: `UploadApp` orchestrates full backfill uploads
- `handlers/features.py`: `FeaturesHandler` — builds feature wide tables

**7. Unified CLI** (`src/alphalab/cli.py`)
- Entry point: `alab` command
- Commands: `--master`, `--all`, `--ticks`, `--fundamental`, `--top-3000`, `--features`

### Data Flow

```
Data Sources → Collection → Validation → Local Storage
```

**Upload Flow:**
1. `UploadApp` fetches universe from Nasdaq FTP (current) or local security master (historical)
2. `DataCollectors` fetch data from sources (Alpaca, SEC)
3. `DataPublishers` write Parquet files to local storage
4. `Validator` checks completeness

### Storage Paths

```
$LOCAL_STORAGE_PATH/
├── data/
│   ├── meta/
│   │   ├── master/
│   │   │   ├── calendar_master.parquet
│   │   │   ├── security_master.parquet
│   │   │   └── prev_universe.json
│   │   └── universe/{YYYY}/{MM}/top3000.txt
│   ├── raw/
│   │   ├── ticks/daily/{security_id}/ticks.parquet
│   │   └── fundamental/{security_id}/fundamental.parquet
│   └── features/{field}.arrow                          ← Arrow IPC wide tables
```

**Storage Strategy:**
- Daily ticks: Single file per security_id, append-merge on upload (read existing → filter year overlap → concat → write)
- Fundamentals: Single file per security_id (keyed by security_id, CIK retained in metadata)
- Universe: Monthly top 3000 symbol lists under `data/meta/universe/`
- Security master: Under `data/meta/master/`

### Key Design Decisions

**1. Security ID-Based Storage**
- Both daily ticks and fundamentals stored under security_id (not symbol/CIK)
- SecurityMaster resolves symbol+date → security_id (tracks business continuity)
- TicksClient provides symbol-based API with transparent security_id resolution
- Session-based caching for symbol lookups (cleared on client reinitialization)

**2. CIK for SEC Collection**
- CIK codes still required for SEC EDGAR API calls (collection layer)
- `CIKResolver` maps tickers to CIKs using SEC company tickers JSON
- CIK stored in local metadata alongside symbol

**3. Symbol Normalization**
- `SymbolNormalizer` converts between symbol formats (e.g., BRKB → BRK.B)
- Uses `SecurityMaster` to verify same security_id before conversion
- Keeps delisted stocks in original format

**4. Parallel Processing**
- Daily ticks: Batch processing with rate limiting (200 symbols/batch, Alpaca)
- Fundamentals: ThreadPoolExecutor with rate limiting (EDGAR API limits)

## Testing

### Test Structure
```
tests/
├── unit/          # Fast, isolated tests (mocked external calls)
├── integration/   # Tests with real storage/SEC (slower)
└── conftest.py    # Shared fixtures, auto-marks unit/integration
```

### Test Markers
- `@pytest.mark.unit`: Fast unit tests (auto-applied to tests/unit/**)
- `@pytest.mark.integration`: Integration tests (auto-applied to tests/integration/**)
- `@pytest.mark.slow`: Slow tests
- `@pytest.mark.external`: Requires external API access

### Common Fixtures (conftest.py)
- `sample_ticker`, `sample_tickers`: Test tickers
- `sample_date`, `sample_date_range`: Test dates
- `sample_year`: Test year (2024)
- `sample_cik`: Apple's CIK (0000320193)

## Critical Files & Configs

**configs/sec_mapping.yaml**
- Maps standardized field names to XBRL tags (SEC EDGAR)
- Used by `fundamental.py` to extract XBRL concepts
- Multiple candidate tags per concept (handles deprecated tags)

**DURATION_CONCEPTS (fundamental.py)**
- Duration concepts (income statement): rev, net_inc, cfo, etc.
- Instant concepts (balance sheet): assets, liab, equity, etc.
- Determines quarterly filtering logic

## Common Patterns

### Adding New Data Collector
1. Create collector class in `collection/` inheriting `DataCollector`
2. Add collector initialization in `DataCollectors` class
3. Add publisher method in `DataPublishers` class
4. Add CLI flags to `storage/cli.py` and `storage/app.py`

### Modifying Storage Paths
1. Update path in `pipeline/publishers.py` (publish methods)
2. Update path in `pipeline/validation.py` (data_exists)
3. Update corresponding client/handler if applicable

## Known Limitations

1. **Fundamentals:** ~75% small-cap coverage (SEC filing requirements)
2. **Historical index constituents:** Not available (survivorship bias)
3. **Fundamental lag:** 45-90 days (SEC filing deadlines)
4. **Daily tick data:** Starts from 2017 (Alpaca limitation)

## Edge Cases

### Symbol Changes
- SecurityMaster tracks security_id across ticker changes
- SymbolNormalizer prevents false matches (e.g., delisted ABCD ≠ ABC.D)
- Historical data kept under original ticker for delisted stocks

### Corporate Actions
- Fundamentals handle share splits via XBRL context (shares outstanding)
- Daily ticks include both raw and adjusted close prices

### Missing Data
- Trading halts: Skip (don't interpolate)
- Data source unavailable: Retry with exponential backoff
- Fundamental filing delays: Expected, tracked separately

## GitHub Actions

| Workflow | Trigger | Description |
|----------|---------|-------------|
| `tests.yml` | Push, PR | pytest + coverage → Codecov (Python 3.12, Ubuntu) |

Secrets documented in `.github/SECRETS.md`.

## Utility Scripts

**Note:** `scripts/` is gitignored. These are one-time/utility scripts not needed at runtime.

- `scripts/build_security_master.py`: Standalone WRDS build → source parquet (Compustat GICS, exchange codes)
- `scripts/xbrl_tag_factory.py`: XBRL tag harvesting, outputs `configs/mapping_suggestions.yaml`
- `scripts/generate_monthly_top3000.py`: Top 3000 universe per month (liquidity-ranked)
- `scripts/test_local_storage.py`: Tests `LocalStorageClient` filesystem backend
- `scripts/trade_calendar.py`: NYSE trading calendar utilities
- `scripts/benchmark_parallel_filing.py`: Benchmarks parallel SEC filing fetches

## Dependencies

Key libraries:
- **polars**: DataFrame processing (faster than pandas for large datasets)
- **requests**: HTTP client for Alpaca, SEC EDGAR
- **pytest**: Testing framework
- **pyarrow**: Parquet I/O

## Performance Notes

- Daily ticks: ~1.2 GB for 5000 symbols × 15 years
- Fundamentals: ~7.5 GB for 5000 symbols × 15 years
- Use threading for I/O-bound operations (local reads/writes)
- Rate limiting: Alpaca API (200 symbols/batch), SEC EDGAR (10 req/sec limit)
