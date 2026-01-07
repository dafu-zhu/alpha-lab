## Codebase Structure

This document describes the layout of the repository and the role of each major package.

### Top-Level Layout

- `src/`: Application code organized by domain (collection, storage, master data, utilities).
- `configs/`: YAML configuration and mappings (S3 settings, SEC field mappings).
- `data/`: Local runtime artifacts (calendars, logs, cached symbol lists).
- `docs/`: Project documentation (API, development, troubleshooting).
- `scripts/`: One-off utilities and helpers.
- `examples/`: Example usage and experiments (if present).
- `main.py`: Entry-point script (if used for orchestration).
- `pyproject.toml`, `uv.lock`: Dependency and environment configuration.

### Core Packages (`src/`)

#### `src/collection/`
Market data collectors that talk to external sources and normalize raw data.

- `alpaca_ticks.py`: Alpaca OHLCV collection for daily/minute bars; parsing and calendar alignment helpers.
- `crsp_ticks.py`: WRDS/CRSP daily OHLCV collection with symbol resolution.
- `fundamental.py`: SEC EDGAR XBRL extraction + transformation of fundamental data.
- `models.py`: Shared data models and enums (tick fields, data point structures).

#### `src/storage/`
S3 upload orchestration, validation, and publishing utilities.

- `upload_app.py`: High-level upload workflow (ticks, minute data, fundamentals).
- `data_collectors.py`: Coordinates data collection for upload flows.
- `data_publishers.py`: Handles S3 upload, metadata, and storage layout.
- `validation.py`: Checks for existing objects and lists S3 contents.
- `config_loader.py`: Loads `configs/storage.yaml` (S3 and transfer config).
- `s3_client.py`: Configured boto3 client with retry and connection tuning.
- `rate_limiter.py`: Thread-safe token bucket limiter (SEC rate control).
- `cik_resolver.py`: Resolves SEC CIKs using SecurityMaster and caching.

#### `src/master/`
Master data and reference mapping.

- `security_master.py`: CRSP-backed symbol/CIK/CUSIP history; normalization helpers.
- `trade_calendar.py`: Builds and stores trading calendars via Alpaca.

#### `src/stock_pool/`
Universe construction and symbol lists.

- `universe.py`: Current stock universe from Nasdaq Trader data.
- `history_universe.py`: Historical universe from CRSP.
- `universe_manager.py`: High-level API for symbol lists and liquidity ranking.

#### `src/update/`
Reserved for update/backfill workflows (check for future pipeline logic).

#### `src/query/`
Reserved for query API and data access layers (check for future query logic).

#### `src/utils/`
Shared helpers for logging, calendars, and mapping.

- `logger.py`: Logger factory and configuration.
- `calendar.py`: Trading day queries from stored calendars.
- `mapping.py`: Calendar alignment + SEC ticker/CIK mapping.

### Configuration (`configs/`)

- `storage.yaml`: S3 client and transfer configuration.
- `approved_mapping.yaml`: SEC concept-to-tag mappings used by fundamentals.
- `mapping_suggestions.yaml`: Candidate tags for mapping exploration.

### Scripts (`scripts/`)

- `generate_monthly_top3000.py`: Universe selection helper.
- `xbrl_tag_factory.py`: Tag/mapping tooling for SEC data.

### Data and Artifacts (`data/`)

Local runtime data (not checked in by default):

- `data/calendar/`: Trading day calendars (Parquet).
- `data/logs/`: Log files organized by module.
- `data/symbols/`: Cached Nasdaq/CRSP symbol lists.

### Typical Flow

1. Collect data in `src/collection/` (Alpaca/CRSP/SEC).
2. Orchestrate uploads via `src/storage/upload_app.py`.
3. Persist to S3 with `data_publishers.py` and validate with `validation.py`.
4. Use `src/utils/` for calendar alignment and logging.

### Dependency Diagram (Data Flow)

```text
External APIs/DBs
  |-- Alpaca (ticks, calendar) ------\
  |-- WRDS/CRSP (daily ticks) ----\    \
  |-- SEC EDGAR (fundamentals) ---+-----+--> src/collection/*
                                   \         (raw data -> normalized)
                                    \
                                     +--> src/master/*
                                          (security master, symbol history)
                                                |
                                                v
src/utils/* (calendar, mapping, logger) <--- src/collection/*
        |                                          |
        v                                          v
src/storage/data_collectors.py --------------> src/storage/data_publishers.py
        |                                          |
        v                                          v
src/storage/upload_app.py ----------------------> S3 (data/raw/*)
        |
        v
src/storage/validation.py (existence checks, listings)
```
