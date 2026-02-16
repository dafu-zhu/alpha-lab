# CLI Reference

```
uv run qdl [OPTIONS]
```

## Commands

| Option | Description |
|--------|-------------|
| `--master` | Build security master + trading calendar |
| `--all` | Master build + download all data types |
| `--ticks` | Download daily OHLCV prices from Alpaca |
| `--fundamental` | Download SEC EDGAR financial statements |
| `--top-3000` | Download monthly top 3000 universe snapshots |
| `--features` | Build feature wide tables from raw data |

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--start` | 2017 | Start year |
| `--end` | 2025 | End year |
| `--overwrite` | false | Re-download even if data exists |
| `--daily-chunk-size` | 200 | Symbols per Alpaca API batch |
| `--daily-sleep-time` | 0.2 | Seconds between API batches |
| `--max-workers` | 50 | Parallel workers for fundamentals |

## Examples

```bash
# Full setup from scratch
uv run qdl --master
uv run qdl --all --start 2017 --end 2025

# Update only prices for current year
uv run qdl --ticks --start 2025 --end 2025

# Rebuild feature tables after new data
uv run qdl --features

# Force re-download fundamentals
uv run qdl --fundamental --overwrite

# Slower API calls (if rate limited)
uv run qdl --ticks --daily-chunk-size 50 --daily-sleep-time 1.0
```

## Environment Variables

Set in `.env` (see [`.env.example`](../.env.example)):

| Variable | Required | Description |
|----------|----------|-------------|
| `LOCAL_STORAGE_PATH` | Yes | Root directory for all downloaded data |
| `ALPACA_API_KEY` | Yes | Alpaca API key (free account) |
| `ALPACA_API_SECRET` | Yes | Alpaca API secret |
| `SEC_USER_AGENT` | Yes | Email for SEC EDGAR (e.g., `you@example.com`) |
| `OpenFIGI_API_KEY` | No | OpenFIGI key for symbol rebrand detection |
