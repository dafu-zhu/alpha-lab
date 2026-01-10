# US Equity Data Lake

A self-hosted, automated market data infrastructure for US equities using official/authoritative sources, with daily updates and programmatic access via Python API.

## Overview

This project provides comprehensive data collection, storage, and query capabilities for US equity markets:

- **Daily tick data (OHLCV)** from CRSP via WRDS (2009+)
- **Minute-level tick data** from Alpaca Market Data API (2016+)
- **Fundamental data** from SEC EDGAR JSON API (2009+)
- **Corporate actions** (dividends, splits)
- **Index constituents** for major US indices
- **Derived technical indicators** (MACD, RSI, etc.)

All data is stored in a flat-file structure on AWS S3, optimized for fast querying and minimal storage costs.

## Features

✅ **Official Data Sources** - All data from authoritative sources (SEC, CRSP, Alpaca)
✅ **Automated Updates** - Daily scheduled updates with error handling and retry logic
✅ **Flat File Storage** - Organized by symbol and time period for efficient querying
✅ **Python Query API** - Simple programmatic access with support for date ranges and multi-symbol queries
✅ **Security Master** - Track stocks across symbol changes, mergers, and corporate actions
✅ **Data Validation** - Comprehensive quality checks and completeness reports

## Installation

### Prerequisites

- Python 3.10+
- AWS account with S3 access
- WRDS account (for CRSP data)
- Alpaca account (for minute-level data)
- SEC EDGAR API access

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/us-equity-datalake.git
cd us-equity-datalake
```

2. Install dependencies using uv:
```bash
uv sync
```

3. Configure environment variables (create `.env` file):
```bash
# WRDS Credentials
WRDS_USERNAME=your_username
WRDS_PASSWORD=your_password

# Alpaca API Credentials
ALPACA_API_KEY=your_api_key
ALPACA_API_SECRET=your_secret_key

# AWS Credentials (for S3 storage)
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=us-east-1

# SEC EDGAR API (required User-Agent)
SEC_USER_AGENT=your_name@example.com
```

4. Initialize the data directories:
```bash
mkdir -p data/{ticks,fundamental,corporate_actions,reference,logs}
```

## Quick Start

### Collecting Daily Tick Data

```python
from quantdl.collection.crsp_ticks import CRSPDailyTicks

# Initialize collector
collector = CRSPDailyTicks()

# Fetch single day
data = collector.get_daily('AAPL', '2024-01-15')
print(data)  # {'timestamp': '2024-01-15', 'open': 182.15, 'close': 183.63, ...}

# Fetch date range
data = collector.get_daily_range('AAPL', '2024-01-01', '2024-01-31')

# Fetch multiple symbols (bulk query)
result = collector.recent_daily_ticks(
    symbols=['AAPL', 'MSFT', 'GOOGL'],
    end_day='2024-01-31',
    window=90
)

collector.close()
```

### Querying Fundamental Data

```python
from quantdl.collection.fundamental import FundamentalCollector

collector = FundamentalCollector()

# Fetch quarterly fundamentals for a symbol
fundamentals = collector.get_fundamentals('AAPL', year=2023, quarter=4)
print(fundamentals)  # Returns metrics like revenue, net_income, assets, etc.

collector.close()
```

### Using the Security Master

The Security Master tracks stocks across symbol changes (e.g., FB → META):

```python
from quantdl.master.security_master import SecurityMaster

sm = SecurityMaster()

# Get security_id for a symbol at a specific date
sid = sm.get_security_id('META', '2023-01-01')  # Returns security_id for Facebook/Meta

# Auto-resolve handles symbol changes
fb_2021 = sm.get_security_id('FB', '2021-01-01', auto_resolve=True)
meta_2023 = sm.get_security_id('META', '2023-01-01', auto_resolve=True)
# fb_2021 == meta_2023 (same company, different symbols)

# Get symbol history
history = sm.get_symbol_history(sid)
# [('META', '2022-06-09', '2024-12-31'), ('FB', '2012-05-18', '2022-06-08')]

sm.close()
```

## Project Structure

```
us-equity-datalake/
├── src/quantdl/              # Main package
│   ├── collection/           # Data collectors (CRSP, Alpaca, SEC)
│   ├── storage/              # Upload and validation logic
│   ├── master/               # Security master (symbol tracking)
│   ├── stock_pool/           # Universe and stock filtering
│   ├── derived/              # Technical indicators
│   └── utils/                # Logging, mapping, rate limiting
├── scripts/                  # One-off scripts and utilities
├── data/                     # Local data cache
│   ├── ticks/                # Daily and minute tick data
│   ├── fundamental/          # Quarterly fundamentals
│   ├── corporate_actions/    # Dividends and splits
│   └── reference/            # Metadata and indices
├── docs/                     # Additional documentation
└── tests/                    # Unit and integration tests
```

## Data Storage Format

### Daily Ticks
- **Path**: `data/ticks/daily/{symbol}/{YYYY}/ticks.json`
- **Format**: JSON with OHLCV fields
- **Coverage**: 2009+ (CRSP)

### Minute Ticks
- **Path**: `data/ticks/minute/{symbol}/{YYYY}/{MM}/{DD}/ticks.parquet`
- **Format**: Parquet for efficient storage
- **Coverage**: 2016+ (Alpaca)

### Fundamentals
- **Path**: `data/fundamental/{symbol}/{YYYY}/fundamental.json`
- **Format**: JSON with quarterly/annual metrics
- **Coverage**: 2009+ (SEC EDGAR)

### Reference Data
- **Path**: `data/reference/ticker_metadata.parquet`
- **Format**: Parquet with symbol metadata
- **Fields**: CIK, CUSIP, company name, exchange

## CLI Commands

The package includes a CLI for common operations:

```bash
# Collect daily ticks for a symbol
quantdl collect-ticks AAPL --start 2024-01-01 --end 2024-12-31

# Upload data to S3
quantdl upload --symbols AAPL,MSFT --year 2024

# Validate data quality
quantdl validate --year 2024

# Generate coverage report
quantdl report --output coverage.html
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_crsp_ticks.py

# Run with coverage
pytest --cov=quantdl --cov-report=html
```

### Code Quality

```bash
# Format code
black src/

# Type checking
mypy src/

# Linting
ruff check src/
```

## Data Quality & Known Limitations

### Coverage
- **Daily ticks**: ~99% coverage for large/mid-cap stocks (2009+)
- **Fundamentals**: ~75% coverage for small-cap stocks (SEC filing requirements)
- **Minute data**: 2016+ only (Alpaca API limitation)

### Known Gaps
- No historical index constituents (survivorship bias)
- Fundamental data lag: 45-90 days (filing deadlines)
- Delisted stocks: historical data preserved but no updates

### Data Quality Checks
- OHLC sanity checks (High ≥ Low, Close within range)
- Volume validation (non-negative)
- Missing data flagged and logged
- Balance sheet equation validation (Assets = Liabilities + Equity)

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **WRDS** for CRSP database access
- **SEC** for EDGAR API and official filings
- **Alpaca** for minute-level market data
- **Nasdaq** for reference data

## Support

For questions or issues:
- Open an issue on GitHub
- Check the [documentation](docs/)
- Review the [troubleshooting guide](docs/TROUBLESHOOTING.md)

## Roadmap

- [ ] Add options data support
- [ ] Implement real-time streaming
- [ ] Add more technical indicators
- [ ] Support for international equities
- [ ] Web-based data explorer UI

---

**Note**: This project requires valid credentials for WRDS, Alpaca, and AWS. Data usage is subject to the terms of service of each provider.
