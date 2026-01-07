# Derived Data Module

This module computes derived metrics from raw data.

## Submodules

### `fundamental.py`

Computes derived fundamental metrics from raw fundamental data.

**Key Classes:**
- `FormulaParser`: Loads formula definitions from `data/xbrl/fundamental.xlsx`
- `DerivedFundamental`: Computes derived metrics for a single symbol/year
- `batch_compute_derived`: Batch processing for multiple symbols

**Quick Start:**

```python
from src.derived.fundamental import DerivedFundamental

# Compute derived metrics
calc = DerivedFundamental(symbol='AAPL', year=2024)
derived_df = calc.run()
```

**Storage:**
- Input: `data/raw/fundamental/{symbol}/{YYYY}/fundamental.parquet`
- Output: `data/derived/fundamental/{symbol}/{YYYY}/fundamental.parquet`

## Documentation

See `docs/derived_fundamentals.md` for detailed documentation:
- Complete list of derived concepts with formulas
- Usage examples
- Data quality considerations
- Troubleshooting guide

## Testing

```bash
# Quick test
python test_derived_simple.py

# Full test suite
pytest tests/test_derived_fundamental.py -v
```

## Implementation Notes

### Design Principles

1. **Hard-Coded Formulas:** All formulas implemented directly in code (based on `data/xbrl/fundamental.xlsx`)
2. **Safe Operations:** Robust handling of None/null/division-by-zero
3. **Dependency Order:** Metrics computed in correct dependency order
4. **No Side Effects:** Pure computation, no data modification
5. **Flexible Storage:** Configurable input/output paths
6. **Centralized Logging:** Uses `utils.logger.setup_logger` for consistent logging

### Formula Categories

- **Profitability:** Margins, EBITDA
- **Cash Flow:** Free cash flow, FCF margin
- **Balance Sheet:** Total debt, net debt, working capital
- **Returns:** ROA, ROE, ROIC
- **Growth:** Revenue growth, asset growth
- **Accruals:** Earnings quality metrics

### Code Structure

```
src/derived/
├── __init__.py              # Module exports
├── fundamental.py           # Main implementation
└── README.md               # This file

data/derived/fundamental/    # Output storage
├── {symbol}/
│   └── {YYYY}/
│       └── fundamental.parquet
```

## Future Modules

Planned additions:

- `technical.py`: Technical indicators (MACD, RSI, Bollinger Bands)
- `valuation.py`: Valuation ratios (P/E, P/B, EV/EBITDA)
- `quality.py`: Quality scores (Piotroski, Altman Z-Score)
- `composite.py`: Composite metrics combining multiple data sources
