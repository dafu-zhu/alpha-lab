# Operator Benchmark & Optimization Design

**Date:** 2026-02-27
**Status:** Approved
**Purpose:** Profile all 68 operators for time/memory efficiency, optimize slow ones with column-parallel processing

## Overview

Two-phase approach:
1. **Benchmark Suite:** Measure all operators with real data (2000×5000), identify slow operators (>1s)
2. **Targeted Optimization:** Apply ThreadPoolExecutor column-parallelism to slow operators

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  scripts/benchmark_operators.py                             │
│  - Loads real data (2000×5000 from LOCAL_STORAGE_PATH)      │
│  - Runs each of 68 operators with profiling                 │
│  - Measures: time, peak memory, input/output shape          │
│  - Outputs: console table + JSON report                     │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  reports/benchmark_YYYY-MM-DD.json                          │
│  {                                                          │
│    "operators": [                                           │
│      {"name": "rank", "time_s": 1.5, "memory_mb": 120, ...} │
│    ],                                                       │
│    "slow_operators": ["ts_corr", "ts_regression", ...]      │
│  }                                                          │
└─────────────────────────────────────────────────────────────┘
```

## Benchmark Categories

| Category | Operators | Test Data |
|----------|-----------|-----------|
| Time-series (1-arg) | `ts_mean`, `ts_sum`, `ts_delta`, etc. | `close` DataFrame |
| Time-series (2-arg) | `ts_corr`, `ts_covariance`, `ts_regression` | `close`, `volume` |
| Cross-sectional | `rank`, `zscore`, `normalize`, etc. | `close` |
| Group | `group_rank`, `group_zscore`, etc. | `close` + `sector` mask |
| Arithmetic/Logical | `add`, `multiply`, `gt`, etc. | `close`, `open` |

## Benchmark Implementation

### Memory Measurement

```python
import tracemalloc

def benchmark_operator(fn, *args, **kwargs):
    tracemalloc.start()
    start = time.perf_counter()

    result = fn(*args, **kwargs)

    duration = time.perf_counter() - start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "time_s": duration,
        "peak_memory_mb": peak / 1024 / 1024,
        "output_shape": (result.height, result.width),
    }
```

### Console Output Format

```
Operator Benchmark Report (2000×5372)
=====================================

Category: Time-Series
┌──────────────────┬──────────┬────────────┬─────────┐
│ Operator         │ Time (s) │ Memory(MB) │ Status  │
├──────────────────┼──────────┼────────────┼─────────┤
│ ts_mean          │    0.012 │       45.2 │ ✓ Fast  │
│ ts_corr          │    8.234 │      312.5 │ ⚠ SLOW  │
│ ts_regression    │   12.456 │      456.7 │ ⚠ SLOW  │
└──────────────────┴──────────┴────────────┴─────────┘

Slow operators (>1s): ts_corr, ts_regression, hump, days_from_last_change
```

**Threshold:** Operators >1s flagged as "SLOW" for optimization.

## Optimization Strategy

### Column-Parallel Processing

For operators with Python loops over columns, apply ThreadPoolExecutor:

```python
from concurrent.futures import ThreadPoolExecutor

def ts_corr_parallel(x: pl.DataFrame, y: pl.DataFrame, d: int) -> pl.DataFrame:
    date_col = x.columns[0]
    value_cols = x.columns[1:]

    def process_column(col: str) -> tuple[str, list]:
        return (col, compute_corr(x[col], y[col], d))

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = dict(executor.map(process_column, value_cols))

    return pl.DataFrame({date_col: x[date_col], **results})
```

### Target Operators

| Operator | Current Pattern | Expected Speedup |
|----------|-----------------|------------------|
| `ts_corr` | Python loop over cols | ~4-8x |
| `ts_covariance` | Python loop over cols | ~4-8x |
| `ts_regression` | Python loop over cols | ~4-8x |
| `hump` | Python loop over rows+cols | ~2-4x |
| `days_from_last_change` | Python loop over cols | ~4-8x |

**Worker count:** `min(8, cpu_count())` — balance parallelism vs overhead for 5000 columns.

## Testing Strategy

### Correctness Tests
- Run each optimized operator against original implementation
- Assert outputs are numerically identical (within float tolerance)
- Use smaller test data (100×50) for fast unit tests

### Performance Regression Tests
- Store baseline benchmark in `reports/benchmark_baseline.json`
- CI can compare against baseline, fail if >20% regression

### CLI Usage

```bash
# Run full benchmark
uv run python scripts/benchmark_operators.py

# Run specific category
uv run python scripts/benchmark_operators.py --category time-series

# Compare against baseline
uv run python scripts/benchmark_operators.py --compare baseline
```

## Files

| File | Purpose |
|------|---------|
| `scripts/benchmark_operators.py` | Benchmark runner script |
| `reports/benchmark_YYYY-MM-DD.json` | Benchmark results |
| `reports/benchmark_baseline.json` | Baseline for regression |
| `src/alphalab/api/operators/time_series.py` | Optimized operators |

## Success Criteria

1. All 68 operators benchmarked with time + memory metrics
2. Slow operators (>1s) identified and optimized
3. Optimized operators pass correctness tests (identical output)
4. No performance regressions in fast operators
