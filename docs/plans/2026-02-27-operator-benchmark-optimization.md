# Operator Benchmark & Optimization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Benchmark all 68 operators for time/memory efficiency, then optimize slow operators (>1s) with column-parallel processing.

**Architecture:** Create benchmark script that runs all operators with real data, measures time+memory via tracemalloc, outputs report. Then parallelize slow operators using ThreadPoolExecutor across columns.

**Tech Stack:** Python stdlib (tracemalloc, time, concurrent.futures), polars, existing profiler

---

## Task 1: Create Benchmark Script Skeleton

**Files:**
- Create: `scripts/benchmark_operators.py`

**Step 1: Create the benchmark script with argument parsing**

```python
#!/usr/bin/env python
"""Benchmark all operators for time and memory efficiency."""
from __future__ import annotations

import argparse
import json
import os
import time
import tracemalloc
from dataclasses import dataclass, asdict
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import polars as pl

from alphalab.api.client import AlphaLabClient


@dataclass
class BenchmarkResult:
    """Result of benchmarking a single operator."""
    name: str
    category: str
    time_s: float
    peak_memory_mb: float
    input_shape: tuple[int, int]
    output_shape: tuple[int, int]
    status: str  # "fast", "slow", "error"
    error: str | None = None


def benchmark_operator(name: str, category: str, fn, *args, **kwargs) -> BenchmarkResult:
    """Benchmark a single operator call."""
    # Get input shape from first DataFrame arg
    input_shape = (0, 0)
    for arg in args:
        if isinstance(arg, pl.DataFrame):
            input_shape = (arg.height, arg.width)
            break

    tracemalloc.start()
    start = time.perf_counter()

    try:
        result = fn(*args, **kwargs)
        duration = time.perf_counter() - start
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        output_shape = (result.height, result.width)
        status = "slow" if duration > 1.0 else "fast"

        return BenchmarkResult(
            name=name,
            category=category,
            time_s=duration,
            peak_memory_mb=peak / 1024 / 1024,
            input_shape=input_shape,
            output_shape=output_shape,
            status=status,
        )
    except Exception as e:
        tracemalloc.stop()
        return BenchmarkResult(
            name=name,
            category=category,
            time_s=0.0,
            peak_memory_mb=0.0,
            input_shape=input_shape,
            output_shape=(0, 0),
            status="error",
            error=str(e),
        )


def print_results(results: list[BenchmarkResult], category: str | None = None) -> None:
    """Print benchmark results as formatted table."""
    filtered = [r for r in results if category is None or r.category == category]
    if not filtered:
        return

    # Group by category
    categories = sorted(set(r.category for r in filtered))

    for cat in categories:
        cat_results = [r for r in filtered if r.category == cat]
        cat_results.sort(key=lambda r: -r.time_s)  # Slowest first

        print(f"\nCategory: {cat}")
        print("┌" + "─" * 22 + "┬" + "─" * 10 + "┬" + "─" * 12 + "┬" + "─" * 10 + "┐")
        print(f"│ {'Operator':<20} │ {'Time(s)':>8} │ {'Memory(MB)':>10} │ {'Status':<8} │")
        print("├" + "─" * 22 + "┼" + "─" * 10 + "┼" + "─" * 12 + "┼" + "─" * 10 + "┤")

        for r in cat_results:
            status_str = "⚠ SLOW" if r.status == "slow" else ("✗ ERR" if r.status == "error" else "✓ Fast")
            print(f"│ {r.name:<20} │ {r.time_s:>8.3f} │ {r.peak_memory_mb:>10.1f} │ {status_str:<8} │")

        print("└" + "─" * 22 + "┴" + "─" * 10 + "┴" + "─" * 12 + "┴" + "─" * 10 + "┘")

    # Summary
    slow = [r for r in filtered if r.status == "slow"]
    if slow:
        print(f"\nSlow operators (>1s): {', '.join(r.name for r in slow)}")


def save_results(results: list[BenchmarkResult], output_path: Path) -> None:
    """Save benchmark results to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "date": str(date.today()),
        "operators": [asdict(r) for r in results],
        "slow_operators": [r.name for r in results if r.status == "slow"],
        "summary": {
            "total": len(results),
            "fast": len([r for r in results if r.status == "fast"]),
            "slow": len([r for r in results if r.status == "slow"]),
            "error": len([r for r in results if r.status == "error"]),
        },
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark operators")
    parser.add_argument("--category", help="Run only specific category")
    parser.add_argument("--output", default=f"reports/benchmark_{date.today()}.json")
    args = parser.parse_args()

    print("Loading data...")
    client = AlphaLabClient(data_path=os.environ["LOCAL_STORAGE_PATH"])
    # Load test data - will be filled in Task 2
    print("Benchmark script ready. Run Task 2 to add operator benchmarks.")


if __name__ == "__main__":
    main()
```

**Step 2: Run to verify script loads**

Run: `uv run python scripts/benchmark_operators.py --help`
Expected: Shows help message with --category and --output options

**Step 3: Commit**

```bash
git add scripts/benchmark_operators.py
git commit -m "feat(benchmark): add benchmark script skeleton"
```

---

## Task 2: Add Time-Series Operator Benchmarks

**Files:**
- Modify: `scripts/benchmark_operators.py`

**Step 1: Add time-series benchmarks to main()**

Replace the `main()` function with:

```python
def main():
    parser = argparse.ArgumentParser(description="Benchmark operators")
    parser.add_argument("--category", help="Run only specific category")
    parser.add_argument("--output", default=f"reports/benchmark_{date.today()}.json")
    args = parser.parse_args()

    print("Loading data...")
    client = AlphaLabClient(data_path=os.environ["LOCAL_STORAGE_PATH"])

    # Load test data
    close = client.get("close")
    volume = client.get("volume")
    print(f"Data shape: {close.shape}")

    from alphalab.api import operators as ops

    results: list[BenchmarkResult] = []

    # Time-series operators (1-arg)
    if args.category is None or args.category == "time-series":
        print("\nBenchmarking time-series operators...")

        ts_1arg = [
            ("ts_mean", lambda: ops.ts_mean(close, 20)),
            ("ts_sum", lambda: ops.ts_sum(close, 20)),
            ("ts_std", lambda: ops.ts_std(close, 20)),
            ("ts_min", lambda: ops.ts_min(close, 20)),
            ("ts_max", lambda: ops.ts_max(close, 20)),
            ("ts_delta", lambda: ops.ts_delta(close, 5)),
            ("ts_delay", lambda: ops.ts_delay(close, 5)),
            ("ts_product", lambda: ops.ts_product(close, 5)),
            ("ts_count_nans", lambda: ops.ts_count_nans(close, 20)),
            ("ts_zscore", lambda: ops.ts_zscore(close, 20)),
            ("ts_scale", lambda: ops.ts_scale(close, 20)),
            ("ts_av_diff", lambda: ops.ts_av_diff(close, 20)),
            ("ts_step", lambda: ops.ts_step(close)),
            ("ts_arg_max", lambda: ops.ts_arg_max(close, 20)),
            ("ts_arg_min", lambda: ops.ts_arg_min(close, 20)),
            ("ts_backfill", lambda: ops.ts_backfill(close, 5)),
            ("kth_element", lambda: ops.kth_element(close, 20, 5)),
            ("last_diff_value", lambda: ops.last_diff_value(close, 20)),
            ("days_from_last_change", lambda: ops.days_from_last_change(close)),
            ("hump", lambda: ops.hump(close, 0.01)),
            ("ts_decay_linear", lambda: ops.ts_decay_linear(close, 10)),
            ("ts_rank", lambda: ops.ts_rank(close, 20)),
            ("ts_quantile", lambda: ops.ts_quantile(close, 20)),
        ]

        for name, fn in ts_1arg:
            print(f"  {name}...", end=" ", flush=True)
            result = benchmark_operator(name, "time-series", fn)
            results.append(result)
            print(f"{result.time_s:.3f}s")

        # Time-series operators (2-arg)
        ts_2arg = [
            ("ts_corr", lambda: ops.ts_corr(close, volume, 20)),
            ("ts_covariance", lambda: ops.ts_covariance(close, volume, 20)),
            ("ts_regression", lambda: ops.ts_regression(close, volume, 20)),
        ]

        for name, fn in ts_2arg:
            print(f"  {name}...", end=" ", flush=True)
            result = benchmark_operator(name, "time-series", fn)
            results.append(result)
            print(f"{result.time_s:.3f}s")

    print_results(results, args.category)
    save_results(results, Path(args.output))


if __name__ == "__main__":
    main()
```

**Step 2: Run benchmark for time-series category**

Run: `uv run python scripts/benchmark_operators.py --category time-series`
Expected: Outputs benchmark table with time-series operators, saves JSON report

**Step 3: Commit**

```bash
git add scripts/benchmark_operators.py
git commit -m "feat(benchmark): add time-series operator benchmarks"
```

---

## Task 3: Add Cross-Sectional and Other Operator Benchmarks

**Files:**
- Modify: `scripts/benchmark_operators.py`

**Step 1: Add remaining operator categories**

Add these benchmark sections to `main()` after the time-series section:

```python
    # Cross-sectional operators
    if args.category is None or args.category == "cross-sectional":
        print("\nBenchmarking cross-sectional operators...")

        cs_ops = [
            ("rank", lambda: ops.rank(close)),
            ("zscore", lambda: ops.zscore(close)),
            ("normalize", lambda: ops.normalize(close)),
            ("scale", lambda: ops.scale(close)),
            ("quantile", lambda: ops.quantile(close)),
            ("winsorize", lambda: ops.winsorize(close)),
            ("bucket", lambda: ops.bucket(ops.rank(close), "0,1,0.25")),
        ]

        for name, fn in cs_ops:
            print(f"  {name}...", end=" ", flush=True)
            result = benchmark_operator(name, "cross-sectional", fn)
            results.append(result)
            print(f"{result.time_s:.3f}s")

    # Arithmetic operators
    if args.category is None or args.category == "arithmetic":
        print("\nBenchmarking arithmetic operators...")

        open_df = client.get("open")

        arith_ops = [
            ("add", lambda: ops.add(close, open_df)),
            ("subtract", lambda: ops.subtract(close, open_df)),
            ("multiply", lambda: ops.multiply(close, open_df)),
            ("divide", lambda: ops.divide(close, open_df)),
            ("abs", lambda: ops.abs(close)),
            ("log", lambda: ops.log(close)),
            ("sqrt", lambda: ops.sqrt(close)),
            ("power", lambda: ops.power(close, 2)),
            ("sign", lambda: ops.sign(close)),
            ("inverse", lambda: ops.inverse(close)),
            ("signed_power", lambda: ops.signed_power(close, 0.5)),
            ("min", lambda: ops.min(close, open_df)),
            ("max", lambda: ops.max(close, open_df)),
            ("reverse", lambda: ops.reverse(close)),
            ("densify", lambda: ops.densify(close)),
        ]

        for name, fn in arith_ops:
            print(f"  {name}...", end=" ", flush=True)
            result = benchmark_operator(name, "arithmetic", fn)
            results.append(result)
            print(f"{result.time_s:.3f}s")

    # Logical operators
    if args.category is None or args.category == "logical":
        print("\nBenchmarking logical operators...")

        open_df = client.get("open")

        logical_ops = [
            ("gt", lambda: ops.gt(close, open_df)),
            ("lt", lambda: ops.lt(close, open_df)),
            ("ge", lambda: ops.ge(close, open_df)),
            ("le", lambda: ops.le(close, open_df)),
            ("eq", lambda: ops.eq(close, open_df)),
            ("ne", lambda: ops.ne(close, open_df)),
            ("and_", lambda: ops.and_(ops.gt(close, open_df), ops.lt(close, volume))),
            ("or_", lambda: ops.or_(ops.gt(close, open_df), ops.lt(close, volume))),
            ("not_", lambda: ops.not_(ops.gt(close, open_df))),
            ("is_nan", lambda: ops.is_nan(close)),
            ("if_else", lambda: ops.if_else(ops.gt(close, open_df), close, open_df)),
        ]

        for name, fn in logical_ops:
            print(f"  {name}...", end=" ", flush=True)
            result = benchmark_operator(name, "logical", fn)
            results.append(result)
            print(f"{result.time_s:.3f}s")

    # Vector operators
    if args.category is None or args.category == "vector":
        print("\nBenchmarking vector operators...")

        open_df = client.get("open")

        vec_ops = [
            ("vec_sum", lambda: ops.vec_sum(close, open_df)),
            ("vec_avg", lambda: ops.vec_avg(close, open_df)),
        ]

        for name, fn in vec_ops:
            print(f"  {name}...", end=" ", flush=True)
            result = benchmark_operator(name, "vector", fn)
            results.append(result)
            print(f"{result.time_s:.3f}s")
```

**Step 2: Run full benchmark**

Run: `uv run python scripts/benchmark_operators.py`
Expected: Benchmarks all operator categories, outputs full report

**Step 3: Commit**

```bash
git add scripts/benchmark_operators.py
git commit -m "feat(benchmark): add all operator category benchmarks"
```

---

## Task 4: Run Benchmark and Identify Slow Operators

**Files:**
- Create: `reports/` directory
- Create: `reports/benchmark_baseline.json`

**Step 1: Run full benchmark and save baseline**

Run: `uv run python scripts/benchmark_operators.py --output reports/benchmark_baseline.json`

**Step 2: Review output and identify slow operators**

Review the console output. Expected slow operators (>1s):
- `ts_corr` (Python loop)
- `ts_covariance` (Python loop)
- `ts_regression` (Python loop)
- `hump` (Python loop)
- `days_from_last_change` (Python loop)

**Step 3: Commit baseline**

```bash
mkdir -p reports
git add reports/benchmark_baseline.json
git commit -m "docs: add benchmark baseline results"
```

---

## Task 5: Optimize `days_from_last_change` with Column Parallelism

**Files:**
- Modify: `src/alphalab/api/operators/time_series.py`
- Test: `tests/unit/api/operators/test_time_series_parallel.py`

**Step 1: Write the correctness test**

```python
# tests/unit/api/operators/test_time_series_parallel.py
"""Tests for parallelized time-series operators."""

import polars as pl
import pytest


def test_days_from_last_change_correctness():
    """Parallelized version produces same results as original."""
    from alphalab.api.operators.time_series import days_from_last_change

    df = pl.DataFrame({
        "Date": [1, 2, 3, 4, 5, 6, 7],
        "A": [1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0],
        "B": [5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 7.0],
    })

    result = days_from_last_change(df)

    # A: changes at idx 2 and 5
    assert result["A"].to_list() == [0, 1, 0, 1, 2, 0, 1]
    # B: changes at idx 3 and 6
    assert result["B"].to_list() == [0, 1, 2, 0, 1, 2, 0]
```

**Step 2: Run test to verify current implementation works**

Run: `uv run pytest tests/unit/api/operators/test_time_series_parallel.py -v`
Expected: PASS (current implementation should work)

**Step 3: Refactor with column parallelism**

Replace `days_from_last_change` in `src/alphalab/api/operators/time_series.py`:

```python
def _compute_days_from_last_change(col_data: list) -> list[int]:
    """Compute days since last change for a single column."""
    days: list[int] = []
    last_change_idx = 0
    for i, val in enumerate(col_data):
        if i == 0:
            days.append(0)
        elif val != col_data[i - 1]:
            last_change_idx = i
            days.append(0)
        else:
            days.append(i - last_change_idx)
    return days


def days_from_last_change(x: pl.DataFrame) -> pl.DataFrame:
    """Days since value changed (column-parallel)."""
    from concurrent.futures import ThreadPoolExecutor

    date_col = x.columns[0]
    value_cols = _get_value_cols(x)

    def process_col(c: str) -> tuple[str, list[int]]:
        return (c, _compute_days_from_last_change(x[c].to_list()))

    # Parallelize across columns
    with ThreadPoolExecutor(max_workers=min(8, len(value_cols))) as executor:
        col_results = dict(executor.map(process_col, value_cols))

    return pl.DataFrame({date_col: x[date_col], **col_results})
```

**Step 4: Run test to verify parallelized version works**

Run: `uv run pytest tests/unit/api/operators/test_time_series_parallel.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/alphalab/api/operators/time_series.py tests/unit/api/operators/test_time_series_parallel.py
git commit -m "perf: parallelize days_from_last_change across columns"
```

---

## Task 6: Optimize `ts_corr` with Column Parallelism

**Files:**
- Modify: `src/alphalab/api/operators/time_series.py`
- Test: `tests/unit/api/operators/test_time_series_parallel.py`

**Step 1: Write the correctness test**

Add to test file:

```python
def test_ts_corr_correctness():
    """Parallelized ts_corr produces same results."""
    from alphalab.api.operators.time_series import ts_corr

    df_x = pl.DataFrame({
        "Date": list(range(10)),
        "A": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "B": [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
    })
    df_y = pl.DataFrame({
        "Date": list(range(10)),
        "A": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "B": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    })

    result = ts_corr(df_x, df_y, 5)

    # A correlates perfectly with itself (corr = 1.0)
    assert result["A"][4] == pytest.approx(1.0, rel=1e-6)
    # B is negatively correlated (corr = -1.0)
    assert result["B"][4] == pytest.approx(-1.0, rel=1e-6)
```

**Step 2: Run test**

Run: `uv run pytest tests/unit/api/operators/test_time_series_parallel.py::test_ts_corr_correctness -v`
Expected: PASS

**Step 3: Refactor with column parallelism**

Replace `ts_corr` in `src/alphalab/api/operators/time_series.py`:

```python
def _compute_rolling_corr(x_vals: list, y_vals: list, d: int) -> list[float | None]:
    """Compute rolling correlation for a single column pair."""
    corrs: list[float | None] = []
    for i in range(len(x_vals)):
        if i < d - 1:
            corrs.append(None)
        else:
            x_win = x_vals[i - d + 1 : i + 1]
            y_win = y_vals[i - d + 1 : i + 1]
            if any(v is None for v in x_win) or any(v is None for v in y_win):
                corrs.append(None)
            else:
                x_mean = sum(x_win) / d
                y_mean = sum(y_win) / d
                cov = sum((xv - x_mean) * (yv - y_mean) for xv, yv in zip(x_win, y_win, strict=True)) / d
                x_std = (sum((xv - x_mean) ** 2 for xv in x_win) / d) ** 0.5
                y_std = (sum((yv - y_mean) ** 2 for yv in y_win) / d) ** 0.5
                if x_std == 0 or y_std == 0:
                    corrs.append(None)
                else:
                    corrs.append(cov / (x_std * y_std))
    return corrs


def ts_corr(x: pl.DataFrame, y: pl.DataFrame, d: int) -> pl.DataFrame:
    """Rolling Pearson correlation (column-parallel)."""
    from concurrent.futures import ThreadPoolExecutor

    date_col = x.columns[0]
    value_cols = _get_value_cols(x)

    def process_col(c: str) -> tuple[str, list[float | None]]:
        return (c, _compute_rolling_corr(x[c].to_list(), y[c].to_list(), d))

    with ThreadPoolExecutor(max_workers=min(8, len(value_cols))) as executor:
        col_results = dict(executor.map(process_col, value_cols))

    return pl.DataFrame({date_col: x[date_col], **col_results})
```

**Step 4: Run test**

Run: `uv run pytest tests/unit/api/operators/test_time_series_parallel.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/alphalab/api/operators/time_series.py tests/unit/api/operators/test_time_series_parallel.py
git commit -m "perf: parallelize ts_corr across columns"
```

---

## Task 7: Optimize `ts_covariance` with Column Parallelism

**Files:**
- Modify: `src/alphalab/api/operators/time_series.py`
- Test: `tests/unit/api/operators/test_time_series_parallel.py`

**Step 1: Write the correctness test**

Add to test file:

```python
def test_ts_covariance_correctness():
    """Parallelized ts_covariance produces same results."""
    from alphalab.api.operators.time_series import ts_covariance

    df_x = pl.DataFrame({
        "Date": list(range(5)),
        "A": [1.0, 2.0, 3.0, 4.0, 5.0],
    })
    df_y = pl.DataFrame({
        "Date": list(range(5)),
        "A": [2.0, 4.0, 6.0, 8.0, 10.0],  # y = 2*x
    })

    result = ts_covariance(df_x, df_y, 3)

    # Covariance of x with 2*x = 2 * var(x)
    # For window [3,4,5] and [6,8,10]: cov should be 2 * var([3,4,5])
    assert result["A"][4] is not None
```

**Step 2: Run test**

Run: `uv run pytest tests/unit/api/operators/test_time_series_parallel.py::test_ts_covariance_correctness -v`
Expected: PASS

**Step 3: Refactor with column parallelism**

Replace `ts_covariance`:

```python
def _compute_rolling_cov(x_vals: list, y_vals: list, d: int) -> list[float | None]:
    """Compute rolling covariance for a single column pair."""
    covs: list[float | None] = []
    for i in range(len(x_vals)):
        if i < d - 1:
            covs.append(None)
        else:
            x_win = x_vals[i - d + 1 : i + 1]
            y_win = y_vals[i - d + 1 : i + 1]
            if any(v is None for v in x_win) or any(v is None for v in y_win):
                covs.append(None)
            else:
                x_mean = sum(x_win) / d
                y_mean = sum(y_win) / d
                cov = sum((xv - x_mean) * (yv - y_mean) for xv, yv in zip(x_win, y_win, strict=True)) / d
                covs.append(cov)
    return covs


def ts_covariance(x: pl.DataFrame, y: pl.DataFrame, d: int) -> pl.DataFrame:
    """Rolling covariance (column-parallel)."""
    from concurrent.futures import ThreadPoolExecutor

    date_col = x.columns[0]
    value_cols = _get_value_cols(x)

    def process_col(c: str) -> tuple[str, list[float | None]]:
        return (c, _compute_rolling_cov(x[c].to_list(), y[c].to_list(), d))

    with ThreadPoolExecutor(max_workers=min(8, len(value_cols))) as executor:
        col_results = dict(executor.map(process_col, value_cols))

    return pl.DataFrame({date_col: x[date_col], **col_results})
```

**Step 4: Run test**

Run: `uv run pytest tests/unit/api/operators/test_time_series_parallel.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/alphalab/api/operators/time_series.py tests/unit/api/operators/test_time_series_parallel.py
git commit -m "perf: parallelize ts_covariance across columns"
```

---

## Task 8: Optimize `ts_regression` with Column Parallelism

**Files:**
- Modify: `src/alphalab/api/operators/time_series.py`
- Test: `tests/unit/api/operators/test_time_series_parallel.py`

**Step 1: Write the correctness test**

Add to test file:

```python
def test_ts_regression_correctness():
    """Parallelized ts_regression produces same results."""
    from alphalab.api.operators.time_series import ts_regression

    df_y = pl.DataFrame({
        "Date": list(range(5)),
        "A": [2.0, 4.0, 6.0, 8.0, 10.0],
    })
    df_x = pl.DataFrame({
        "Date": list(range(5)),
        "A": [1.0, 2.0, 3.0, 4.0, 5.0],
    })

    # y = 2*x, so beta should be ~2.0
    result = ts_regression(df_y, df_x, 3, rettype="beta")
    assert result["A"][4] == pytest.approx(2.0, rel=1e-6)

    # alpha should be ~0.0
    result_alpha = ts_regression(df_y, df_x, 3, rettype="alpha")
    assert result_alpha["A"][4] == pytest.approx(0.0, abs=1e-6)
```

**Step 2: Run test**

Run: `uv run pytest tests/unit/api/operators/test_time_series_parallel.py::test_ts_regression_correctness -v`
Expected: PASS

**Step 3: Refactor with column parallelism**

Extract the regression logic into a helper function and parallelize:

```python
def _compute_regression_col(
    y_vals: list, x_vals: list, d: int, lag: int, rettype: int
) -> list[float | None]:
    """Compute rolling regression for a single column."""
    import math

    if lag > 0:
        x_vals = [None] * lag + x_vals[:-lag] if lag < len(x_vals) else [None] * len(x_vals)

    results: list[float | None] = []
    for i in range(len(y_vals)):
        start_idx = max(0, i - d + 1)
        y_win_raw = y_vals[start_idx : i + 1]
        x_win_raw = x_vals[start_idx : i + 1]

        pairs = [(yv, xv) for yv, xv in zip(y_win_raw, x_win_raw, strict=True) if yv is not None and xv is not None]

        if len(pairs) < 2:
            results.append(None)
            continue

        y_win = [p[0] for p in pairs]
        x_win = [p[1] for p in pairs]
        n = len(pairs)
        x_mean = sum(x_win) / n
        y_mean = sum(y_win) / n

        ss_xx = sum((xv - x_mean) ** 2 for xv in x_win)
        ss_yy = sum((yv - y_mean) ** 2 for yv in y_win)
        ss_xy = sum((xv - x_mean) * (yv - y_mean) for xv, yv in zip(x_win, y_win, strict=True))

        if ss_xx == 0:
            results.append(None)
            continue

        beta = ss_xy / ss_xx
        alpha = y_mean - beta * x_mean
        y_pred = [alpha + beta * xv for xv in x_win]
        residuals = [yv - yp for yv, yp in zip(y_win, y_pred, strict=True)]
        ss_res = sum(r**2 for r in residuals)

        if rettype == 0:  # residual
            if y_vals[i] is None or x_vals[i] is None:
                results.append(None)
            else:
                results.append(y_vals[i] - (alpha + beta * x_vals[i]))
        elif rettype == 1:  # beta
            results.append(beta)
        elif rettype == 2:  # alpha
            results.append(alpha)
        elif rettype == 3:  # predicted
            if x_vals[i] is None:
                results.append(None)
            else:
                results.append(alpha + beta * x_vals[i])
        elif rettype == 4:  # correlation
            if ss_xx == 0 or ss_yy == 0:
                results.append(None)
            else:
                results.append(ss_xy / math.sqrt(ss_xx * ss_yy))
        elif rettype == 5:  # r-squared
            if ss_yy == 0:
                results.append(None)
            else:
                results.append(1 - ss_res / ss_yy)
        elif rettype == 6:  # t-stat beta
            if n <= 2 or ss_res == 0:
                results.append(None)
            else:
                mse = ss_res / (n - 2)
                se_beta = math.sqrt(mse / ss_xx)
                results.append(beta / se_beta if se_beta != 0 else None)
        elif rettype == 7:  # t-stat alpha
            if n <= 2 or ss_res == 0:
                results.append(None)
            else:
                mse = ss_res / (n - 2)
                se_alpha = math.sqrt(mse * (1 / n + x_mean**2 / ss_xx))
                results.append(alpha / se_alpha if se_alpha != 0 else None)
        elif rettype == 8:  # stderr beta
            if n <= 2:
                results.append(None)
            else:
                mse = ss_res / (n - 2)
                results.append(math.sqrt(mse / ss_xx))
        elif rettype == 9:  # stderr alpha
            if n <= 2:
                results.append(None)
            else:
                mse = ss_res / (n - 2)
                results.append(math.sqrt(mse * (1 / n + x_mean**2 / ss_xx)))
        else:
            results.append(None)

    return results


def ts_regression(
    y: pl.DataFrame,
    x: pl.DataFrame,
    d: int,
    lag: int = 0,
    rettype: int | str = 0,
) -> pl.DataFrame:
    """Rolling OLS regression (column-parallel)."""
    from concurrent.futures import ThreadPoolExecutor

    rettype_map = {
        "resid": 0, "residual": 0, "beta": 1, "slope": 1,
        "alpha": 2, "intercept": 2, "predicted": 3, "pred": 3,
        "corr": 4, "correlation": 4, "r_squared": 5, "rsquared": 5, "r2": 5,
        "tstat_beta": 6, "tstat_alpha": 7, "stderr_beta": 8, "stderr_alpha": 9,
    }
    if isinstance(rettype, str):
        rettype = rettype_map.get(rettype.lower(), 0)

    date_col = y.columns[0]
    value_cols = _get_value_cols(y)

    def process_col(c: str) -> tuple[str, list[float | None]]:
        y_vals = y[c].to_list()
        x_vals = x[c].to_list()
        return (c, _compute_regression_col(y_vals, x_vals, d, lag, rettype))

    with ThreadPoolExecutor(max_workers=min(8, len(value_cols))) as executor:
        col_results = dict(executor.map(process_col, value_cols))

    return pl.DataFrame({date_col: y[date_col], **col_results})
```

**Step 4: Run tests**

Run: `uv run pytest tests/unit/api/operators/test_time_series_parallel.py -v`
Expected: PASS

**Step 5: Run full test suite**

Run: `uv run pytest tests/ -x -q`
Expected: All pass

**Step 6: Commit**

```bash
git add src/alphalab/api/operators/time_series.py tests/unit/api/operators/test_time_series_parallel.py
git commit -m "perf: parallelize ts_regression across columns"
```

---

## Task 9: Optimize `hump` with Column Parallelism

**Files:**
- Modify: `src/alphalab/api/operators/time_series.py`
- Test: `tests/unit/api/operators/test_time_series_parallel.py`

**Step 1: Write the correctness test**

Add to test file:

```python
def test_hump_correctness():
    """Parallelized hump produces same results."""
    from alphalab.api.operators.time_series import hump

    df = pl.DataFrame({
        "Date": [1, 2, 3],
        "A": [1.0, 10.0, 2.0],  # Large jump at row 2
        "B": [5.0, 5.5, 5.2],   # Small changes
    })

    result = hump(df, hump=0.1)

    # Changes should be limited by hump factor
    assert result["A"].to_list()[0] == 1.0  # First row unchanged
    # Large jump limited
    assert result["B"].to_list()[0] == 5.0
```

**Step 2: Run test**

Run: `uv run pytest tests/unit/api/operators/test_time_series_parallel.py::test_hump_correctness -v`
Expected: PASS

**Step 3: Refactor with column parallelism**

Replace `hump`:

```python
def _compute_hump_col(col_data: list, all_col_data: dict[str, list], hump_factor: float, col_name: str) -> list[float | None]:
    """Compute hump-limited values for a single column."""
    n = len(col_data)
    out: list[float | None] = []

    for i in range(n):
        if i == 0:
            out.append(col_data[0])
        else:
            # Compute limit from all columns at row i
            row_sum = sum(abs(all_col_data[c][i] or 0) for c in all_col_data)
            limit = hump_factor * row_sum

            prev = out[i - 1]
            curr = col_data[i]
            if prev is None or curr is None:
                out.append(curr)
            else:
                change = curr - prev
                if abs(change) > limit:
                    out.append(prev + (1 if change > 0 else -1) * limit)
                else:
                    out.append(prev)

    return out


def hump(x: pl.DataFrame, hump: float = 0.01) -> pl.DataFrame:
    """Limit change magnitude (column-parallel where possible)."""
    from concurrent.futures import ThreadPoolExecutor

    date_col = x.columns[0]
    value_cols = _get_value_cols(x)

    # Pre-extract all column data (needed for row_sum calculation)
    all_col_data = {c: x[c].to_list() for c in value_cols}

    def process_col(c: str) -> tuple[str, list[float | None]]:
        return (c, _compute_hump_col(all_col_data[c], all_col_data, hump, c))

    with ThreadPoolExecutor(max_workers=min(8, len(value_cols))) as executor:
        col_results = dict(executor.map(process_col, value_cols))

    return pl.DataFrame({date_col: x[date_col], **col_results})
```

**Step 4: Run test**

Run: `uv run pytest tests/unit/api/operators/test_time_series_parallel.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/alphalab/api/operators/time_series.py tests/unit/api/operators/test_time_series_parallel.py
git commit -m "perf: parallelize hump across columns"
```

---

## Task 10: Re-run Benchmark and Verify Improvements

**Files:**
- Update: `reports/benchmark_baseline.json`

**Step 1: Run benchmark again**

Run: `uv run python scripts/benchmark_operators.py --output reports/benchmark_optimized.json`

**Step 2: Compare results**

Review the output. Expected improvements:
- `ts_corr`: ~4-8x faster
- `ts_covariance`: ~4-8x faster
- `ts_regression`: ~4-8x faster
- `hump`: ~2-4x faster
- `days_from_last_change`: ~4-8x faster

**Step 3: Run full test suite**

Run: `uv run pytest tests/ -q`
Expected: All tests pass

**Step 4: Commit optimized benchmark**

```bash
git add reports/benchmark_optimized.json
git commit -m "docs: add optimized benchmark results"
```

---

## Summary

| Task | Description | Commits |
|------|-------------|---------|
| 1 | Benchmark script skeleton | 1 |
| 2 | Time-series benchmarks | 1 |
| 3 | All category benchmarks | 1 |
| 4 | Run baseline benchmark | 1 |
| 5 | Parallelize `days_from_last_change` | 1 |
| 6 | Parallelize `ts_corr` | 1 |
| 7 | Parallelize `ts_covariance` | 1 |
| 8 | Parallelize `ts_regression` | 1 |
| 9 | Parallelize `hump` | 1 |
| 10 | Re-run benchmark, verify | 1 |

**Total: 10 tasks, ~10 commits**
