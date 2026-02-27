#!/usr/bin/env python
"""Benchmark all operators for time and memory efficiency."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import tracemalloc
from collections.abc import Callable
from dataclasses import dataclass, asdict
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import polars as pl

from alphalab.api.client import AlphaLabClient
from alphalab.api import operators as ops

# Threshold for marking an operator as "slow"
SLOW_THRESHOLD_S = 1.0


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


def benchmark_operator(
    name: str,
    category: str,
    fn: Callable[[], pl.DataFrame],
    input_df: pl.DataFrame | None = None,
) -> BenchmarkResult:
    """Benchmark a single operator call."""
    input_shape = (input_df.height, input_df.width) if input_df is not None else (0, 0)

    tracemalloc.start()
    start = time.perf_counter()

    try:
        result = fn()
        duration = time.perf_counter() - start
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        output_shape = (result.height, result.width)
        status = "slow" if duration > SLOW_THRESHOLD_S else "fast"

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
        print(f"\nSlow operators (>{SLOW_THRESHOLD_S}s): {', '.join(r.name for r in slow)}")


def run_benchmarks(
    benchmarks: list[tuple[str, Callable[[], pl.DataFrame]]],
    category: str,
    input_df: pl.DataFrame,
    results: list[BenchmarkResult],
) -> None:
    """Run a list of benchmarks and append results."""
    for name, fn in benchmarks:
        print(f"  {name}...", end=" ", flush=True)
        result = benchmark_operator(name, category, fn, input_df)
        results.append(result)
        print(f"{result.time_s:.3f}s")


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
    data_path = os.environ.get("LOCAL_STORAGE_PATH")
    if not data_path:
        print("Error: LOCAL_STORAGE_PATH environment variable not set")
        print("Set it in .env file or export LOCAL_STORAGE_PATH=/path/to/data")
        sys.exit(1)
    client = AlphaLabClient(data_path=data_path)

    # Load test data
    close = client.get("close")
    volume = client.get("volume")
    print(f"Data shape: {close.shape}")

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
        run_benchmarks(ts_1arg, "time-series", close, results)

        # Time-series operators (2-arg)
        ts_2arg = [
            ("ts_corr", lambda: ops.ts_corr(close, volume, 20)),
            ("ts_covariance", lambda: ops.ts_covariance(close, volume, 20)),
            ("ts_regression", lambda: ops.ts_regression(close, volume, 20)),
        ]
        run_benchmarks(ts_2arg, "time-series", close, results)

    print_results(results, args.category)
    save_results(results, Path(args.output))


if __name__ == "__main__":
    main()
