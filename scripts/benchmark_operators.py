#!/usr/bin/env python
"""Benchmark all operators for time and memory efficiency."""
from __future__ import annotations

import argparse
import json
import os
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


def benchmark_operator(name: str, category: str, fn: Callable[..., pl.DataFrame], *args, **kwargs) -> BenchmarkResult:
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

    data_path = os.environ.get("LOCAL_STORAGE_PATH")
    if not data_path:
        print("Error: LOCAL_STORAGE_PATH environment variable not set")
        print("Set it in .env or export LOCAL_STORAGE_PATH=/path/to/data")
        return

    print("Loading data...")
    client = AlphaLabClient(data_path=data_path)
    # Load test data - will be filled in Task 2
    print("Benchmark script ready. Run Task 2 to add operator benchmarks.")


if __name__ == "__main__":
    main()
