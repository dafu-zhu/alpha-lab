#!/usr/bin/env python
"""Quick verification of parallelized operators.

Benchmarks the 5 operators that were parallelized in Tasks 5-9:
- days_from_last_change (Task 5)
- ts_corr (Task 6)
- ts_covariance (Task 7)
- ts_regression (Task 8)
- hump (Task 9)

Compares against baseline times from benchmark_baseline.json.
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import polars as pl

from alphalab.api.client import AlphaLabClient
from alphalab.api import operators as ops


@dataclass
class OptimizationResult:
    """Result of benchmarking an optimized operator."""
    name: str
    baseline_time_s: float
    optimized_time_s: float
    speedup: float
    input_rows: int
    input_cols: int
    status: str  # "improved", "same", "slower"


def run_benchmark(
    name: str,
    fn,
    baseline_time: float,
    n_rows: int,
    n_cols: int,
) -> OptimizationResult:
    """Run benchmark for a single operator."""
    # Warm-up run
    _ = fn()

    # Timed run
    start = time.perf_counter()
    _ = fn()
    elapsed = time.perf_counter() - start

    speedup = baseline_time / elapsed if elapsed > 0 else float('inf')

    if speedup > 1.5:
        status = "improved"
    elif speedup > 0.8:
        status = "same"
    else:
        status = "slower"

    return OptimizationResult(
        name=name,
        baseline_time_s=baseline_time,
        optimized_time_s=elapsed,
        speedup=speedup,
        input_rows=n_rows,
        input_cols=n_cols,
        status=status,
    )


def main():
    # Check for --quick flag for smaller dataset
    quick_mode = "--quick" in sys.argv

    print("Loading data...")
    data_path = os.environ.get("LOCAL_STORAGE_PATH")
    if not data_path:
        print("Error: LOCAL_STORAGE_PATH environment variable not set")
        sys.exit(1)

    client = AlphaLabClient(data_path=data_path)

    # Load test data
    close = client.get("close")
    volume = client.get("volume")

    if quick_mode:
        # Use subset for quick testing
        close = close.head(500)
        volume = volume.head(500)
        print(f"Quick mode: using {close.shape} data")
    else:
        print(f"Full data shape: {close.shape}")

    # Baseline times from benchmark_baseline.json (full dataset: 2262 rows, 5705 cols)
    # For quick mode, we'll scale expected times proportionally
    full_baselines = {
        "days_from_last_change": 19.70,
        "ts_corr": 579.07,
        "ts_covariance": 343.05,
        "ts_regression": 1840.79,
        "hump": 31.97,
    }

    # Scale baselines for quick mode (approximately linear with rows)
    if quick_mode:
        scale_factor = 500 / 2262
        baselines = {k: v * scale_factor for k, v in full_baselines.items()}
    else:
        baselines = full_baselines

    # Define operator benchmarks
    operators = [
        ("days_from_last_change", lambda: ops.days_from_last_change(close)),
        ("ts_corr", lambda: ops.ts_corr(close, volume, 20)),
        ("ts_covariance", lambda: ops.ts_covariance(close, volume, 20)),
        ("ts_regression", lambda: ops.ts_regression(close, volume, 20)),
        ("hump", lambda: ops.hump(close, 0.01)),
    ]

    results: list[OptimizationResult] = []

    print(f"\nBenchmarking optimized operators ({close.height} rows, {close.width} cols):")
    print("-" * 70)

    for name, fn in operators:
        print(f"  {name}...", end=" ", flush=True)
        result = run_benchmark(
            name=name,
            fn=fn,
            baseline_time=baselines[name],
            n_rows=close.height,
            n_cols=close.width,
        )
        results.append(result)

        status_emoji = {
            "improved": "[OK]",
            "same": "[--]",
            "slower": "[!!]",
        }[result.status]

        print(f"{result.optimized_time_s:.3f}s (baseline: {result.baseline_time_s:.1f}s, speedup: {result.speedup:.1f}x) {status_emoji}")

    print("-" * 70)

    # Summary
    improved = [r for r in results if r.status == "improved"]
    same = [r for r in results if r.status == "same"]
    slower = [r for r in results if r.status == "slower"]

    print(f"\nSummary:")
    print(f"  Improved: {len(improved)}/{len(results)}")
    print(f"  Same: {len(same)}/{len(results)}")
    print(f"  Slower: {len(slower)}/{len(results)}")

    if improved:
        avg_speedup = sum(r.speedup for r in improved) / len(improved)
        print(f"  Average speedup (improved): {avg_speedup:.1f}x")

    # Save results
    output_path = Path("reports/benchmark_optimized.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "date": str(date.today()),
        "description": "Benchmark results after parallelization optimizations (Tasks 5-9)",
        "mode": "quick" if quick_mode else "full",
        "data_shape": [close.height, close.width],
        "operators": [asdict(r) for r in results],
        "summary": {
            "total": len(results),
            "improved": len(improved),
            "same": len(same),
            "slower": len(slower),
            "avg_speedup": sum(r.speedup for r in results) / len(results),
        },
        "baseline_source": "reports/benchmark_baseline.json",
        "optimizations_applied": [
            "days_from_last_change: ThreadPoolExecutor column-parallel (Task 5)",
            "ts_corr: ThreadPoolExecutor column-parallel (Task 6)",
            "ts_covariance: ThreadPoolExecutor column-parallel (Task 7)",
            "ts_regression: ThreadPoolExecutor column-parallel (Task 8)",
            "hump: ThreadPoolExecutor column-parallel with pre-computed row limits (Task 9)",
        ],
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    # Return exit code based on results
    if slower:
        print("\nWARNING: Some operators are slower than baseline!")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
