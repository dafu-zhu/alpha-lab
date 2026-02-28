"""Profile newly optimized operators.

Usage:
    uv run python scripts/profile_new_operators.py
"""

import os
import time

import polars as pl
from dotenv import load_dotenv

load_dotenv()


def profile(name, func, *args, **kwargs):
    """Profile a single step and return result with timing."""
    t0 = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    print(f"  {name}: {elapsed:.3f}s")
    return result, elapsed


def main():
    from alphalab.api.client import AlphaLabClient
    from alphalab.api.operators.arithmetic import densify
    from alphalab.api.operators.cross_sectional import bucket, quantile, rank
    from alphalab.api.operators.time_series import ts_regression

    print("=" * 60)
    print("New Operator Profiling")
    print("=" * 60)

    # Load data
    print("\n1. Loading data...")
    client = AlphaLabClient(data_path=os.environ["LOCAL_STORAGE_PATH"])
    close, _ = profile("close", client.get, "close")
    returns, _ = profile("returns", client.get, "returns")
    print(f"   Shape: {close.shape}")

    # ts_regression with different rettype values
    print("\n2. ts_regression profiling...")
    print("   (d=20, comparing rettype values)")

    # Warmup JIT
    small_df = close.head(100).select(close.columns[:10])
    _ = ts_regression(small_df, small_df, 5, rettype=0)
    _ = ts_regression(small_df, small_df, 5, lag=1, rettype=6)

    results = {}
    for rt in [0, 1, 4, 5, 6, 7, 8, 9]:
        name = {
            0: "residual",
            1: "beta",
            4: "corr",
            5: "r2",
            6: "tstat_beta",
            7: "tstat_alpha",
            8: "stderr_beta",
            9: "stderr_alpha",
        }[rt]
        _, t = profile(
            f"rettype={rt} ({name})", ts_regression, close, returns, 20, rettype=rt
        )
        results[name] = t

    print("\n   With lag=5:")
    for rt in [1, 6]:
        name = {1: "beta", 6: "tstat_beta"}[rt]
        _, t = profile(
            f"lag=5, rettype={rt} ({name})",
            ts_regression,
            close,
            returns,
            20,
            lag=5,
            rettype=rt,
        )
        results[f"lag5_{name}"] = t

    # quantile
    print("\n3. quantile profiling...")
    ranked, _ = profile("rank (for reference)", rank, close)
    _, t_q_gauss = profile("quantile(gaussian)", quantile, close, driver="gaussian")
    _, t_q_unif = profile("quantile(uniform)", quantile, close, driver="uniform")
    _, t_q_cauchy = profile("quantile(cauchy)", quantile, close, driver="cauchy")

    # densify
    print("\n4. densify profiling...")
    _, t_densify = profile("densify", densify, close)

    # bucket
    print("\n5. bucket profiling...")
    _, t_bucket = profile("bucket(0,1,0.1)", bucket, ranked, "0,1,0.1")
    _, t_bucket2 = profile("bucket(0,1,0.25)", bucket, ranked, "0,1,0.25")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"\nData: {close.height:,} rows × {len(close.columns)-1:,} symbols")
    print(f"\nts_regression (d=20):")
    print(f"  rettype 0-5 (basic): {results['residual']:.3f}s - {results['r2']:.3f}s")
    print(
        f"  rettype 6-9 (stats): {results['tstat_beta']:.3f}s - {results['stderr_alpha']:.3f}s"
    )
    print(f"  with lag=5: {results['lag5_beta']:.3f}s")
    print(f"\nquantile:")
    print(f"  gaussian: {t_q_gauss:.3f}s")
    print(f"  uniform:  {t_q_unif:.3f}s")
    print(f"  cauchy:   {t_q_cauchy:.3f}s")
    print(f"\ndensify: {t_densify:.3f}s")
    print(f"\nbucket: {t_bucket:.3f}s")


if __name__ == "__main__":
    main()
