"""Profile group operator performance.

Usage:
    uv run python scripts/profile_group_operators.py
"""

import os
import time

import polars as pl
from dotenv import load_dotenv

load_dotenv()


def profile_step(name: str, func, *args, **kwargs):
    """Profile a single step and return result with timing."""
    t0 = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    print(f"  {name}: {elapsed:.3f}s")
    return result, elapsed


def main():
    from alphalab.api.client import AlphaLabClient
    from alphalab.api.operators._numba_kernels import group_neutralize_rows
    from alphalab.api.operators.cross_sectional import rank
    from alphalab.api.operators.group import (
        _align_and_extract,
        _rebuild_dataframe,
        group_neutralize,
    )

    print("=" * 60)
    print("Group Operator Profiling")
    print("=" * 60)

    # Load data
    print("\n1. Loading data...")
    client = AlphaLabClient(data_path=os.environ["LOCAL_STORAGE_PATH"])
    return_assets, t_load1 = profile_step("return_assets", client.get, "return_assets")
    sector, t_load2 = profile_step("sector", client.get, "sector")
    print(f"  Shapes: return_assets={return_assets.shape}, sector={sector.shape}")

    # Rank
    print("\n2. Cross-sectional rank...")
    ranked, t_rank = profile_step("rank()", rank, return_assets)

    # Warmup numba
    print("\n3. Warmup (JIT compilation)...")
    warmup_x = pl.DataFrame({"date": [1, 2], "A": [1.0, 2.0], "B": [3.0, 4.0]})
    warmup_g = pl.DataFrame({"date": [1, 2], "A": ["g1", "g1"], "B": ["g2", "g2"]})
    _ = group_neutralize(warmup_x, warmup_g)
    print("  Done")

    # Profile group_neutralize internals
    print("\n4. group_neutralize() breakdown...")

    values, groups, date_col, value_cols = None, None, None, None

    def step_align():
        nonlocal values, groups, date_col, value_cols
        values, groups, date_col, value_cols = _align_and_extract(ranked, sector)
        return values

    _, t_align = profile_step("_align_and_extract()", step_align)

    result_arr = None
    def step_kernel():
        nonlocal result_arr
        result_arr = group_neutralize_rows(values, groups)
        return result_arr

    _, t_kernel = profile_step("group_neutralize_rows()", step_kernel)

    _, t_rebuild = profile_step(
        "_rebuild_dataframe()",
        _rebuild_dataframe, result_arr, ranked, date_col, value_cols
    )

    # Full operation
    print("\n5. Full group_neutralize()...")
    _, t_full = profile_step("group_neutralize()", group_neutralize, ranked, sector)

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"\nData dimensions:")
    print(f"  Rows (dates): {ranked.height:,}")
    print(f"  Cols (symbols): {len(value_cols):,}")
    print(f"  Total cells: {ranked.height * len(value_cols):,}")

    print(f"\ngroup_neutralize() breakdown:")
    print(f"  _align_and_extract(): {t_align:.3f}s ({t_align/t_full*100:.1f}%)")
    print(f"  group_neutralize_rows(): {t_kernel:.3f}s ({t_kernel/t_full*100:.1f}%)")
    print(f"  _rebuild_dataframe(): {t_rebuild:.3f}s ({t_rebuild/t_full*100:.1f}%)")
    print(f"  Total: {t_full:.3f}s")

    print(f"\nFull pipeline (quality_factor.py equivalent):")
    total = t_load1 + t_load2 + t_rank + t_full
    print(f"  Load data: {t_load1 + t_load2:.2f}s")
    print(f"  rank(): {t_rank:.2f}s")
    print(f"  group_neutralize(): {t_full:.2f}s")
    print(f"  Total: {total:.2f}s")


if __name__ == "__main__":
    main()
