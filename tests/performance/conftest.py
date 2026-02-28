"""Performance benchmark fixtures for AlphaLab operators.

Provides production-scale test data and performance thresholds for all 68 operators.
Data dimensions: 4000 rows x 5000 columns (simulating ~16 years of daily data for 5000 stocks).
"""

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest


# Performance thresholds in milliseconds for 4000x5000 data
# Categorized by computational complexity
PERFORMANCE_THRESHOLDS_MS = {
    # =========================================================================
    # ARITHMETIC OPERATORS (element-wise)
    # Note: thresholds set with large headroom for system load variability
    # =========================================================================
    "abs": 3000,
    "add": 3000,
    "subtract": 3000,
    "multiply": 3000,
    "divide": 5000,
    "inverse": 3000,
    "log": 5000,
    "max": 3000,
    "min": 3000,
    "power": 3000,
    "signed_power": 10000,
    "sqrt": 5000,
    "sign": 3000,
    "reverse": 3000,
    "densify": 25000,
    # =========================================================================
    # LOGICAL OPERATORS (element-wise)
    # =========================================================================
    "and_": 3000,
    "or_": 3000,
    "not_": 3000,
    "if_else": 3000,
    "is_nan": 3000,
    "lt": 3000,
    "le": 3000,
    "gt": 3000,
    "ge": 3000,
    "eq": 3000,
    "ne": 3000,
    # =========================================================================
    # VECTOR OPERATORS (reduction across columns)
    # =========================================================================
    "vec_avg": 3000,
    "vec_sum": 3000,
    # =========================================================================
    # CROSS-SECTIONAL OPERATORS (row-wise operations)
    # =========================================================================
    "rank": 25000,
    "zscore": 10000,
    "normalize": 5000,
    "scale": 15000,
    "quantile": 25000,
    "bucket": 10000,
    "winsorize": 10000,
    # =========================================================================
    # TIME-SERIES OPERATORS - Basic (rolling window)
    # =========================================================================
    "ts_mean": 5000,
    "ts_sum": 5000,
    "ts_std": 5000,
    "ts_min": 5000,
    "ts_max": 5000,
    "ts_delta": 3000,
    "ts_delay": 3000,
    # =========================================================================
    # TIME-SERIES OPERATORS - Rolling (more complex aggregations)
    # =========================================================================
    "ts_product": 10000,
    "ts_count_nans": 5000,
    "ts_zscore": 5000,
    "ts_scale": 10000,
    "ts_av_diff": 5000,
    "ts_step": 15000,
    # =========================================================================
    # TIME-SERIES OPERATORS - Arg (finding indices)
    # =========================================================================
    "ts_arg_max": 15000,
    "ts_arg_min": 15000,
    # =========================================================================
    # TIME-SERIES OPERATORS - Lookback (conditional lookback)
    # =========================================================================
    "ts_backfill": 5000,
    "kth_element": 5000,
    "last_diff_value": 10000,
    "days_from_last_change": 15000,
    # =========================================================================
    # TIME-SERIES OPERATORS - Stateful (more complex state tracking)
    # =========================================================================
    "hump": 15000,
    "ts_decay_linear": 10000,
    "ts_rank": 25000,
    # =========================================================================
    # TIME-SERIES OPERATORS - Two-variable (cross-column computations)
    # =========================================================================
    "ts_corr": 10000,
    "ts_covariance": 10000,
    "ts_quantile": 30000,
    "ts_regression": 15000,
    # =========================================================================
    # GROUP OPERATORS (grouped cross-sectional operations)
    # Note: group ops iterate over many groups, taking significantly longer
    # =========================================================================
    "group_rank": 60000,
    "group_zscore": 60000,
    "group_scale": 60000,
    "group_neutralize": 60000,
    "group_mean": 60000,
    "group_backfill": 60000,
    # =========================================================================
    # TRANSFORMATIONAL OPERATORS (complex conditional logic)
    # =========================================================================
    "trade_when": 5000,
}

# Verify we have thresholds for all 68 operators
assert len(PERFORMANCE_THRESHOLDS_MS) == 68, (
    f"Expected 68 operators, got {len(PERFORMANCE_THRESHOLDS_MS)}"
)


@pytest.fixture(scope="session")
def benchmark_df() -> pl.DataFrame:
    """Create production-scale benchmark DataFrame.

    Dimensions: 4000 rows x 5000 columns
    - Simulates ~16 years of daily trading data (252 days/year * 16 years ~ 4000)
    - 5000 columns represent individual stock symbols

    Data characteristics:
    - Random float values with realistic distribution (mean=0, std=1)
    - ~5% NaN values to simulate missing data
    - First column is timestamp (date)

    Returns:
        pl.DataFrame with timestamp column + 5000 numeric columns
    """
    n_rows = 4000
    n_cols = 5000

    # Create random seed for reproducibility
    rng = np.random.default_rng(42)

    # Generate random data with ~5% NaN values
    data = rng.standard_normal((n_rows, n_cols))
    nan_mask = rng.random((n_rows, n_cols)) < 0.05
    data[nan_mask] = np.nan

    # Create date range (trading days approximation)
    start_date = date(2008, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_rows)]

    # Create symbol column names
    symbols = [f"S{i:04d}" for i in range(n_cols)]

    # Build DataFrame with timestamp as first column
    df_dict = {"timestamp": dates}
    for i, symbol in enumerate(symbols):
        df_dict[symbol] = data[:, i]

    return pl.DataFrame(df_dict)


@pytest.fixture(scope="session")
def benchmark_df_pair(benchmark_df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Create a pair of benchmark DataFrames for two-variable operators.

    Returns:
        Tuple of (df1, df2) with same dimensions but different random data
    """
    n_rows = benchmark_df.height
    n_cols = benchmark_df.width - 1  # Exclude timestamp

    rng = np.random.default_rng(123)  # Different seed

    data = rng.standard_normal((n_rows, n_cols))
    nan_mask = rng.random((n_rows, n_cols)) < 0.05
    data[nan_mask] = np.nan

    # Use same timestamps as benchmark_df
    timestamps = benchmark_df.get_column("timestamp")
    symbols = benchmark_df.columns[1:]

    df_dict = {"timestamp": timestamps}
    for i, symbol in enumerate(symbols):
        df_dict[symbol] = data[:, i]

    return benchmark_df, pl.DataFrame(df_dict)


@pytest.fixture(scope="session")
def benchmark_group_mask(benchmark_df: pl.DataFrame) -> pl.DataFrame:
    """Create group mask for group operators.

    Assigns each stock to one of 11 GICS sectors (integer encoded).
    Same dimensions as benchmark_df but with integer values 0-10.

    Returns:
        pl.DataFrame with timestamp column + integer group assignments
    """
    n_rows = benchmark_df.height
    n_cols = benchmark_df.width - 1

    rng = np.random.default_rng(456)

    # 11 GICS sectors
    n_groups = 11
    group_assignments = rng.integers(0, n_groups, size=(n_rows, n_cols))

    timestamps = benchmark_df.get_column("timestamp")
    symbols = benchmark_df.columns[1:]

    df_dict = {"timestamp": timestamps}
    for i, symbol in enumerate(symbols):
        df_dict[symbol] = group_assignments[:, i]

    return pl.DataFrame(df_dict)


@pytest.fixture(scope="session")
def benchmark_bool_df(benchmark_df: pl.DataFrame) -> pl.DataFrame:
    """Create boolean DataFrame for logical operators.

    Returns:
        pl.DataFrame with timestamp column + boolean values (~50% True)
    """
    n_rows = benchmark_df.height
    n_cols = benchmark_df.width - 1

    rng = np.random.default_rng(789)

    bool_data = rng.random((n_rows, n_cols)) > 0.5

    timestamps = benchmark_df.get_column("timestamp")
    symbols = benchmark_df.columns[1:]

    df_dict = {"timestamp": timestamps}
    for i, symbol in enumerate(symbols):
        df_dict[symbol] = bool_data[:, i]

    return pl.DataFrame(df_dict)


# Default window size for time-series operators
DEFAULT_WINDOW = 20


@pytest.fixture
def window() -> int:
    """Default rolling window size for time-series operators."""
    return DEFAULT_WINDOW
