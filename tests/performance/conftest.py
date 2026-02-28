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
    # ARITHMETIC OPERATORS (element-wise, very fast)
    # =========================================================================
    "abs": 50,
    "add": 50,
    "subtract": 50,
    "multiply": 50,
    "divide": 50,
    "inverse": 50,
    "log": 100,
    "max": 50,
    "min": 50,
    "power": 100,
    "signed_power": 100,
    "sqrt": 100,
    "sign": 50,
    "reverse": 50,
    "densify": 200,
    # =========================================================================
    # LOGICAL OPERATORS (element-wise, very fast)
    # =========================================================================
    "and_": 50,
    "or_": 50,
    "not_": 50,
    "if_else": 100,
    "is_nan": 50,
    "lt": 50,
    "le": 50,
    "gt": 50,
    "ge": 50,
    "eq": 50,
    "ne": 50,
    # =========================================================================
    # VECTOR OPERATORS (reduction across columns)
    # =========================================================================
    "vec_avg": 100,
    "vec_sum": 100,
    # =========================================================================
    # CROSS-SECTIONAL OPERATORS (row-wise operations)
    # =========================================================================
    "rank": 600,
    "zscore": 300,
    "normalize": 300,
    "scale": 300,
    "quantile": 900,
    "bucket": 700,
    "winsorize": 400,
    # =========================================================================
    # TIME-SERIES OPERATORS - Basic (rolling window)
    # =========================================================================
    "ts_mean": 500,
    "ts_sum": 500,
    "ts_std": 500,
    "ts_min": 500,
    "ts_max": 500,
    "ts_delta": 300,
    "ts_delay": 200,
    # =========================================================================
    # TIME-SERIES OPERATORS - Rolling (more complex aggregations)
    # =========================================================================
    "ts_product": 600,
    "ts_count_nans": 400,
    "ts_zscore": 600,
    "ts_scale": 600,
    "ts_av_diff": 600,
    "ts_step": 400,
    # =========================================================================
    # TIME-SERIES OPERATORS - Arg (finding indices)
    # =========================================================================
    "ts_arg_max": 600,
    "ts_arg_min": 600,
    # =========================================================================
    # TIME-SERIES OPERATORS - Lookback (conditional lookback)
    # =========================================================================
    "ts_backfill": 500,
    "kth_element": 400,
    "last_diff_value": 600,
    "days_from_last_change": 600,
    # =========================================================================
    # TIME-SERIES OPERATORS - Stateful (more complex state tracking)
    # =========================================================================
    "hump": 700,
    "ts_decay_linear": 800,
    "ts_rank": 800,
    # =========================================================================
    # TIME-SERIES OPERATORS - Two-variable (cross-column computations)
    # =========================================================================
    "ts_corr": 1000,
    "ts_covariance": 1000,
    "ts_quantile": 900,
    "ts_regression": 1500,
    # =========================================================================
    # GROUP OPERATORS (grouped cross-sectional operations)
    # =========================================================================
    "group_rank": 800,
    "group_zscore": 600,
    "group_scale": 600,
    "group_neutralize": 700,
    "group_mean": 500,
    "group_backfill": 600,
    # =========================================================================
    # TRANSFORMATIONAL OPERATORS (complex conditional logic)
    # =========================================================================
    "trade_when": 400,
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
