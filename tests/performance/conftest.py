"""Performance benchmark fixtures for AlphaLab operators.

Provides test data and performance thresholds for all 68 operators.
Data dimensions: 1000 rows x 1000 columns (1M cells).
"""

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest


# Performance thresholds in milliseconds for 1000x1000 data (1M cells)
# Categorized by computational complexity
PERFORMANCE_THRESHOLDS_MS = {
    # =========================================================================
    # ARITHMETIC OPERATORS (element-wise) - target <100ms
    # =========================================================================
    "abs": 100,
    "add": 100,
    "subtract": 100,
    "multiply": 100,
    "divide": 100,
    "inverse": 100,
    "log": 100,
    "max": 100,
    "min": 100,
    "power": 100,
    "signed_power": 200,
    "sqrt": 100,
    "sign": 100,
    "reverse": 100,
    "densify": 500,
    # =========================================================================
    # LOGICAL OPERATORS (element-wise) - target <100ms
    # =========================================================================
    "and_": 100,
    "or_": 100,
    "not_": 100,
    "if_else": 100,
    "is_nan": 100,
    "lt": 100,
    "le": 100,
    "gt": 100,
    "ge": 100,
    "eq": 100,
    "ne": 100,
    # =========================================================================
    # VECTOR OPERATORS (reduction across columns) - target <100ms
    # =========================================================================
    "vec_avg": 100,
    "vec_sum": 100,
    # =========================================================================
    # CROSS-SECTIONAL OPERATORS (row-wise operations) - target <500ms
    # =========================================================================
    "rank": 500,
    "zscore": 300,
    "normalize": 200,
    "scale": 500,
    "quantile": 500,
    "bucket": 300,
    "winsorize": 300,
    # =========================================================================
    # TIME-SERIES OPERATORS - Basic (rolling window) - target <300ms
    # =========================================================================
    "ts_mean": 300,
    "ts_sum": 300,
    "ts_std": 300,
    "ts_min": 300,
    "ts_max": 300,
    "ts_delta": 200,
    "ts_delay": 200,
    # =========================================================================
    # TIME-SERIES OPERATORS - Rolling (more complex aggregations) - target <500ms
    # =========================================================================
    "ts_product": 500,
    "ts_count_nans": 300,
    "ts_zscore": 500,
    "ts_scale": 500,
    "ts_av_diff": 300,
    "ts_step": 500,
    # =========================================================================
    # TIME-SERIES OPERATORS - Arg (finding indices) - target <500ms
    # =========================================================================
    "ts_arg_max": 500,
    "ts_arg_min": 500,
    # =========================================================================
    # TIME-SERIES OPERATORS - Lookback (conditional lookback) - target <500ms
    # =========================================================================
    "ts_backfill": 300,
    "kth_element": 300,
    "last_diff_value": 500,
    "days_from_last_change": 500,
    # =========================================================================
    # TIME-SERIES OPERATORS - Stateful (more complex state tracking) - target <500ms
    # =========================================================================
    "hump": 500,
    "ts_decay_linear": 500,
    "ts_rank": 500,
    # =========================================================================
    # TIME-SERIES OPERATORS - Two-variable (cross-column computations) - target <800ms
    # =========================================================================
    "ts_corr": 800,
    "ts_covariance": 800,
    "ts_quantile": 800,
    "ts_regression": 800,
    # =========================================================================
    # GROUP OPERATORS (grouped cross-sectional operations) - target <1000ms
    # =========================================================================
    "group_rank": 1000,
    "group_zscore": 1000,
    "group_scale": 1000,
    "group_neutralize": 1000,
    "group_mean": 1000,
    "group_backfill": 4000,  # Uses complex forward-fill logic
    # =========================================================================
    # TRANSFORMATIONAL OPERATORS (complex conditional logic) - target <300ms
    # =========================================================================
    "trade_when": 300,
}

# Verify we have thresholds for all 68 operators
assert len(PERFORMANCE_THRESHOLDS_MS) == 68, (
    f"Expected 68 operators, got {len(PERFORMANCE_THRESHOLDS_MS)}"
)


@pytest.fixture(scope="session")
def benchmark_df() -> pl.DataFrame:
    """Create benchmark DataFrame.

    Dimensions: 1000 rows x 1000 columns (1M cells)
    - Simulates ~4 years of daily trading data (252 days/year * 4 years ~ 1000)
    - 1000 columns represent individual stock symbols

    Data characteristics:
    - Random float values with realistic distribution (mean=0, std=1)
    - ~5% NaN values to simulate missing data
    - First column is timestamp (date)

    Returns:
        pl.DataFrame with timestamp column + 1000 numeric columns
    """
    n_rows = 1000
    n_cols = 1000

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
