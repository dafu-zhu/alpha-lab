"""Performance benchmarks for all 68 AlphaLab operators.

Runs each operator against performance thresholds defined in conftest.py.
Uses warmup runs for JIT compilation and reports median of 3 iterations.
"""

import statistics
import time

import polars as pl
import pytest

from alphalab.api import operators as ops
from tests.performance.conftest import PERFORMANCE_THRESHOLDS_MS


def benchmark_operator(op_func, *args, **kwargs) -> float:
    """Run operator with warmup and return median time in milliseconds.

    Args:
        op_func: Operator function to benchmark
        *args: Positional arguments for the operator
        **kwargs: Keyword arguments for the operator

    Returns:
        Median execution time in milliseconds across 3 iterations
    """
    # Warmup run (for JIT compilation)
    _ = op_func(*args, **kwargs)

    # Timed runs
    times = []
    for _ in range(3):
        start = time.perf_counter()
        _ = op_func(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        times.append(elapsed_ms)

    return statistics.median(times)


# =============================================================================
# ARITHMETIC OPERATORS (single DataFrame, element-wise)
# =============================================================================

ARITHMETIC_SINGLE_DF = [
    ("abs", ops.abs),
    ("inverse", ops.inverse),
    ("log", ops.log),
    ("sqrt", ops.sqrt),
    ("sign", ops.sign),
    ("reverse", ops.reverse),
    ("densify", ops.densify),
]


@pytest.mark.parametrize("op_name,op_func", ARITHMETIC_SINGLE_DF)
def test_arithmetic_single_df(op_name, op_func, benchmark_df):
    """Benchmark arithmetic operators that take a single DataFrame."""
    threshold_ms = PERFORMANCE_THRESHOLDS_MS[op_name]
    elapsed_ms = benchmark_operator(op_func, benchmark_df)

    assert elapsed_ms < threshold_ms, (
        f"{op_name}: {elapsed_ms:.1f}ms > {threshold_ms}ms threshold"
    )
    print(f"{op_name}: {elapsed_ms:.1f}ms (threshold: {threshold_ms}ms)")


ARITHMETIC_TWO_DF = [
    ("add", ops.add),
    ("subtract", ops.subtract),
    ("multiply", ops.multiply),
    ("divide", ops.divide),
    ("max", ops.max),
    ("min", ops.min),
]


@pytest.mark.parametrize("op_name,op_func", ARITHMETIC_TWO_DF)
def test_arithmetic_two_df(op_name, op_func, benchmark_df_pair):
    """Benchmark arithmetic operators that take two DataFrames."""
    df1, df2 = benchmark_df_pair
    threshold_ms = PERFORMANCE_THRESHOLDS_MS[op_name]
    elapsed_ms = benchmark_operator(op_func, df1, df2)

    assert elapsed_ms < threshold_ms, (
        f"{op_name}: {elapsed_ms:.1f}ms > {threshold_ms}ms threshold"
    )
    print(f"{op_name}: {elapsed_ms:.1f}ms (threshold: {threshold_ms}ms)")


ARITHMETIC_POWER = [
    ("power", ops.power),
    ("signed_power", ops.signed_power),
]


@pytest.mark.parametrize("op_name,op_func", ARITHMETIC_POWER)
def test_arithmetic_power(op_name, op_func, benchmark_df):
    """Benchmark power operators (DataFrame, scalar exponent)."""
    threshold_ms = PERFORMANCE_THRESHOLDS_MS[op_name]
    elapsed_ms = benchmark_operator(op_func, benchmark_df, 2)

    assert elapsed_ms < threshold_ms, (
        f"{op_name}: {elapsed_ms:.1f}ms > {threshold_ms}ms threshold"
    )
    print(f"{op_name}: {elapsed_ms:.1f}ms (threshold: {threshold_ms}ms)")


# =============================================================================
# LOGICAL OPERATORS (boolean operations)
# =============================================================================

LOGICAL_SINGLE_BOOL = [
    ("not_", ops.not_),
    ("is_nan", ops.is_nan),
]


@pytest.mark.parametrize("op_name,op_func", LOGICAL_SINGLE_BOOL)
def test_logical_single_bool(op_name, op_func, benchmark_bool_df, benchmark_df):
    """Benchmark logical operators that take a single DataFrame."""
    threshold_ms = PERFORMANCE_THRESHOLDS_MS[op_name]
    # is_nan works on numeric df, not_ works on bool df
    df = benchmark_df if op_name == "is_nan" else benchmark_bool_df
    elapsed_ms = benchmark_operator(op_func, df)

    assert elapsed_ms < threshold_ms, (
        f"{op_name}: {elapsed_ms:.1f}ms > {threshold_ms}ms threshold"
    )
    print(f"{op_name}: {elapsed_ms:.1f}ms (threshold: {threshold_ms}ms)")


LOGICAL_TWO_BOOL = [
    ("and_", ops.and_),
    ("or_", ops.or_),
]


@pytest.mark.parametrize("op_name,op_func", LOGICAL_TWO_BOOL)
def test_logical_two_bool(op_name, op_func, benchmark_bool_df):
    """Benchmark logical operators that take two boolean DataFrames."""
    threshold_ms = PERFORMANCE_THRESHOLDS_MS[op_name]
    # Use same bool df for both args (valid for AND/OR)
    elapsed_ms = benchmark_operator(op_func, benchmark_bool_df, benchmark_bool_df)

    assert elapsed_ms < threshold_ms, (
        f"{op_name}: {elapsed_ms:.1f}ms > {threshold_ms}ms threshold"
    )
    print(f"{op_name}: {elapsed_ms:.1f}ms (threshold: {threshold_ms}ms)")


def test_if_else(benchmark_bool_df, benchmark_df_pair):
    """Benchmark if_else operator."""
    df1, df2 = benchmark_df_pair
    threshold_ms = PERFORMANCE_THRESHOLDS_MS["if_else"]
    elapsed_ms = benchmark_operator(ops.if_else, benchmark_bool_df, df1, df2)

    assert elapsed_ms < threshold_ms, (
        f"if_else: {elapsed_ms:.1f}ms > {threshold_ms}ms threshold"
    )
    print(f"if_else: {elapsed_ms:.1f}ms (threshold: {threshold_ms}ms)")


COMPARISON_OPS = [
    ("lt", ops.lt),
    ("le", ops.le),
    ("gt", ops.gt),
    ("ge", ops.ge),
    ("eq", ops.eq),
    ("ne", ops.ne),
]


@pytest.mark.parametrize("op_name,op_func", COMPARISON_OPS)
def test_comparison_ops(op_name, op_func, benchmark_df):
    """Benchmark comparison operators (DataFrame vs scalar)."""
    threshold_ms = PERFORMANCE_THRESHOLDS_MS[op_name]
    elapsed_ms = benchmark_operator(op_func, benchmark_df, 0.0)

    assert elapsed_ms < threshold_ms, (
        f"{op_name}: {elapsed_ms:.1f}ms > {threshold_ms}ms threshold"
    )
    print(f"{op_name}: {elapsed_ms:.1f}ms (threshold: {threshold_ms}ms)")


# =============================================================================
# VECTOR OPERATORS (reduction across columns)
# =============================================================================


@pytest.fixture(scope="module")
def benchmark_list_df(benchmark_df):
    """Create DataFrame with list-type columns for vector operators.

    Vector operators work on list columns, not regular numeric columns.
    """
    # Convert each row to a list for list-type operations
    # Using a smaller sample since list operations are memory-intensive
    n_rows = 100
    n_cols = 50

    date_col = benchmark_df.columns[0]
    value_cols = benchmark_df.columns[1:n_cols + 1]

    # Create list columns: each cell contains a list of values
    data = {date_col: benchmark_df[date_col].head(n_rows)}
    for col in value_cols:
        values = benchmark_df[col].head(n_rows).to_list()
        # Wrap each value in a list to simulate vector fields
        data[col] = [[v] * 10 if v is not None else None for v in values]

    return pl.DataFrame(data)


VECTOR_OPS = [
    ("vec_avg", ops.vec_avg),
    ("vec_sum", ops.vec_sum),
]


@pytest.mark.parametrize("op_name,op_func", VECTOR_OPS)
def test_vector_ops(op_name, op_func, benchmark_list_df):
    """Benchmark vector operators (work on list columns)."""
    threshold_ms = PERFORMANCE_THRESHOLDS_MS[op_name]
    elapsed_ms = benchmark_operator(op_func, benchmark_list_df)

    assert elapsed_ms < threshold_ms, (
        f"{op_name}: {elapsed_ms:.1f}ms > {threshold_ms}ms threshold"
    )
    print(f"{op_name}: {elapsed_ms:.1f}ms (threshold: {threshold_ms}ms)")


# =============================================================================
# CROSS-SECTIONAL OPERATORS (row-wise operations)
# =============================================================================

CROSS_SECTIONAL_OPS = [
    ("rank", ops.rank),
    ("zscore", ops.zscore),
    ("normalize", ops.normalize),
    ("scale", ops.scale),
    ("winsorize", ops.winsorize),
]


@pytest.mark.parametrize("op_name,op_func", CROSS_SECTIONAL_OPS)
def test_cross_sectional_ops(op_name, op_func, benchmark_df):
    """Benchmark cross-sectional operators."""
    threshold_ms = PERFORMANCE_THRESHOLDS_MS[op_name]
    elapsed_ms = benchmark_operator(op_func, benchmark_df)

    assert elapsed_ms < threshold_ms, (
        f"{op_name}: {elapsed_ms:.1f}ms > {threshold_ms}ms threshold"
    )
    print(f"{op_name}: {elapsed_ms:.1f}ms (threshold: {threshold_ms}ms)")


def test_quantile(benchmark_df):
    """Benchmark quantile operator."""
    threshold_ms = PERFORMANCE_THRESHOLDS_MS["quantile"]
    elapsed_ms = benchmark_operator(ops.quantile, benchmark_df, driver="gaussian")

    assert elapsed_ms < threshold_ms, (
        f"quantile: {elapsed_ms:.1f}ms > {threshold_ms}ms threshold"
    )
    print(f"quantile: {elapsed_ms:.1f}ms (threshold: {threshold_ms}ms)")


def test_bucket(benchmark_df):
    """Benchmark bucket operator."""
    threshold_ms = PERFORMANCE_THRESHOLDS_MS["bucket"]
    elapsed_ms = benchmark_operator(ops.bucket, benchmark_df, range_spec="0,1,0.1")

    assert elapsed_ms < threshold_ms, (
        f"bucket: {elapsed_ms:.1f}ms > {threshold_ms}ms threshold"
    )
    print(f"bucket: {elapsed_ms:.1f}ms (threshold: {threshold_ms}ms)")


# =============================================================================
# TIME-SERIES OPERATORS - Basic (rolling window)
# =============================================================================

TS_BASIC_OPS = [
    ("ts_mean", ops.ts_mean),
    ("ts_sum", ops.ts_sum),
    ("ts_std", ops.ts_std),
    ("ts_min", ops.ts_min),
    ("ts_max", ops.ts_max),
]


@pytest.mark.parametrize("op_name,op_func", TS_BASIC_OPS)
def test_ts_basic_ops(op_name, op_func, benchmark_df, window):
    """Benchmark basic time-series operators."""
    threshold_ms = PERFORMANCE_THRESHOLDS_MS[op_name]
    elapsed_ms = benchmark_operator(op_func, benchmark_df, window)

    assert elapsed_ms < threshold_ms, (
        f"{op_name}: {elapsed_ms:.1f}ms > {threshold_ms}ms threshold"
    )
    print(f"{op_name}: {elapsed_ms:.1f}ms (threshold: {threshold_ms}ms)")


def test_ts_delta(benchmark_df):
    """Benchmark ts_delta operator."""
    threshold_ms = PERFORMANCE_THRESHOLDS_MS["ts_delta"]
    elapsed_ms = benchmark_operator(ops.ts_delta, benchmark_df, d=1)

    assert elapsed_ms < threshold_ms, (
        f"ts_delta: {elapsed_ms:.1f}ms > {threshold_ms}ms threshold"
    )
    print(f"ts_delta: {elapsed_ms:.1f}ms (threshold: {threshold_ms}ms)")


def test_ts_delay(benchmark_df):
    """Benchmark ts_delay operator."""
    threshold_ms = PERFORMANCE_THRESHOLDS_MS["ts_delay"]
    elapsed_ms = benchmark_operator(ops.ts_delay, benchmark_df, d=1)

    assert elapsed_ms < threshold_ms, (
        f"ts_delay: {elapsed_ms:.1f}ms > {threshold_ms}ms threshold"
    )
    print(f"ts_delay: {elapsed_ms:.1f}ms (threshold: {threshold_ms}ms)")


# =============================================================================
# TIME-SERIES OPERATORS - Rolling (more complex aggregations)
# =============================================================================

TS_ROLLING_OPS = [
    ("ts_product", ops.ts_product),
    ("ts_count_nans", ops.ts_count_nans),
    ("ts_zscore", ops.ts_zscore),
    ("ts_scale", ops.ts_scale),
    ("ts_av_diff", ops.ts_av_diff),
]


@pytest.mark.parametrize("op_name,op_func", TS_ROLLING_OPS)
def test_ts_rolling_ops(op_name, op_func, benchmark_df, window):
    """Benchmark rolling time-series operators."""
    threshold_ms = PERFORMANCE_THRESHOLDS_MS[op_name]
    elapsed_ms = benchmark_operator(op_func, benchmark_df, window)

    assert elapsed_ms < threshold_ms, (
        f"{op_name}: {elapsed_ms:.1f}ms > {threshold_ms}ms threshold"
    )
    print(f"{op_name}: {elapsed_ms:.1f}ms (threshold: {threshold_ms}ms)")


def test_ts_step(benchmark_df):
    """Benchmark ts_step operator (no window parameter)."""
    threshold_ms = PERFORMANCE_THRESHOLDS_MS["ts_step"]
    elapsed_ms = benchmark_operator(ops.ts_step, benchmark_df)

    assert elapsed_ms < threshold_ms, (
        f"ts_step: {elapsed_ms:.1f}ms > {threshold_ms}ms threshold"
    )
    print(f"ts_step: {elapsed_ms:.1f}ms (threshold: {threshold_ms}ms)")


# =============================================================================
# TIME-SERIES OPERATORS - Arg (finding indices)
# =============================================================================

TS_ARG_OPS = [
    ("ts_arg_max", ops.ts_arg_max),
    ("ts_arg_min", ops.ts_arg_min),
]


@pytest.mark.parametrize("op_name,op_func", TS_ARG_OPS)
def test_ts_arg_ops(op_name, op_func, benchmark_df, window):
    """Benchmark arg time-series operators."""
    threshold_ms = PERFORMANCE_THRESHOLDS_MS[op_name]
    elapsed_ms = benchmark_operator(op_func, benchmark_df, window)

    assert elapsed_ms < threshold_ms, (
        f"{op_name}: {elapsed_ms:.1f}ms > {threshold_ms}ms threshold"
    )
    print(f"{op_name}: {elapsed_ms:.1f}ms (threshold: {threshold_ms}ms)")


# =============================================================================
# TIME-SERIES OPERATORS - Lookback (conditional lookback)
# =============================================================================

def test_ts_backfill(benchmark_df, window):
    """Benchmark ts_backfill operator."""
    threshold_ms = PERFORMANCE_THRESHOLDS_MS["ts_backfill"]
    elapsed_ms = benchmark_operator(ops.ts_backfill, benchmark_df, window)

    assert elapsed_ms < threshold_ms, (
        f"ts_backfill: {elapsed_ms:.1f}ms > {threshold_ms}ms threshold"
    )
    print(f"ts_backfill: {elapsed_ms:.1f}ms (threshold: {threshold_ms}ms)")


def test_kth_element(benchmark_df, window):
    """Benchmark kth_element operator."""
    threshold_ms = PERFORMANCE_THRESHOLDS_MS["kth_element"]
    elapsed_ms = benchmark_operator(ops.kth_element, benchmark_df, d=window, k=5)

    assert elapsed_ms < threshold_ms, (
        f"kth_element: {elapsed_ms:.1f}ms > {threshold_ms}ms threshold"
    )
    print(f"kth_element: {elapsed_ms:.1f}ms (threshold: {threshold_ms}ms)")


def test_last_diff_value(benchmark_df, window):
    """Benchmark last_diff_value operator."""
    threshold_ms = PERFORMANCE_THRESHOLDS_MS["last_diff_value"]
    elapsed_ms = benchmark_operator(ops.last_diff_value, benchmark_df, window)

    assert elapsed_ms < threshold_ms, (
        f"last_diff_value: {elapsed_ms:.1f}ms > {threshold_ms}ms threshold"
    )
    print(f"last_diff_value: {elapsed_ms:.1f}ms (threshold: {threshold_ms}ms)")


def test_days_from_last_change(benchmark_df):
    """Benchmark days_from_last_change operator."""
    threshold_ms = PERFORMANCE_THRESHOLDS_MS["days_from_last_change"]
    elapsed_ms = benchmark_operator(ops.days_from_last_change, benchmark_df)

    assert elapsed_ms < threshold_ms, (
        f"days_from_last_change: {elapsed_ms:.1f}ms > {threshold_ms}ms threshold"
    )
    print(f"days_from_last_change: {elapsed_ms:.1f}ms (threshold: {threshold_ms}ms)")


# =============================================================================
# TIME-SERIES OPERATORS - Stateful (more complex state tracking)
# =============================================================================

def test_hump(benchmark_df):
    """Benchmark hump operator."""
    threshold_ms = PERFORMANCE_THRESHOLDS_MS["hump"]
    elapsed_ms = benchmark_operator(ops.hump, benchmark_df, hump_factor=0.01)

    assert elapsed_ms < threshold_ms, (
        f"hump: {elapsed_ms:.1f}ms > {threshold_ms}ms threshold"
    )
    print(f"hump: {elapsed_ms:.1f}ms (threshold: {threshold_ms}ms)")


def test_ts_decay_linear(benchmark_df, window):
    """Benchmark ts_decay_linear operator."""
    threshold_ms = PERFORMANCE_THRESHOLDS_MS["ts_decay_linear"]
    elapsed_ms = benchmark_operator(ops.ts_decay_linear, benchmark_df, window)

    assert elapsed_ms < threshold_ms, (
        f"ts_decay_linear: {elapsed_ms:.1f}ms > {threshold_ms}ms threshold"
    )
    print(f"ts_decay_linear: {elapsed_ms:.1f}ms (threshold: {threshold_ms}ms)")


def test_ts_rank(benchmark_df, window):
    """Benchmark ts_rank operator."""
    threshold_ms = PERFORMANCE_THRESHOLDS_MS["ts_rank"]
    elapsed_ms = benchmark_operator(ops.ts_rank, benchmark_df, window)

    assert elapsed_ms < threshold_ms, (
        f"ts_rank: {elapsed_ms:.1f}ms > {threshold_ms}ms threshold"
    )
    print(f"ts_rank: {elapsed_ms:.1f}ms (threshold: {threshold_ms}ms)")


# =============================================================================
# TIME-SERIES OPERATORS - Two-variable (cross-column computations)
# =============================================================================

def test_ts_corr(benchmark_df_pair, window):
    """Benchmark ts_corr operator."""
    df1, df2 = benchmark_df_pair
    threshold_ms = PERFORMANCE_THRESHOLDS_MS["ts_corr"]
    elapsed_ms = benchmark_operator(ops.ts_corr, df1, df2, window)

    assert elapsed_ms < threshold_ms, (
        f"ts_corr: {elapsed_ms:.1f}ms > {threshold_ms}ms threshold"
    )
    print(f"ts_corr: {elapsed_ms:.1f}ms (threshold: {threshold_ms}ms)")


def test_ts_covariance(benchmark_df_pair, window):
    """Benchmark ts_covariance operator."""
    df1, df2 = benchmark_df_pair
    threshold_ms = PERFORMANCE_THRESHOLDS_MS["ts_covariance"]
    elapsed_ms = benchmark_operator(ops.ts_covariance, df1, df2, window)

    assert elapsed_ms < threshold_ms, (
        f"ts_covariance: {elapsed_ms:.1f}ms > {threshold_ms}ms threshold"
    )
    print(f"ts_covariance: {elapsed_ms:.1f}ms (threshold: {threshold_ms}ms)")


def test_ts_quantile(benchmark_df, window):
    """Benchmark ts_quantile operator."""
    threshold_ms = PERFORMANCE_THRESHOLDS_MS["ts_quantile"]
    elapsed_ms = benchmark_operator(ops.ts_quantile, benchmark_df, window)

    assert elapsed_ms < threshold_ms, (
        f"ts_quantile: {elapsed_ms:.1f}ms > {threshold_ms}ms threshold"
    )
    print(f"ts_quantile: {elapsed_ms:.1f}ms (threshold: {threshold_ms}ms)")


def test_ts_regression(benchmark_df_pair, window):
    """Benchmark ts_regression operator."""
    df1, df2 = benchmark_df_pair
    threshold_ms = PERFORMANCE_THRESHOLDS_MS["ts_regression"]
    elapsed_ms = benchmark_operator(ops.ts_regression, df1, df2, window)

    assert elapsed_ms < threshold_ms, (
        f"ts_regression: {elapsed_ms:.1f}ms > {threshold_ms}ms threshold"
    )
    print(f"ts_regression: {elapsed_ms:.1f}ms (threshold: {threshold_ms}ms)")


# =============================================================================
# GROUP OPERATORS (grouped cross-sectional operations)
# =============================================================================

GROUP_OPS_SIMPLE = [
    ("group_rank", ops.group_rank),
    ("group_zscore", ops.group_zscore),
    ("group_scale", ops.group_scale),
    ("group_neutralize", ops.group_neutralize),
]


@pytest.mark.parametrize("op_name,op_func", GROUP_OPS_SIMPLE)
def test_group_ops_simple(op_name, op_func, benchmark_df, benchmark_group_mask):
    """Benchmark group operators (df, group)."""
    threshold_ms = PERFORMANCE_THRESHOLDS_MS[op_name]
    elapsed_ms = benchmark_operator(op_func, benchmark_df, benchmark_group_mask)

    assert elapsed_ms < threshold_ms, (
        f"{op_name}: {elapsed_ms:.1f}ms > {threshold_ms}ms threshold"
    )
    print(f"{op_name}: {elapsed_ms:.1f}ms (threshold: {threshold_ms}ms)")


def test_group_mean(benchmark_df, benchmark_group_mask):
    """Benchmark group_mean operator (df, weights, group)."""
    threshold_ms = PERFORMANCE_THRESHOLDS_MS["group_mean"]
    # Use benchmark_df as both values and weights
    elapsed_ms = benchmark_operator(
        ops.group_mean, benchmark_df, benchmark_df, benchmark_group_mask
    )

    assert elapsed_ms < threshold_ms, (
        f"group_mean: {elapsed_ms:.1f}ms > {threshold_ms}ms threshold"
    )
    print(f"group_mean: {elapsed_ms:.1f}ms (threshold: {threshold_ms}ms)")


def test_group_backfill(benchmark_df, benchmark_group_mask, window):
    """Benchmark group_backfill operator."""
    threshold_ms = PERFORMANCE_THRESHOLDS_MS["group_backfill"]
    elapsed_ms = benchmark_operator(
        ops.group_backfill, benchmark_df, benchmark_group_mask, window
    )

    assert elapsed_ms < threshold_ms, (
        f"group_backfill: {elapsed_ms:.1f}ms > {threshold_ms}ms threshold"
    )
    print(f"group_backfill: {elapsed_ms:.1f}ms (threshold: {threshold_ms}ms)")


# =============================================================================
# TRANSFORMATIONAL OPERATORS (complex conditional logic)
# =============================================================================

def test_trade_when(benchmark_bool_df, benchmark_df):
    """Benchmark trade_when operator."""
    threshold_ms = PERFORMANCE_THRESHOLDS_MS["trade_when"]
    # Use bool_df for triggers, benchmark_df for alpha, scalar for exit
    elapsed_ms = benchmark_operator(
        ops.trade_when, benchmark_bool_df, benchmark_df, -1
    )

    assert elapsed_ms < threshold_ms, (
        f"trade_when: {elapsed_ms:.1f}ms > {threshold_ms}ms threshold"
    )
    print(f"trade_when: {elapsed_ms:.1f}ms (threshold: {threshold_ms}ms)")
