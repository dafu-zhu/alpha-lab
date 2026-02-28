"""Stress tests for alpha operators under extreme conditions.

Tests verify operators don't crash and produce sane results when given:
1. Extreme values (1e308, -1e308, 1e-308)
2. High null density (90% nulls)
3. Window > data length
4. All-constant columns
5. Single-row DataFrames

Focus is on crash prevention and basic sanity, not exact numerical accuracy.
"""

from datetime import date

import numpy as np
import polars as pl
import pytest

from alphalab.api.operators import (
    # Arithmetic
    abs as op_abs,  # Avoid shadowing builtin
    add,
    densify,
    divide,
    inverse,
    log,
    max as op_max,  # Avoid shadowing builtin
    min as op_min,  # Avoid shadowing builtin
    multiply,
    power,
    reverse,
    sign,
    signed_power,
    sqrt,
    subtract,
    # Cross-sectional
    bucket,
    normalize,
    quantile,
    rank,
    scale,
    winsorize,
    zscore,
    # Group
    group_backfill,
    group_mean,
    group_neutralize,
    group_rank,
    group_scale,
    group_zscore,
    # Logical
    and_,
    eq,
    ge,
    gt,
    if_else,
    is_nan,
    le,
    lt,
    ne,
    not_,
    or_,
    # Time-series
    days_from_last_change,
    hump,
    kth_element,
    last_diff_value,
    ts_arg_max,
    ts_arg_min,
    ts_av_diff,
    ts_backfill,
    ts_corr,
    ts_count_nans,
    ts_covariance,
    ts_decay_linear,
    ts_delay,
    ts_delta,
    ts_max,
    ts_mean,
    ts_min,
    ts_product,
    ts_quantile,
    ts_rank,
    ts_regression,
    ts_scale,
    ts_std,
    ts_step,
    ts_sum,
    ts_zscore,
    # Transformational
    trade_when,
    # Vector
    vec_avg,
    vec_sum,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def extreme_values_df() -> pl.DataFrame:
    """DataFrame with extreme floating point values."""
    dates = pl.date_range(date(2024, 1, 1), date(2024, 1, 10), eager=True)
    return pl.DataFrame({
        "Date": dates,
        "MAX_POS": [1e308] * 10,
        "MAX_NEG": [-1e308] * 10,
        "MIN_POS": [1e-308] * 10,
        "MIXED": [1e308, -1e308, 1e-308, -1e-308, 0.0, 1e308, -1e308, 1e-308, -1e-308, 0.0],
    })


@pytest.fixture
def extreme_values_pair() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Two DataFrames with extreme values for two-variable operators."""
    dates = pl.date_range(date(2024, 1, 1), date(2024, 1, 10), eager=True)
    df1 = pl.DataFrame({
        "Date": dates,
        "A": [1e308, 1e100, 1e-100, 1e-308, 0.0, -1e308, -1e100, -1e-100, -1e-308, 0.0],
        "B": [1e200, -1e200, 1e-200, -1e-200, 1.0, -1.0, 1e150, -1e150, 1e-150, -1e-150],
    })
    df2 = pl.DataFrame({
        "Date": dates,
        "A": [1e307, 1e99, 1e-99, 1e-307, 0.0, -1e307, -1e99, -1e-99, -1e-307, 0.0],
        "B": [1e199, -1e199, 1e-199, -1e-199, 2.0, -2.0, 1e149, -1e149, 1e-149, -1e-149],
    })
    return df1, df2


@pytest.fixture
def high_null_density_df() -> pl.DataFrame:
    """DataFrame with 90% null values."""
    np.random.seed(42)
    n_rows = 100
    dates = pl.date_range(date(2024, 1, 1), date(2024, 4, 9), eager=True)[:n_rows]

    # Create data with 90% nulls
    def make_sparse_column():
        data = np.random.randn(n_rows) * 100
        mask = np.random.random(n_rows) < 0.9  # 90% null
        data[mask] = np.nan
        return data.tolist()

    return pl.DataFrame({
        "Date": dates,
        "SPARSE_A": make_sparse_column(),
        "SPARSE_B": make_sparse_column(),
        "SPARSE_C": make_sparse_column(),
    })


@pytest.fixture
def constant_columns_df() -> pl.DataFrame:
    """DataFrame with all-constant columns."""
    dates = pl.date_range(date(2024, 1, 1), date(2024, 1, 20), eager=True)
    return pl.DataFrame({
        "Date": dates,
        "CONST_ZERO": [0.0] * 20,
        "CONST_ONE": [1.0] * 20,
        "CONST_NEG": [-100.0] * 20,
        "CONST_BIG": [1e10] * 20,
    })


@pytest.fixture
def single_row_df() -> pl.DataFrame:
    """DataFrame with a single row."""
    return pl.DataFrame({
        "Date": [date(2024, 1, 1)],
        "A": [100.0],
        "B": [200.0],
        "C": [150.0],
    })


@pytest.fixture
def short_df() -> pl.DataFrame:
    """DataFrame shorter than typical window sizes."""
    dates = pl.date_range(date(2024, 1, 1), date(2024, 1, 3), eager=True)
    return pl.DataFrame({
        "Date": dates,
        "A": [100.0, 101.0, 102.0],
        "B": [200.0, 201.0, 202.0],
        "C": [150.0, 151.0, 152.0],
    })


@pytest.fixture
def group_mask_df() -> pl.DataFrame:
    """DataFrame with group assignments for testing group operators."""
    dates = pl.date_range(date(2024, 1, 1), date(2024, 1, 10), eager=True)
    return pl.DataFrame({
        "Date": dates,
        # Two groups: "A" and "B"
        "SYM1": ["A"] * 10,
        "SYM2": ["A"] * 10,
        "SYM3": ["B"] * 10,
        "SYM4": ["B"] * 10,
    })


@pytest.fixture
def group_values_df() -> pl.DataFrame:
    """DataFrame with values for testing group operators."""
    dates = pl.date_range(date(2024, 1, 1), date(2024, 1, 10), eager=True)
    np.random.seed(42)
    return pl.DataFrame({
        "Date": dates,
        "SYM1": np.random.randn(10) * 10 + 100,
        "SYM2": np.random.randn(10) * 10 + 100,
        "SYM3": np.random.randn(10) * 10 + 200,
        "SYM4": np.random.randn(10) * 10 + 200,
    })


@pytest.fixture
def group_weights_df() -> pl.DataFrame:
    """DataFrame with weights for testing group_mean."""
    dates = pl.date_range(date(2024, 1, 1), date(2024, 1, 10), eager=True)
    return pl.DataFrame({
        "Date": dates,
        "SYM1": [1.0] * 10,
        "SYM2": [2.0] * 10,
        "SYM3": [1.0] * 10,
        "SYM4": [2.0] * 10,
    })


@pytest.fixture
def boolean_df() -> pl.DataFrame:
    """DataFrame with boolean values for logical operators."""
    dates = pl.date_range(date(2024, 1, 1), date(2024, 1, 10), eager=True)
    return pl.DataFrame({
        "Date": dates,
        "A": [True, False, True, False, True, False, True, False, True, False],
        "B": [True, True, False, False, True, True, False, False, True, True],
        "C": [False, False, False, True, True, True, False, False, False, True],
    })


@pytest.fixture
def list_df() -> pl.DataFrame:
    """DataFrame with list columns for vector operators."""
    dates = pl.date_range(date(2024, 1, 1), date(2024, 1, 5), eager=True)
    return pl.DataFrame({
        "Date": dates,
        "A": [[1.0, 2.0, 3.0], [4.0, 5.0], [6.0], [], [1.0, np.nan, 3.0]],
        "B": [[10.0, 20.0], [30.0, 40.0, 50.0], [60.0, 70.0], [80.0], [np.nan]],
    })


# =============================================================================
# TEST HELPERS
# =============================================================================


def assert_valid_result(result: pl.DataFrame, expected_shape: tuple[int, int]) -> None:
    """Assert result is a valid DataFrame with expected shape.

    Does NOT require all values to be finite - allows inf/NaN for extreme inputs.
    """
    assert isinstance(result, pl.DataFrame), f"Expected DataFrame, got {type(result)}"
    assert result.shape == expected_shape, f"Expected shape {expected_shape}, got {result.shape}"


def assert_no_crash(func, *args, **kwargs) -> pl.DataFrame:
    """Assert function doesn't crash and returns a DataFrame."""
    try:
        result = func(*args, **kwargs)
        assert isinstance(result, pl.DataFrame), f"Expected DataFrame, got {type(result)}"
        return result
    except Exception as e:
        pytest.fail(f"Operator {func.__name__} crashed with: {type(e).__name__}: {e}")


# =============================================================================
# 1. EXTREME VALUES TESTS
# =============================================================================


class TestExtremeValues:
    """Test operators with extreme floating point values (1e308, -1e308, 1e-308)."""

    def test_ts_mean_extreme(self, extreme_values_df: pl.DataFrame) -> None:
        """ts_mean should handle extreme values without crashing."""
        result = assert_no_crash(ts_mean, extreme_values_df, 3)
        assert_valid_result(result, extreme_values_df.shape)

    def test_ts_std_extreme(self, extreme_values_df: pl.DataFrame) -> None:
        """ts_std should handle extreme values without crashing."""
        result = assert_no_crash(ts_std, extreme_values_df, 3)
        assert_valid_result(result, extreme_values_df.shape)

    def test_ts_sum_extreme(self, extreme_values_df: pl.DataFrame) -> None:
        """ts_sum should handle extreme values (may produce inf)."""
        result = assert_no_crash(ts_sum, extreme_values_df, 3)
        assert_valid_result(result, extreme_values_df.shape)

    def test_ts_min_max_extreme(self, extreme_values_df: pl.DataFrame) -> None:
        """ts_min/ts_max should handle extreme values."""
        result_min = assert_no_crash(ts_min, extreme_values_df, 3)
        result_max = assert_no_crash(ts_max, extreme_values_df, 3)
        assert_valid_result(result_min, extreme_values_df.shape)
        assert_valid_result(result_max, extreme_values_df.shape)

    def test_ts_product_extreme(self, extreme_values_df: pl.DataFrame) -> None:
        """ts_product with extreme values (will likely overflow to inf)."""
        result = assert_no_crash(ts_product, extreme_values_df, 2)
        assert_valid_result(result, extreme_values_df.shape)

    def test_ts_corr_extreme(self, extreme_values_pair: tuple[pl.DataFrame, pl.DataFrame]) -> None:
        """ts_corr should handle extreme values."""
        df1, df2 = extreme_values_pair
        result = assert_no_crash(ts_corr, df1, df2, 3)
        assert_valid_result(result, df1.shape)

    def test_ts_covariance_extreme(self, extreme_values_pair: tuple[pl.DataFrame, pl.DataFrame]) -> None:
        """ts_covariance should handle extreme values."""
        df1, df2 = extreme_values_pair
        result = assert_no_crash(ts_covariance, df1, df2, 3)
        assert_valid_result(result, df1.shape)

    def test_rank_extreme(self, extreme_values_df: pl.DataFrame) -> None:
        """rank should handle extreme values."""
        result = assert_no_crash(rank, extreme_values_df)
        assert_valid_result(result, extreme_values_df.shape)

        # Ranks should be in [0, 1] or NaN
        value_cols = extreme_values_df.columns[1:]
        for col in value_cols:
            values = result[col].to_numpy()
            valid = values[~np.isnan(values)]
            assert np.all(valid >= 0.0) and np.all(valid <= 1.0), f"Ranks out of [0,1] for {col}"

    def test_zscore_extreme(self, extreme_values_df: pl.DataFrame) -> None:
        """zscore should handle extreme values."""
        result = assert_no_crash(zscore, extreme_values_df)
        assert_valid_result(result, extreme_values_df.shape)

    def test_quantile_extreme(self, extreme_values_df: pl.DataFrame) -> None:
        """quantile should handle extreme values."""
        result = assert_no_crash(quantile, extreme_values_df)
        assert_valid_result(result, extreme_values_df.shape)

    def test_arithmetic_extreme(self, extreme_values_df: pl.DataFrame) -> None:
        """Arithmetic operators should handle extreme values."""
        # op_abs
        result = assert_no_crash(op_abs, extreme_values_df)
        assert_valid_result(result, extreme_values_df.shape)

        # sign
        result = assert_no_crash(sign, extreme_values_df)
        assert_valid_result(result, extreme_values_df.shape)

        # sqrt (may produce NaN for negative values)
        result = assert_no_crash(sqrt, extreme_values_df)
        assert_valid_result(result, extreme_values_df.shape)

    def test_log_extreme(self, extreme_values_df: pl.DataFrame) -> None:
        """log should handle extreme values (NaN for negative/zero)."""
        result = assert_no_crash(log, extreme_values_df)
        assert_valid_result(result, extreme_values_df.shape)

    def test_inverse_extreme(self, extreme_values_df: pl.DataFrame) -> None:
        """inverse should handle extreme values (very small result for large values)."""
        result = assert_no_crash(inverse, extreme_values_df)
        assert_valid_result(result, extreme_values_df.shape)


# =============================================================================
# 2. HIGH NULL DENSITY TESTS
# =============================================================================


class TestHighNullDensity:
    """Test operators with 90% null values."""

    def test_ts_mean_sparse(self, high_null_density_df: pl.DataFrame) -> None:
        """ts_mean should handle sparse data."""
        result = assert_no_crash(ts_mean, high_null_density_df, 10)
        assert_valid_result(result, high_null_density_df.shape)

    def test_ts_std_sparse(self, high_null_density_df: pl.DataFrame) -> None:
        """ts_std should handle sparse data."""
        result = assert_no_crash(ts_std, high_null_density_df, 10)
        assert_valid_result(result, high_null_density_df.shape)

    def test_ts_sum_sparse(self, high_null_density_df: pl.DataFrame) -> None:
        """ts_sum should handle sparse data."""
        result = assert_no_crash(ts_sum, high_null_density_df, 10)
        assert_valid_result(result, high_null_density_df.shape)

    def test_ts_delta_sparse(self, high_null_density_df: pl.DataFrame) -> None:
        """ts_delta should handle sparse data."""
        result = assert_no_crash(ts_delta, high_null_density_df, 5)
        assert_valid_result(result, high_null_density_df.shape)

    def test_ts_delay_sparse(self, high_null_density_df: pl.DataFrame) -> None:
        """ts_delay should handle sparse data."""
        result = assert_no_crash(ts_delay, high_null_density_df, 5)
        assert_valid_result(result, high_null_density_df.shape)

    def test_ts_rank_sparse(self, high_null_density_df: pl.DataFrame) -> None:
        """ts_rank should handle sparse data."""
        result = assert_no_crash(ts_rank, high_null_density_df, 10)
        assert_valid_result(result, high_null_density_df.shape)

    def test_ts_zscore_sparse(self, high_null_density_df: pl.DataFrame) -> None:
        """ts_zscore should handle sparse data."""
        result = assert_no_crash(ts_zscore, high_null_density_df, 10)
        assert_valid_result(result, high_null_density_df.shape)

    def test_ts_decay_linear_sparse(self, high_null_density_df: pl.DataFrame) -> None:
        """ts_decay_linear should handle sparse data."""
        result = assert_no_crash(ts_decay_linear, high_null_density_df, 10)
        assert_valid_result(result, high_null_density_df.shape)

    def test_rank_sparse(self, high_null_density_df: pl.DataFrame) -> None:
        """Cross-sectional rank should handle sparse data."""
        result = assert_no_crash(rank, high_null_density_df)
        assert_valid_result(result, high_null_density_df.shape)

    def test_zscore_sparse(self, high_null_density_df: pl.DataFrame) -> None:
        """Cross-sectional zscore should handle sparse data."""
        result = assert_no_crash(zscore, high_null_density_df)
        assert_valid_result(result, high_null_density_df.shape)

    def test_winsorize_sparse(self, high_null_density_df: pl.DataFrame) -> None:
        """winsorize should handle sparse data."""
        result = assert_no_crash(winsorize, high_null_density_df, std=2.0)
        assert_valid_result(result, high_null_density_df.shape)

    def test_bucket_sparse(self, high_null_density_df: pl.DataFrame) -> None:
        """bucket should handle sparse data."""
        result = assert_no_crash(bucket, high_null_density_df, "0,100,10")
        assert_valid_result(result, high_null_density_df.shape)


# =============================================================================
# 3. WINDOW > DATA LENGTH TESTS
# =============================================================================


class TestWindowExceedsData:
    """Test operators when window size exceeds data length."""

    def test_ts_mean_oversized_window(self, short_df: pl.DataFrame) -> None:
        """ts_mean with window > data length should not crash."""
        result = assert_no_crash(ts_mean, short_df, 100)
        assert_valid_result(result, short_df.shape)

        # Should still produce values (min_periods=1)
        value_cols = short_df.columns[1:]
        for col in value_cols:
            values = result[col].to_numpy()
            # At least some valid values expected
            assert not np.all(np.isnan(values)), f"All NaN for {col}"

    def test_ts_std_oversized_window(self, short_df: pl.DataFrame) -> None:
        """ts_std with window > data length should not crash."""
        result = assert_no_crash(ts_std, short_df, 100)
        assert_valid_result(result, short_df.shape)

    def test_ts_sum_oversized_window(self, short_df: pl.DataFrame) -> None:
        """ts_sum with window > data length should not crash."""
        result = assert_no_crash(ts_sum, short_df, 100)
        assert_valid_result(result, short_df.shape)

    def test_ts_min_max_oversized_window(self, short_df: pl.DataFrame) -> None:
        """ts_min/ts_max with window > data length should not crash."""
        result_min = assert_no_crash(ts_min, short_df, 100)
        result_max = assert_no_crash(ts_max, short_df, 100)
        assert_valid_result(result_min, short_df.shape)
        assert_valid_result(result_max, short_df.shape)

    def test_ts_arg_min_max_oversized_window(self, short_df: pl.DataFrame) -> None:
        """ts_arg_min/ts_arg_max with window > data length should not crash."""
        result_arg_min = assert_no_crash(ts_arg_min, short_df, 100)
        result_arg_max = assert_no_crash(ts_arg_max, short_df, 100)
        assert_valid_result(result_arg_min, short_df.shape)
        assert_valid_result(result_arg_max, short_df.shape)

    def test_ts_delta_oversized_window(self, short_df: pl.DataFrame) -> None:
        """ts_delta with d > data length should not crash."""
        result = assert_no_crash(ts_delta, short_df, 100)
        assert_valid_result(result, short_df.shape)

    def test_ts_delay_oversized_window(self, short_df: pl.DataFrame) -> None:
        """ts_delay with d > data length should not crash."""
        result = assert_no_crash(ts_delay, short_df, 100)
        assert_valid_result(result, short_df.shape)

    def test_ts_rank_oversized_window(self, short_df: pl.DataFrame) -> None:
        """ts_rank with window > data length should not crash."""
        result = assert_no_crash(ts_rank, short_df, 100)
        assert_valid_result(result, short_df.shape)

    def test_ts_zscore_oversized_window(self, short_df: pl.DataFrame) -> None:
        """ts_zscore with window > data length should not crash."""
        result = assert_no_crash(ts_zscore, short_df, 100)
        assert_valid_result(result, short_df.shape)

    def test_ts_decay_linear_oversized_window(self, short_df: pl.DataFrame) -> None:
        """ts_decay_linear with window > data length should not crash."""
        result = assert_no_crash(ts_decay_linear, short_df, 100)
        assert_valid_result(result, short_df.shape)

    def test_ts_regression_oversized_window(self, short_df: pl.DataFrame) -> None:
        """ts_regression with window > data length should not crash."""
        df2 = short_df.with_columns([
            pl.col(c) * 2 for c in short_df.columns[1:]
        ])
        result = assert_no_crash(ts_regression, short_df, df2, 100)
        assert_valid_result(result, short_df.shape)


# =============================================================================
# 4. ALL-CONSTANT COLUMNS TESTS
# =============================================================================


class TestConstantColumns:
    """Test operators with all-constant columns (zero variance)."""

    def test_ts_mean_constant(self, constant_columns_df: pl.DataFrame) -> None:
        """ts_mean on constant data should return same values."""
        result = assert_no_crash(ts_mean, constant_columns_df, 5)
        assert_valid_result(result, constant_columns_df.shape)

        # Mean of constant values should be the constant
        for col in ["CONST_ONE", "CONST_NEG", "CONST_BIG"]:
            orig = constant_columns_df[col][0]
            result_vals = result[col].to_numpy()
            valid = result_vals[~np.isnan(result_vals)]
            if len(valid) > 0:
                np.testing.assert_allclose(valid, orig, rtol=1e-10)

    def test_ts_std_constant(self, constant_columns_df: pl.DataFrame) -> None:
        """ts_std on constant data should return 0 (or null for single value)."""
        result = assert_no_crash(ts_std, constant_columns_df, 5)
        assert_valid_result(result, constant_columns_df.shape)

        # Std of constant values should be 0
        for col in ["CONST_ONE", "CONST_NEG", "CONST_BIG"]:
            result_vals = result[col].to_numpy()
            valid = result_vals[~np.isnan(result_vals)]
            if len(valid) > 0:
                np.testing.assert_allclose(valid, 0.0, atol=1e-10)

    def test_ts_zscore_constant(self, constant_columns_df: pl.DataFrame) -> None:
        """ts_zscore on constant data should handle zero std gracefully."""
        result = assert_no_crash(ts_zscore, constant_columns_df, 5)
        assert_valid_result(result, constant_columns_df.shape)
        # Should produce NaN or 0 when std=0

    def test_zscore_constant_row(self, constant_columns_df: pl.DataFrame) -> None:
        """Cross-sectional zscore with constant row should return NaN."""
        # Create a row where all values are the same
        dates = pl.date_range(date(2024, 1, 1), date(2024, 1, 5), eager=True)
        constant_row_df = pl.DataFrame({
            "Date": dates,
            "A": [100.0] * 5,
            "B": [100.0] * 5,
            "C": [100.0] * 5,
        })

        result = assert_no_crash(zscore, constant_row_df)
        assert_valid_result(result, constant_row_df.shape)

        # All values should be NaN (std=0 case)
        for col in ["A", "B", "C"]:
            values = result[col].to_numpy()
            assert np.all(np.isnan(values)), f"zscore should be NaN for constant row in {col}"

    def test_rank_constant(self, constant_columns_df: pl.DataFrame) -> None:
        """rank should handle constant columns."""
        result = assert_no_crash(rank, constant_columns_df)
        assert_valid_result(result, constant_columns_df.shape)

    def test_quantile_constant(self, constant_columns_df: pl.DataFrame) -> None:
        """quantile should handle constant columns."""
        result = assert_no_crash(quantile, constant_columns_df)
        assert_valid_result(result, constant_columns_df.shape)

    def test_normalize_constant(self, constant_columns_df: pl.DataFrame) -> None:
        """normalize should handle constant columns."""
        result = assert_no_crash(normalize, constant_columns_df)
        assert_valid_result(result, constant_columns_df.shape)

    def test_scale_constant(self, constant_columns_df: pl.DataFrame) -> None:
        """scale should handle constant columns (zero sum/abs case)."""
        result = assert_no_crash(scale, constant_columns_df)
        assert_valid_result(result, constant_columns_df.shape)

    def test_ts_delta_constant(self, constant_columns_df: pl.DataFrame) -> None:
        """ts_delta on constant data should return 0."""
        result = assert_no_crash(ts_delta, constant_columns_df, 5)
        assert_valid_result(result, constant_columns_df.shape)

        # Delta of constant should be 0
        for col in ["CONST_ONE", "CONST_NEG"]:
            result_vals = result[col].to_numpy()
            valid = result_vals[~np.isnan(result_vals)]
            if len(valid) > 0:
                np.testing.assert_allclose(valid, 0.0, atol=1e-10)


# =============================================================================
# 5. SINGLE-ROW DATAFRAME TESTS
# =============================================================================


class TestSingleRow:
    """Test operators with single-row DataFrames."""

    def test_ts_mean_single_row(self, single_row_df: pl.DataFrame) -> None:
        """ts_mean with single row should return the value."""
        result = assert_no_crash(ts_mean, single_row_df, 5)
        assert_valid_result(result, single_row_df.shape)

        # Mean of single value = the value
        for col in ["A", "B", "C"]:
            assert result[col][0] == single_row_df[col][0]

    def test_ts_std_single_row(self, single_row_df: pl.DataFrame) -> None:
        """ts_std with single row should return null (needs min 2 samples)."""
        result = assert_no_crash(ts_std, single_row_df, 5)
        assert_valid_result(result, single_row_df.shape)

    def test_ts_sum_single_row(self, single_row_df: pl.DataFrame) -> None:
        """ts_sum with single row should return the value."""
        result = assert_no_crash(ts_sum, single_row_df, 5)
        assert_valid_result(result, single_row_df.shape)

    def test_ts_delta_single_row(self, single_row_df: pl.DataFrame) -> None:
        """ts_delta with single row should return null."""
        result = assert_no_crash(ts_delta, single_row_df, 1)
        assert_valid_result(result, single_row_df.shape)

    def test_ts_delay_single_row(self, single_row_df: pl.DataFrame) -> None:
        """ts_delay with single row should return null."""
        result = assert_no_crash(ts_delay, single_row_df, 1)
        assert_valid_result(result, single_row_df.shape)

    def test_rank_single_row(self, single_row_df: pl.DataFrame) -> None:
        """rank with single row should work."""
        result = assert_no_crash(rank, single_row_df)
        assert_valid_result(result, single_row_df.shape)

    def test_zscore_single_row(self, single_row_df: pl.DataFrame) -> None:
        """zscore with single row should work."""
        result = assert_no_crash(zscore, single_row_df)
        assert_valid_result(result, single_row_df.shape)

    def test_quantile_single_row(self, single_row_df: pl.DataFrame) -> None:
        """quantile with single row should work."""
        result = assert_no_crash(quantile, single_row_df)
        assert_valid_result(result, single_row_df.shape)

    def test_winsorize_single_row(self, single_row_df: pl.DataFrame) -> None:
        """winsorize with single row should work."""
        result = assert_no_crash(winsorize, single_row_df)
        assert_valid_result(result, single_row_df.shape)

    def test_bucket_single_row(self, single_row_df: pl.DataFrame) -> None:
        """bucket with single row should work."""
        result = assert_no_crash(bucket, single_row_df, "0,250,50")
        assert_valid_result(result, single_row_df.shape)

    def test_normalize_single_row(self, single_row_df: pl.DataFrame) -> None:
        """normalize with single row should work."""
        result = assert_no_crash(normalize, single_row_df)
        assert_valid_result(result, single_row_df.shape)

    def test_scale_single_row(self, single_row_df: pl.DataFrame) -> None:
        """scale with single row should work."""
        result = assert_no_crash(scale, single_row_df)
        assert_valid_result(result, single_row_df.shape)

    def test_arithmetic_single_row(self, single_row_df: pl.DataFrame) -> None:
        """Arithmetic operators with single row should work."""
        result_abs = assert_no_crash(op_abs, single_row_df)
        result_sign = assert_no_crash(sign, single_row_df)
        result_log = assert_no_crash(log, single_row_df)
        result_sqrt = assert_no_crash(sqrt, single_row_df)

        assert_valid_result(result_abs, single_row_df.shape)
        assert_valid_result(result_sign, single_row_df.shape)
        assert_valid_result(result_log, single_row_df.shape)
        assert_valid_result(result_sqrt, single_row_df.shape)


# =============================================================================
# COMBINED STRESS TESTS
# =============================================================================


class TestCombinedStress:
    """Combined stress scenarios."""

    def test_extreme_plus_nulls(self) -> None:
        """Test extreme values combined with high null density."""
        dates = pl.date_range(date(2024, 1, 1), date(2024, 1, 20), eager=True)
        data = [
            1e308, np.nan, np.nan, np.nan, np.nan,
            -1e308, np.nan, np.nan, np.nan, np.nan,
            1e-308, np.nan, np.nan, np.nan, np.nan,
            0.0, np.nan, np.nan, np.nan, np.nan,
        ]
        df = pl.DataFrame({
            "Date": dates,
            "A": data,
            "B": data[::-1],
        })

        # All operators should handle this
        result_mean = assert_no_crash(ts_mean, df, 5)
        result_rank = assert_no_crash(rank, df)
        result_zscore = assert_no_crash(zscore, df)

        assert_valid_result(result_mean, df.shape)
        assert_valid_result(result_rank, df.shape)
        assert_valid_result(result_zscore, df.shape)

    def test_binary_ops_extreme_values(self) -> None:
        """Test binary operators with extreme values."""
        dates = pl.date_range(date(2024, 1, 1), date(2024, 1, 5), eager=True)
        df1 = pl.DataFrame({
            "Date": dates,
            "A": [1e308, -1e308, 1e-308, 0.0, 1.0],
        })
        df2 = pl.DataFrame({
            "Date": dates,
            "A": [1e-308, 1e308, -1e308, 1.0, 0.0],
        })

        # These may produce inf/nan but shouldn't crash
        result_add = assert_no_crash(add, df1, df2)
        result_sub = assert_no_crash(subtract, df1, df2)
        result_mul = assert_no_crash(multiply, df1, df2)
        result_div = assert_no_crash(divide, df1, df2)

        assert_valid_result(result_add, df1.shape)
        assert_valid_result(result_sub, df1.shape)
        assert_valid_result(result_mul, df1.shape)
        assert_valid_result(result_div, df1.shape)

    def test_power_edge_cases(self) -> None:
        """Test power operator with edge cases."""
        dates = pl.date_range(date(2024, 1, 1), date(2024, 1, 5), eager=True)
        df = pl.DataFrame({
            "Date": dates,
            "A": [0.0, 1.0, -1.0, 2.0, 1e100],
        })

        # Various power edge cases
        result_pow0 = assert_no_crash(power, df, 0)
        result_pow1 = assert_no_crash(power, df, 1)
        result_pow_neg = assert_no_crash(power, df, -1)
        result_pow_frac = assert_no_crash(power, df, 0.5)

        assert_valid_result(result_pow0, df.shape)
        assert_valid_result(result_pow1, df.shape)
        assert_valid_result(result_pow_neg, df.shape)
        assert_valid_result(result_pow_frac, df.shape)

    def test_signed_power_edge_cases(self) -> None:
        """Test signed_power operator with edge cases."""
        dates = pl.date_range(date(2024, 1, 1), date(2024, 1, 5), eager=True)
        df = pl.DataFrame({
            "Date": dates,
            "A": [0.0, 1.0, -1.0, -2.0, 1e100],
        })

        result = assert_no_crash(signed_power, df, 0.5)
        assert_valid_result(result, df.shape)


# =============================================================================
# 6. LOGICAL OPERATORS TESTS
# =============================================================================


class TestLogicalOperators:
    """Test logical and comparison operators."""

    def test_and_or_not(self, boolean_df: pl.DataFrame) -> None:
        """Test logical AND, OR, NOT operators."""
        # Create two boolean DataFrames
        df1 = boolean_df.select("Date", "A", "B", "C")
        df2 = boolean_df.select(
            "Date",
            pl.col("A").alias("A"),
            pl.col("B").alias("B"),
            pl.col("C").alias("C"),
        )

        result_and = assert_no_crash(and_, df1, df2)
        result_or = assert_no_crash(or_, df1, df2)
        result_not = assert_no_crash(not_, df1)

        assert_valid_result(result_and, df1.shape)
        assert_valid_result(result_or, df1.shape)
        assert_valid_result(result_not, df1.shape)

    def test_comparison_operators_with_scalar(self, extreme_values_df: pl.DataFrame) -> None:
        """Test comparison operators with scalar values."""
        result_lt = assert_no_crash(lt, extreme_values_df, 0.0)
        result_le = assert_no_crash(le, extreme_values_df, 0.0)
        result_gt = assert_no_crash(gt, extreme_values_df, 0.0)
        result_ge = assert_no_crash(ge, extreme_values_df, 0.0)
        result_eq = assert_no_crash(eq, extreme_values_df, 0.0)
        result_ne = assert_no_crash(ne, extreme_values_df, 0.0)

        for result in [result_lt, result_le, result_gt, result_ge, result_eq, result_ne]:
            assert_valid_result(result, extreme_values_df.shape)

    def test_comparison_operators_with_dataframe(
        self, extreme_values_pair: tuple[pl.DataFrame, pl.DataFrame]
    ) -> None:
        """Test comparison operators with two DataFrames."""
        df1, df2 = extreme_values_pair

        result_lt = assert_no_crash(lt, df1, df2)
        result_le = assert_no_crash(le, df1, df2)
        result_gt = assert_no_crash(gt, df1, df2)
        result_ge = assert_no_crash(ge, df1, df2)
        result_eq = assert_no_crash(eq, df1, df2)
        result_ne = assert_no_crash(ne, df1, df2)

        for result in [result_lt, result_le, result_gt, result_ge, result_eq, result_ne]:
            assert_valid_result(result, df1.shape)

    def test_is_nan(self, high_null_density_df: pl.DataFrame) -> None:
        """Test is_nan with high null density."""
        result = assert_no_crash(is_nan, high_null_density_df)
        assert_valid_result(result, high_null_density_df.shape)

        # Should return all boolean values
        for col in high_null_density_df.columns[1:]:
            assert result[col].dtype == pl.Boolean

    def test_if_else_scalar(self, boolean_df: pl.DataFrame) -> None:
        """Test if_else with scalar then/else values."""
        cond = boolean_df.select("Date", "A", "B", "C")
        result = assert_no_crash(if_else, cond, 1.0, 0.0)
        assert_valid_result(result, cond.shape)

    def test_if_else_dataframe(
        self,
        boolean_df: pl.DataFrame,
        extreme_values_df: pl.DataFrame,
    ) -> None:
        """Test if_else with DataFrame then/else values."""
        # Create compatible DataFrames
        dates = pl.date_range(date(2024, 1, 1), date(2024, 1, 10), eager=True)
        cond = pl.DataFrame({
            "Date": dates,
            "MAX_POS": [True, False] * 5,
            "MAX_NEG": [False, True] * 5,
            "MIN_POS": [True, True, False, False] * 2 + [True, True],
            "MIXED": [False] * 10,
        })
        then_df = extreme_values_df
        else_df = extreme_values_df.select(
            "Date",
            *[(-pl.col(c)).alias(c) for c in extreme_values_df.columns[1:]]
        )

        result = assert_no_crash(if_else, cond, then_df, else_df)
        assert_valid_result(result, cond.shape)


# =============================================================================
# 7. GROUP OPERATORS TESTS
# =============================================================================


class TestGroupOperators:
    """Test group operators with group mask."""

    def test_group_rank(
        self,
        group_values_df: pl.DataFrame,
        group_mask_df: pl.DataFrame,
    ) -> None:
        """Test group_rank with standard inputs."""
        result = assert_no_crash(group_rank, group_values_df, group_mask_df)
        assert_valid_result(result, group_values_df.shape)

        # Ranks should be in [0, 1]
        for col in group_values_df.columns[1:]:
            values = result[col].to_numpy()
            valid = values[~np.isnan(values)]
            assert np.all(valid >= 0.0) and np.all(valid <= 1.0)

    def test_group_zscore(
        self,
        group_values_df: pl.DataFrame,
        group_mask_df: pl.DataFrame,
    ) -> None:
        """Test group_zscore with standard inputs."""
        result = assert_no_crash(group_zscore, group_values_df, group_mask_df)
        assert_valid_result(result, group_values_df.shape)

    def test_group_scale(
        self,
        group_values_df: pl.DataFrame,
        group_mask_df: pl.DataFrame,
    ) -> None:
        """Test group_scale with standard inputs."""
        result = assert_no_crash(group_scale, group_values_df, group_mask_df)
        assert_valid_result(result, group_values_df.shape)

        # Scaled values should be in [0, 1]
        for col in group_values_df.columns[1:]:
            values = result[col].to_numpy()
            valid = values[~np.isnan(values)]
            assert np.all(valid >= 0.0) and np.all(valid <= 1.0)

    def test_group_neutralize(
        self,
        group_values_df: pl.DataFrame,
        group_mask_df: pl.DataFrame,
    ) -> None:
        """Test group_neutralize with standard inputs."""
        result = assert_no_crash(group_neutralize, group_values_df, group_mask_df)
        assert_valid_result(result, group_values_df.shape)

    def test_group_mean(
        self,
        group_values_df: pl.DataFrame,
        group_weights_df: pl.DataFrame,
        group_mask_df: pl.DataFrame,
    ) -> None:
        """Test group_mean with weights."""
        result = assert_no_crash(
            group_mean, group_values_df, group_weights_df, group_mask_df
        )
        assert_valid_result(result, group_values_df.shape)

    def test_group_backfill(
        self,
        high_null_density_df: pl.DataFrame,
    ) -> None:
        """Test group_backfill with high null density."""
        # Create group mask matching the sparse data
        dates = high_null_density_df["Date"]
        group_mask = pl.DataFrame({
            "Date": dates,
            "SPARSE_A": ["G1"] * len(dates),
            "SPARSE_B": ["G1"] * len(dates),
            "SPARSE_C": ["G2"] * len(dates),
        })

        result = assert_no_crash(group_backfill, high_null_density_df, group_mask, 5)
        assert_valid_result(result, high_null_density_df.shape)

    def test_group_ops_extreme_values(self, extreme_values_df: pl.DataFrame) -> None:
        """Test group operators with extreme values."""
        dates = extreme_values_df["Date"]
        group_mask = pl.DataFrame({
            "Date": dates,
            **{c: ["G1"] * len(dates) for c in extreme_values_df.columns[1:]}
        })

        result_rank = assert_no_crash(group_rank, extreme_values_df, group_mask)
        result_zscore = assert_no_crash(group_zscore, extreme_values_df, group_mask)
        result_scale = assert_no_crash(group_scale, extreme_values_df, group_mask)
        result_neutralize = assert_no_crash(group_neutralize, extreme_values_df, group_mask)

        assert_valid_result(result_rank, extreme_values_df.shape)
        assert_valid_result(result_zscore, extreme_values_df.shape)
        assert_valid_result(result_scale, extreme_values_df.shape)
        assert_valid_result(result_neutralize, extreme_values_df.shape)


# =============================================================================
# 8. ADDITIONAL TIME-SERIES OPERATORS TESTS
# =============================================================================


class TestAdditionalTimeSeriesOperators:
    """Test additional time-series operators not covered in other tests."""

    def test_ts_quantile_gaussian(self, extreme_values_df: pl.DataFrame) -> None:
        """Test ts_quantile with gaussian driver."""
        result = assert_no_crash(ts_quantile, extreme_values_df, 5, "gaussian")
        assert_valid_result(result, extreme_values_df.shape)

    def test_ts_quantile_uniform(self, extreme_values_df: pl.DataFrame) -> None:
        """Test ts_quantile with uniform driver."""
        result = assert_no_crash(ts_quantile, extreme_values_df, 5, "uniform")
        assert_valid_result(result, extreme_values_df.shape)

    def test_ts_scale(self, extreme_values_df: pl.DataFrame) -> None:
        """Test ts_scale with extreme values."""
        result = assert_no_crash(ts_scale, extreme_values_df, 5)
        assert_valid_result(result, extreme_values_df.shape)

    def test_ts_scale_with_constant(self, short_df: pl.DataFrame) -> None:
        """Test ts_scale with constant offset."""
        result = assert_no_crash(ts_scale, short_df, 3, constant=0.5)
        assert_valid_result(result, short_df.shape)

    def test_ts_step(self, extreme_values_df: pl.DataFrame) -> None:
        """Test ts_step produces row counter."""
        result = assert_no_crash(ts_step, extreme_values_df)
        assert_valid_result(result, extreme_values_df.shape)

        # Should produce 1, 2, 3, ..., n
        for col in extreme_values_df.columns[1:]:
            values = result[col].to_list()
            expected = list(range(1, len(extreme_values_df) + 1))
            assert values == expected

    def test_ts_av_diff(self, extreme_values_df: pl.DataFrame) -> None:
        """Test ts_av_diff with extreme values."""
        result = assert_no_crash(ts_av_diff, extreme_values_df, 3)
        assert_valid_result(result, extreme_values_df.shape)

    def test_ts_backfill(self, high_null_density_df: pl.DataFrame) -> None:
        """Test ts_backfill with high null density."""
        result = assert_no_crash(ts_backfill, high_null_density_df, 5)
        assert_valid_result(result, high_null_density_df.shape)

    def test_ts_count_nans(self, high_null_density_df: pl.DataFrame) -> None:
        """Test ts_count_nans with high null density."""
        result = assert_no_crash(ts_count_nans, high_null_density_df, 10)
        assert_valid_result(result, high_null_density_df.shape)

        # Count should be integer values
        for col in high_null_density_df.columns[1:]:
            values = result[col].to_numpy()
            valid = values[~np.isnan(values)]
            assert np.all(valid >= 0)

    def test_hump(self, extreme_values_df: pl.DataFrame) -> None:
        """Test hump with extreme values."""
        result = assert_no_crash(hump, extreme_values_df, 0.01)
        assert_valid_result(result, extreme_values_df.shape)

    def test_hump_high_factor(self, short_df: pl.DataFrame) -> None:
        """Test hump with high hump factor."""
        result = assert_no_crash(hump, short_df, 1.0)
        assert_valid_result(result, short_df.shape)

    def test_kth_element(self, extreme_values_df: pl.DataFrame) -> None:
        """Test kth_element retrieves correct lookback."""
        result_k0 = assert_no_crash(kth_element, extreme_values_df, 5, 0)
        result_k1 = assert_no_crash(kth_element, extreme_values_df, 5, 1)
        result_k3 = assert_no_crash(kth_element, extreme_values_df, 5, 3)

        assert_valid_result(result_k0, extreme_values_df.shape)
        assert_valid_result(result_k1, extreme_values_df.shape)
        assert_valid_result(result_k3, extreme_values_df.shape)

    def test_last_diff_value(self, constant_columns_df: pl.DataFrame) -> None:
        """Test last_diff_value with constant columns (no changes)."""
        result = assert_no_crash(last_diff_value, constant_columns_df, 5)
        assert_valid_result(result, constant_columns_df.shape)

    def test_last_diff_value_varying(self, short_df: pl.DataFrame) -> None:
        """Test last_diff_value with varying values."""
        result = assert_no_crash(last_diff_value, short_df, 3)
        assert_valid_result(result, short_df.shape)

    def test_days_from_last_change(self, constant_columns_df: pl.DataFrame) -> None:
        """Test days_from_last_change with constant columns."""
        result = assert_no_crash(days_from_last_change, constant_columns_df)
        assert_valid_result(result, constant_columns_df.shape)

    def test_days_from_last_change_varying(self, short_df: pl.DataFrame) -> None:
        """Test days_from_last_change with varying values."""
        result = assert_no_crash(days_from_last_change, short_df)
        assert_valid_result(result, short_df.shape)


# =============================================================================
# 9. ADDITIONAL ARITHMETIC OPERATORS TESTS
# =============================================================================


class TestAdditionalArithmeticOperators:
    """Test additional arithmetic operators not covered in other tests."""

    def test_min_max_element_wise(
        self, extreme_values_pair: tuple[pl.DataFrame, pl.DataFrame]
    ) -> None:
        """Test element-wise min/max with two DataFrames."""
        df1, df2 = extreme_values_pair

        result_max = assert_no_crash(op_max, df1, df2)
        result_min = assert_no_crash(op_min, df1, df2)

        assert_valid_result(result_max, df1.shape)
        assert_valid_result(result_min, df1.shape)

    def test_reverse(self, extreme_values_df: pl.DataFrame) -> None:
        """Test reverse (negation) with extreme values."""
        result = assert_no_crash(reverse, extreme_values_df)
        assert_valid_result(result, extreme_values_df.shape)

        # Check negation is correct
        for col in extreme_values_df.columns[1:]:
            orig = extreme_values_df[col].to_numpy()
            neg = result[col].to_numpy()
            valid_mask = ~np.isnan(orig) & ~np.isnan(neg)
            np.testing.assert_allclose(neg[valid_mask], -orig[valid_mask])

    def test_densify(self, short_df: pl.DataFrame) -> None:
        """Test densify remaps to consecutive integers per row."""
        result = assert_no_crash(densify, short_df)
        assert_valid_result(result, short_df.shape)

        # Densify maps row-wise: each row's values become 0, 1, 2, ... n-1
        # For short_df with 3 columns (A, B, C), each row has 3 unique values
        # so each row should have values from {0, 1, 2}
        for i in range(len(result)):
            row_vals = [result[col][i] for col in short_df.columns[1:]]
            valid = [v for v in row_vals if not np.isnan(v)]
            unique_row_vals = set(valid)
            # All row values should be integers from 0 to n-1
            assert unique_row_vals.issubset({0.0, 1.0, 2.0})

    def test_densify_with_nulls(self, high_null_density_df: pl.DataFrame) -> None:
        """Test densify with high null density."""
        result = assert_no_crash(densify, high_null_density_df)
        assert_valid_result(result, high_null_density_df.shape)


# =============================================================================
# 10. VECTOR OPERATORS TESTS
# =============================================================================


class TestVectorOperators:
    """Test vector operators for list columns."""

    def test_vec_avg(self, list_df: pl.DataFrame) -> None:
        """Test vec_avg computes mean of lists."""
        result = assert_no_crash(vec_avg, list_df)
        assert_valid_result(result, list_df.shape)

    def test_vec_sum(self, list_df: pl.DataFrame) -> None:
        """Test vec_sum computes sum of lists."""
        result = assert_no_crash(vec_sum, list_df)
        assert_valid_result(result, list_df.shape)

    def test_vec_ops_empty_lists(self) -> None:
        """Test vector operators with empty lists."""
        dates = pl.date_range(date(2024, 1, 1), date(2024, 1, 3), eager=True)
        df = pl.DataFrame({
            "Date": dates,
            "A": [[], [], []],
            "B": [[], [1.0], []],
        })

        result_avg = assert_no_crash(vec_avg, df)
        result_sum = assert_no_crash(vec_sum, df)

        assert_valid_result(result_avg, df.shape)
        assert_valid_result(result_sum, df.shape)


# =============================================================================
# 11. TRANSFORMATIONAL OPERATORS TESTS
# =============================================================================


class TestTransformationalOperators:
    """Test transformational operators."""

    def test_trade_when_basic(self) -> None:
        """Test trade_when with basic entry/exit signals."""
        dates = pl.date_range(date(2024, 1, 1), date(2024, 1, 10), eager=True)

        trigger_trade = pl.DataFrame({
            "Date": dates,
            "A": [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            "B": [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        })

        alpha = pl.DataFrame({
            "Date": dates,
            "A": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0],
            "B": [200.0, 201.0, 202.0, 203.0, 204.0, 205.0, 206.0, 207.0, 208.0, 209.0],
        })

        trigger_exit = pl.DataFrame({
            "Date": dates,
            "A": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            "B": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        })

        result = assert_no_crash(trade_when, trigger_trade, alpha, trigger_exit)
        assert_valid_result(result, trigger_trade.shape)

    def test_trade_when_scalar_exit(self) -> None:
        """Test trade_when with scalar exit (never exit)."""
        dates = pl.date_range(date(2024, 1, 1), date(2024, 1, 5), eager=True)

        trigger_trade = pl.DataFrame({
            "Date": dates,
            "A": [1.0, 0.0, 1.0, 0.0, 0.0],
        })

        alpha = pl.DataFrame({
            "Date": dates,
            "A": [100.0, 101.0, 102.0, 103.0, 104.0],
        })

        # Scalar exit = -1 means never exit
        result = assert_no_crash(trade_when, trigger_trade, alpha, -1)
        assert_valid_result(result, trigger_trade.shape)

    def test_trade_when_extreme_values(self, extreme_values_df: pl.DataFrame) -> None:
        """Test trade_when with extreme alpha values."""
        dates = extreme_values_df["Date"]
        trigger_trade = pl.DataFrame({
            "Date": dates,
            **{c: [1.0] + [0.0] * 9 for c in extreme_values_df.columns[1:]}
        })

        result = assert_no_crash(trade_when, trigger_trade, extreme_values_df, -1)
        assert_valid_result(result, extreme_values_df.shape)
