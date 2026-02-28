"""Smoke tests for all operators - quick CI sanity checks.

Tests all 68 operators with small data (500 rows x 100 columns) to verify:
- Operators run without errors
- Output shape matches input (where applicable)
- Basic type correctness

Target: < 30s total for CI.
"""

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from alphalab.api import operators as ops


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture(scope="module")
def smoke_df() -> pl.DataFrame:
    """Create small DataFrame for smoke testing: 500 rows x 100 columns."""
    np.random.seed(42)
    n_rows = 500
    n_cols = 100

    start_date = date(2022, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_rows)]

    data = {"Date": dates}
    for i in range(n_cols):
        # Mix of positive values with some NaNs
        col_data = np.random.randn(n_rows) * 10 + 100
        # Add ~5% NaNs
        nan_mask = np.random.random(n_rows) < 0.05
        col_data[nan_mask] = np.nan
        data[f"S{i:03d}"] = col_data

    return pl.DataFrame(data)


@pytest.fixture(scope="module")
def smoke_df_pair(smoke_df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Create pair of DataFrames for two-variable operators."""
    np.random.seed(43)
    n_rows = smoke_df.height
    n_cols = smoke_df.width - 1  # Exclude Date

    start_date = date(2022, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_rows)]

    data = {"Date": dates}
    for i in range(n_cols):
        col_data = np.random.randn(n_rows) * 10 + 100
        nan_mask = np.random.random(n_rows) < 0.05
        col_data[nan_mask] = np.nan
        data[f"S{i:03d}"] = col_data

    return smoke_df, pl.DataFrame(data)


@pytest.fixture(scope="module")
def smoke_group_mask(smoke_df: pl.DataFrame) -> pl.DataFrame:
    """Create group mask for group operators (string groups)."""
    np.random.seed(44)
    n_rows = smoke_df.height
    n_cols = smoke_df.width - 1

    start_date = date(2022, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_rows)]

    # String group membership (e.g., sector names)
    groups = ["GroupA", "GroupB", "GroupC"]
    data = {"Date": dates}
    for i in range(n_cols):
        data[f"S{i:03d}"] = np.random.choice(groups, n_rows)

    return pl.DataFrame(data)


@pytest.fixture(scope="module")
def smoke_bool_df(smoke_df: pl.DataFrame) -> pl.DataFrame:
    """Create boolean DataFrame for logical operators."""
    np.random.seed(45)
    n_rows = smoke_df.height
    n_cols = smoke_df.width - 1

    start_date = date(2022, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_rows)]

    data = {"Date": dates}
    for i in range(n_cols):
        # True boolean values
        data[f"S{i:03d}"] = np.random.choice([True, False], n_rows)

    return pl.DataFrame(data)


@pytest.fixture(scope="module")
def smoke_weight_df(smoke_df: pl.DataFrame) -> pl.DataFrame:
    """Create weight DataFrame for group_mean."""
    np.random.seed(46)
    n_rows = smoke_df.height
    n_cols = smoke_df.width - 1

    start_date = date(2022, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_rows)]

    data = {"Date": dates}
    for i in range(n_cols):
        # Positive weights
        data[f"S{i:03d}"] = np.abs(np.random.randn(n_rows)) + 0.1

    return pl.DataFrame(data)


@pytest.fixture(scope="module")
def smoke_list_df() -> pl.DataFrame:
    """Create DataFrame with list columns for vector operators."""
    np.random.seed(47)
    n_rows = 500

    start_date = date(2022, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_rows)]

    # Create list columns
    data = {"Date": dates}
    for i in range(10):  # Fewer columns for list type
        data[f"S{i:03d}"] = [[np.random.randn() for _ in range(5)] for _ in range(n_rows)]

    return pl.DataFrame(data)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def assert_valid_output(result: pl.DataFrame, original: pl.DataFrame, name: str) -> None:
    """Assert basic output validity."""
    assert isinstance(result, pl.DataFrame), f"{name}: Expected DataFrame"
    assert result.height == original.height, f"{name}: Row count mismatch"
    assert result.width == original.width, f"{name}: Column count mismatch"
    # First column should be Date
    assert result.columns[0] == "Date", f"{name}: First column should be Date"


# =============================================================================
# ARITHMETIC OPERATORS (15)
# =============================================================================


class TestArithmeticSmoke:
    """Smoke tests for arithmetic operators."""

    def test_abs(self, smoke_df: pl.DataFrame) -> None:
        result = ops.abs(smoke_df)
        assert_valid_output(result, smoke_df, "abs")

    def test_add(self, smoke_df_pair: tuple) -> None:
        df1, df2 = smoke_df_pair
        result = ops.add(df1, df2)
        assert_valid_output(result, df1, "add")

    def test_subtract(self, smoke_df_pair: tuple) -> None:
        df1, df2 = smoke_df_pair
        result = ops.subtract(df1, df2)
        assert_valid_output(result, df1, "subtract")

    def test_multiply(self, smoke_df_pair: tuple) -> None:
        df1, df2 = smoke_df_pair
        result = ops.multiply(df1, df2)
        assert_valid_output(result, df1, "multiply")

    def test_divide(self, smoke_df_pair: tuple) -> None:
        df1, df2 = smoke_df_pair
        result = ops.divide(df1, df2)
        assert_valid_output(result, df1, "divide")

    def test_inverse(self, smoke_df: pl.DataFrame) -> None:
        result = ops.inverse(smoke_df)
        assert_valid_output(result, smoke_df, "inverse")

    def test_log(self, smoke_df: pl.DataFrame) -> None:
        # Use abs to avoid log of negative
        positive_df = ops.abs(smoke_df)
        result = ops.log(positive_df)
        assert_valid_output(result, smoke_df, "log")

    def test_max(self, smoke_df_pair: tuple) -> None:
        df1, df2 = smoke_df_pair
        result = ops.max(df1, df2)
        assert_valid_output(result, df1, "max")

    def test_min(self, smoke_df_pair: tuple) -> None:
        df1, df2 = smoke_df_pair
        result = ops.min(df1, df2)
        assert_valid_output(result, df1, "min")

    def test_power(self, smoke_df: pl.DataFrame) -> None:
        result = ops.power(smoke_df, 2)
        assert_valid_output(result, smoke_df, "power")

    def test_signed_power(self, smoke_df: pl.DataFrame) -> None:
        result = ops.signed_power(smoke_df, 2)
        assert_valid_output(result, smoke_df, "signed_power")

    def test_sqrt(self, smoke_df: pl.DataFrame) -> None:
        # Use abs to avoid sqrt of negative
        positive_df = ops.abs(smoke_df)
        result = ops.sqrt(positive_df)
        assert_valid_output(result, smoke_df, "sqrt")

    def test_sign(self, smoke_df: pl.DataFrame) -> None:
        result = ops.sign(smoke_df)
        assert_valid_output(result, smoke_df, "sign")

    def test_reverse(self, smoke_df: pl.DataFrame) -> None:
        result = ops.reverse(smoke_df)
        assert_valid_output(result, smoke_df, "reverse")

    def test_densify(self, smoke_df: pl.DataFrame) -> None:
        result = ops.densify(smoke_df)
        assert_valid_output(result, smoke_df, "densify")


# =============================================================================
# CROSS-SECTIONAL OPERATORS (7)
# =============================================================================


class TestCrossSectionalSmoke:
    """Smoke tests for cross-sectional operators."""

    def test_bucket(self, smoke_df: pl.DataFrame) -> None:
        # First rank to [0,1], then bucket
        ranked = ops.rank(smoke_df)
        result = ops.bucket(ranked, range_spec="0,1,0.25")
        assert_valid_output(result, smoke_df, "bucket")

    def test_rank(self, smoke_df: pl.DataFrame) -> None:
        result = ops.rank(smoke_df)
        assert_valid_output(result, smoke_df, "rank")

    def test_zscore(self, smoke_df: pl.DataFrame) -> None:
        result = ops.zscore(smoke_df)
        assert_valid_output(result, smoke_df, "zscore")

    def test_normalize(self, smoke_df: pl.DataFrame) -> None:
        result = ops.normalize(smoke_df)
        assert_valid_output(result, smoke_df, "normalize")

    def test_scale(self, smoke_df: pl.DataFrame) -> None:
        result = ops.scale(smoke_df)
        assert_valid_output(result, smoke_df, "scale")

    def test_quantile(self, smoke_df: pl.DataFrame) -> None:
        result = ops.quantile(smoke_df, driver="gaussian", sigma=1.0)
        assert_valid_output(result, smoke_df, "quantile")

    def test_winsorize(self, smoke_df: pl.DataFrame) -> None:
        result = ops.winsorize(smoke_df, std=4.0)
        assert_valid_output(result, smoke_df, "winsorize")


# =============================================================================
# GROUP OPERATORS (6)
# =============================================================================


class TestGroupSmoke:
    """Smoke tests for group operators."""

    def test_group_rank(self, smoke_df: pl.DataFrame, smoke_group_mask: pl.DataFrame) -> None:
        result = ops.group_rank(smoke_df, smoke_group_mask)
        assert_valid_output(result, smoke_df, "group_rank")

    def test_group_zscore(self, smoke_df: pl.DataFrame, smoke_group_mask: pl.DataFrame) -> None:
        result = ops.group_zscore(smoke_df, smoke_group_mask)
        assert_valid_output(result, smoke_df, "group_zscore")

    def test_group_scale(self, smoke_df: pl.DataFrame, smoke_group_mask: pl.DataFrame) -> None:
        result = ops.group_scale(smoke_df, smoke_group_mask)
        assert_valid_output(result, smoke_df, "group_scale")

    def test_group_neutralize(self, smoke_df: pl.DataFrame, smoke_group_mask: pl.DataFrame) -> None:
        result = ops.group_neutralize(smoke_df, smoke_group_mask)
        assert_valid_output(result, smoke_df, "group_neutralize")

    def test_group_mean(
        self,
        smoke_df: pl.DataFrame,
        smoke_weight_df: pl.DataFrame,
        smoke_group_mask: pl.DataFrame,
    ) -> None:
        result = ops.group_mean(smoke_df, smoke_weight_df, smoke_group_mask)
        assert_valid_output(result, smoke_df, "group_mean")

    def test_group_backfill(
        self, smoke_df: pl.DataFrame, smoke_group_mask: pl.DataFrame
    ) -> None:
        result = ops.group_backfill(smoke_df, smoke_group_mask, d=5, std=4.0)
        assert_valid_output(result, smoke_df, "group_backfill")


# =============================================================================
# LOGICAL OPERATORS (11)
# =============================================================================


class TestLogicalSmoke:
    """Smoke tests for logical operators."""

    def test_and(self, smoke_bool_df: pl.DataFrame) -> None:
        result = ops.and_(smoke_bool_df, smoke_bool_df)
        assert_valid_output(result, smoke_bool_df, "and_")

    def test_or(self, smoke_bool_df: pl.DataFrame) -> None:
        result = ops.or_(smoke_bool_df, smoke_bool_df)
        assert_valid_output(result, smoke_bool_df, "or_")

    def test_not(self, smoke_bool_df: pl.DataFrame) -> None:
        result = ops.not_(smoke_bool_df)
        assert_valid_output(result, smoke_bool_df, "not_")

    def test_if_else(self, smoke_bool_df: pl.DataFrame, smoke_df: pl.DataFrame) -> None:
        result = ops.if_else(smoke_bool_df, smoke_df, smoke_df)
        assert_valid_output(result, smoke_df, "if_else")

    def test_is_nan(self, smoke_df: pl.DataFrame) -> None:
        result = ops.is_nan(smoke_df)
        assert_valid_output(result, smoke_df, "is_nan")

    def test_lt(self, smoke_df_pair: tuple) -> None:
        df1, df2 = smoke_df_pair
        result = ops.lt(df1, df2)
        assert_valid_output(result, df1, "lt")

    def test_le(self, smoke_df_pair: tuple) -> None:
        df1, df2 = smoke_df_pair
        result = ops.le(df1, df2)
        assert_valid_output(result, df1, "le")

    def test_gt(self, smoke_df_pair: tuple) -> None:
        df1, df2 = smoke_df_pair
        result = ops.gt(df1, df2)
        assert_valid_output(result, df1, "gt")

    def test_ge(self, smoke_df_pair: tuple) -> None:
        df1, df2 = smoke_df_pair
        result = ops.ge(df1, df2)
        assert_valid_output(result, df1, "ge")

    def test_eq(self, smoke_df_pair: tuple) -> None:
        df1, df2 = smoke_df_pair
        result = ops.eq(df1, df2)
        assert_valid_output(result, df1, "eq")

    def test_ne(self, smoke_df_pair: tuple) -> None:
        df1, df2 = smoke_df_pair
        result = ops.ne(df1, df2)
        assert_valid_output(result, df1, "ne")


# =============================================================================
# TIME-SERIES OPERATORS - BASIC (7)
# =============================================================================


class TestTimeSeriesBasicSmoke:
    """Smoke tests for basic time-series operators."""

    def test_ts_mean(self, smoke_df: pl.DataFrame) -> None:
        result = ops.ts_mean(smoke_df, 10)
        assert_valid_output(result, smoke_df, "ts_mean")

    def test_ts_sum(self, smoke_df: pl.DataFrame) -> None:
        result = ops.ts_sum(smoke_df, 10)
        assert_valid_output(result, smoke_df, "ts_sum")

    def test_ts_std(self, smoke_df: pl.DataFrame) -> None:
        result = ops.ts_std(smoke_df, 10)
        assert_valid_output(result, smoke_df, "ts_std")

    def test_ts_min(self, smoke_df: pl.DataFrame) -> None:
        result = ops.ts_min(smoke_df, 10)
        assert_valid_output(result, smoke_df, "ts_min")

    def test_ts_max(self, smoke_df: pl.DataFrame) -> None:
        result = ops.ts_max(smoke_df, 10)
        assert_valid_output(result, smoke_df, "ts_max")

    def test_ts_delta(self, smoke_df: pl.DataFrame) -> None:
        result = ops.ts_delta(smoke_df, 5)
        assert_valid_output(result, smoke_df, "ts_delta")

    def test_ts_delay(self, smoke_df: pl.DataFrame) -> None:
        result = ops.ts_delay(smoke_df, 5)
        assert_valid_output(result, smoke_df, "ts_delay")


# =============================================================================
# TIME-SERIES OPERATORS - ROLLING (6)
# =============================================================================


class TestTimeSeriesRollingSmoke:
    """Smoke tests for rolling time-series operators."""

    def test_ts_product(self, smoke_df: pl.DataFrame) -> None:
        result = ops.ts_product(smoke_df, 5)
        assert_valid_output(result, smoke_df, "ts_product")

    def test_ts_count_nans(self, smoke_df: pl.DataFrame) -> None:
        result = ops.ts_count_nans(smoke_df, 10)
        assert_valid_output(result, smoke_df, "ts_count_nans")

    def test_ts_zscore(self, smoke_df: pl.DataFrame) -> None:
        result = ops.ts_zscore(smoke_df, 10)
        assert_valid_output(result, smoke_df, "ts_zscore")

    def test_ts_scale(self, smoke_df: pl.DataFrame) -> None:
        result = ops.ts_scale(smoke_df, 10)
        assert_valid_output(result, smoke_df, "ts_scale")

    def test_ts_av_diff(self, smoke_df: pl.DataFrame) -> None:
        result = ops.ts_av_diff(smoke_df, 10)
        assert_valid_output(result, smoke_df, "ts_av_diff")

    def test_ts_step(self, smoke_df: pl.DataFrame) -> None:
        result = ops.ts_step(smoke_df)
        assert_valid_output(result, smoke_df, "ts_step")


# =============================================================================
# TIME-SERIES OPERATORS - ARG (2)
# =============================================================================


class TestTimeSeriesArgSmoke:
    """Smoke tests for arg time-series operators."""

    def test_ts_arg_max(self, smoke_df: pl.DataFrame) -> None:
        result = ops.ts_arg_max(smoke_df, 10)
        assert_valid_output(result, smoke_df, "ts_arg_max")

    def test_ts_arg_min(self, smoke_df: pl.DataFrame) -> None:
        result = ops.ts_arg_min(smoke_df, 10)
        assert_valid_output(result, smoke_df, "ts_arg_min")


# =============================================================================
# TIME-SERIES OPERATORS - LOOKBACK (4)
# =============================================================================


class TestTimeSeriesLookbackSmoke:
    """Smoke tests for lookback time-series operators."""

    def test_ts_backfill(self, smoke_df: pl.DataFrame) -> None:
        result = ops.ts_backfill(smoke_df, 10)
        assert_valid_output(result, smoke_df, "ts_backfill")

    def test_kth_element(self, smoke_df: pl.DataFrame) -> None:
        result = ops.kth_element(smoke_df, d=10, k=5)
        assert_valid_output(result, smoke_df, "kth_element")

    def test_last_diff_value(self, smoke_df: pl.DataFrame) -> None:
        result = ops.last_diff_value(smoke_df, 10)
        assert_valid_output(result, smoke_df, "last_diff_value")

    def test_days_from_last_change(self, smoke_df: pl.DataFrame) -> None:
        result = ops.days_from_last_change(smoke_df)
        assert_valid_output(result, smoke_df, "days_from_last_change")


# =============================================================================
# TIME-SERIES OPERATORS - STATEFUL (3)
# =============================================================================


class TestTimeSeriesStatefulSmoke:
    """Smoke tests for stateful time-series operators."""

    def test_hump(self, smoke_df: pl.DataFrame) -> None:
        result = ops.hump(smoke_df, 0.5)
        assert_valid_output(result, smoke_df, "hump")

    def test_ts_decay_linear(self, smoke_df: pl.DataFrame) -> None:
        result = ops.ts_decay_linear(smoke_df, 10)
        assert_valid_output(result, smoke_df, "ts_decay_linear")

    def test_ts_rank(self, smoke_df: pl.DataFrame) -> None:
        result = ops.ts_rank(smoke_df, 10)
        assert_valid_output(result, smoke_df, "ts_rank")


# =============================================================================
# TIME-SERIES OPERATORS - TWO-VARIABLE (4)
# =============================================================================


class TestTimeSeriesTwoVariableSmoke:
    """Smoke tests for two-variable time-series operators."""

    def test_ts_corr(self, smoke_df_pair: tuple) -> None:
        df1, df2 = smoke_df_pair
        result = ops.ts_corr(df1, df2, 10)
        assert_valid_output(result, df1, "ts_corr")

    def test_ts_covariance(self, smoke_df_pair: tuple) -> None:
        df1, df2 = smoke_df_pair
        result = ops.ts_covariance(df1, df2, 10)
        assert_valid_output(result, df1, "ts_covariance")

    def test_ts_quantile(self, smoke_df: pl.DataFrame) -> None:
        # ts_quantile is a single-variable operator with rolling window
        result = ops.ts_quantile(smoke_df, 10, driver="gaussian")
        assert_valid_output(result, smoke_df, "ts_quantile")

    def test_ts_regression(self, smoke_df_pair: tuple) -> None:
        df1, df2 = smoke_df_pair
        # ts_regression returns single DataFrame based on rettype
        result = ops.ts_regression(df1, df2, 10, rettype="resid")
        assert_valid_output(result, df1, "ts_regression")


# =============================================================================
# TRANSFORMATIONAL OPERATORS (1)
# =============================================================================


class TestTransformationalSmoke:
    """Smoke tests for transformational operators."""

    def test_trade_when(self, smoke_df: pl.DataFrame, smoke_bool_df: pl.DataFrame) -> None:
        # trade_when(trigger_trade, alpha, trigger_exit)
        # Use bool_df > 0.5 as trigger, -1 means never exit
        trigger_trade = ops.gt(smoke_df, 100)  # Values > 100 trigger entry
        result = ops.trade_when(trigger_trade, smoke_df, -1)
        assert_valid_output(result, smoke_df, "trade_when")


# =============================================================================
# VECTOR OPERATORS (2)
# =============================================================================


class TestVectorSmoke:
    """Smoke tests for vector operators."""

    def test_vec_avg(self, smoke_list_df: pl.DataFrame) -> None:
        result = ops.vec_avg(smoke_list_df)
        assert_valid_output(result, smoke_list_df, "vec_avg")

    def test_vec_sum(self, smoke_list_df: pl.DataFrame) -> None:
        result = ops.vec_sum(smoke_list_df)
        assert_valid_output(result, smoke_list_df, "vec_sum")


# =============================================================================
# OPERATOR COUNT VERIFICATION
# =============================================================================


class TestOperatorCoverage:
    """Verify all operators are covered."""

    def test_all_operators_covered(self) -> None:
        """Verify we test all 68 operators in __all__."""
        from alphalab.api.operators import __all__ as all_operators

        # Count expected: 68 operators
        assert len(all_operators) == 68, f"Expected 68 operators, got {len(all_operators)}"

        # Map test class to operators
        tested_operators = {
            # Arithmetic (15)
            "abs", "add", "subtract", "multiply", "divide", "inverse", "log",
            "max", "min", "power", "signed_power", "sqrt", "sign", "reverse", "densify",
            # Cross-sectional (7)
            "bucket", "rank", "zscore", "normalize", "scale", "quantile", "winsorize",
            # Group (6)
            "group_rank", "group_zscore", "group_scale", "group_neutralize",
            "group_mean", "group_backfill",
            # Logical (11)
            "and_", "or_", "not_", "if_else", "is_nan", "lt", "le", "gt", "ge", "eq", "ne",
            # Time-series basic (7)
            "ts_mean", "ts_sum", "ts_std", "ts_min", "ts_max", "ts_delta", "ts_delay",
            # Time-series rolling (6)
            "ts_product", "ts_count_nans", "ts_zscore", "ts_scale", "ts_av_diff", "ts_step",
            # Time-series arg (2)
            "ts_arg_max", "ts_arg_min",
            # Time-series lookback (4)
            "ts_backfill", "kth_element", "last_diff_value", "days_from_last_change",
            # Time-series stateful (3)
            "hump", "ts_decay_linear", "ts_rank",
            # Time-series two-variable (4)
            "ts_corr", "ts_covariance", "ts_quantile", "ts_regression",
            # Transformational (1)
            "trade_when",
            # Vector (2)
            "vec_avg", "vec_sum",
        }

        # Check coverage
        all_set = set(all_operators)
        missing = all_set - tested_operators
        extra = tested_operators - all_set

        assert not missing, f"Operators not tested: {missing}"
        assert not extra, f"Tests for non-existent operators: {extra}"
