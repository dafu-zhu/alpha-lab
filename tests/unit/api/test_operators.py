"""Tests for alpha operators."""

import math
from datetime import date

import polars as pl
import pytest

from quantdl.api.operators import (
    abs as op_abs,
    bucket,
)
from quantdl.api.operators import (
    add,
    and_,
    days_from_last_change,
    densify,
    divide,
    eq,
    ge,
    group_backfill,
    group_mean,
    group_neutralize,
    group_rank,
    group_scale,
    group_zscore,
    gt,
    hump,
    if_else,
    inverse,
    is_nan,
    kth_element,
    last_diff_value,
    le,
    log,
    lt,
    multiply,
    ne,
    normalize,
    not_,
    or_,
    power,
    quantile,
    rank,
    reverse,
    scale,
    sign,
    signed_power,
    sqrt,
    subtract,
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
    vec_avg,
    vec_sum,
    winsorize,
    zscore,
)
from quantdl.api.operators import (
    max as op_max,
)
from quantdl.api.operators import (
    min as op_min,
)


@pytest.fixture
def wide_df() -> pl.DataFrame:
    """Create sample wide DataFrame."""
    return pl.DataFrame({
        "timestamp": pl.date_range(date(2024, 1, 1), date(2024, 1, 10), eager=True),
        "AAPL": [100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0, 108.0, 107.0, 110.0],
        "MSFT": [200.0, 202.0, 201.0, 203.0, 205.0, 204.0, 206.0, 208.0, 207.0, 210.0],
        "GOOGL": [150.0, 152.0, 151.0, 153.0, 155.0, 154.0, 156.0, 158.0, 157.0, 160.0],
    })


# =============================================================================
# TIME-SERIES OPERATORS
# =============================================================================


class TestTimeSeriesOperators:
    """Time-series operator tests."""

    def test_ts_mean(self, wide_df: pl.DataFrame) -> None:
        """Test rolling mean with partial windows."""
        result = ts_mean(wide_df, 3)

        assert result.columns == wide_df.columns
        assert len(result) == len(wide_df)

        # Partial windows allowed: row 0 has mean of [100], row 1 has mean of [100, 102]
        assert result["AAPL"][0] == 100.0
        assert abs(result["AAPL"][1] - 101.0) < 0.01  # (100 + 102) / 2

        # Third value should be mean of first 3
        expected = (100.0 + 102.0 + 101.0) / 3
        assert abs(result["AAPL"][2] - expected) < 0.01

    def test_ts_sum(self, wide_df: pl.DataFrame) -> None:
        """Test rolling sum."""
        result = ts_sum(wide_df, 3)

        expected = 100.0 + 102.0 + 101.0
        assert abs(result["AAPL"][2] - expected) < 0.01

    def test_ts_std(self, wide_df: pl.DataFrame) -> None:
        """Test rolling standard deviation."""
        result = ts_std(wide_df, 3)

        assert result.columns == wide_df.columns
        # Std should be positive
        assert result["AAPL"][2] is not None
        assert result["AAPL"][2] > 0

    def test_ts_min(self, wide_df: pl.DataFrame) -> None:
        """Test rolling minimum."""
        result = ts_min(wide_df, 3)

        # Min of 100, 102, 101 is 100
        assert result["AAPL"][2] == 100.0

    def test_ts_max(self, wide_df: pl.DataFrame) -> None:
        """Test rolling maximum."""
        result = ts_max(wide_df, 3)

        # Max of 100, 102, 101 is 102
        assert result["AAPL"][2] == 102.0

    def test_ts_delta(self, wide_df: pl.DataFrame) -> None:
        """Test difference."""
        result = ts_delta(wide_df, 1)

        # Second value - first value = 102 - 100 = 2
        assert result["AAPL"][1] == 2.0

    def test_ts_delay(self, wide_df: pl.DataFrame) -> None:
        """Test lag."""
        result = ts_delay(wide_df, 1)

        # First value should be null
        assert result["AAPL"][0] is None
        # Second value should be first original value
        assert result["AAPL"][1] == 100.0

    def test_ts_product(self, wide_df: pl.DataFrame) -> None:
        """Test rolling product."""
        result = ts_product(wide_df, 3)
        assert result.columns == wide_df.columns
        # Product of 100, 102, 101
        expected = 100.0 * 102.0 * 101.0
        assert abs(result["AAPL"][2] - expected) < 0.01

    def test_ts_count_nans(self) -> None:
        """Test counting nulls in window."""
        df = pl.DataFrame({
            "timestamp": pl.date_range(date(2024, 1, 1), date(2024, 1, 5), eager=True),
            "AAPL": [100.0, None, 101.0, None, 105.0],
        })
        result = ts_count_nans(df, 3)
        # At idx 2: window [100, None, 101] has 1 null
        assert result["AAPL"][2] == 1
        # At idx 3: window [None, 101, None] has 2 nulls
        assert result["AAPL"][3] == 2

    def test_ts_zscore(self, wide_df: pl.DataFrame) -> None:
        """Test rolling z-score."""
        result = ts_zscore(wide_df, 3)
        assert result.columns == wide_df.columns
        # Z-score exists for idx >= 2
        assert result["AAPL"][2] is not None
        # Z-score should be finite
        assert not math.isnan(result["AAPL"][2])

    def test_ts_scale(self, wide_df: pl.DataFrame) -> None:
        """Test rolling min-max scale."""
        result = ts_scale(wide_df, 3)
        assert result.columns == wide_df.columns
        # Values should be in [0, 1] range
        for i in range(2, len(result)):
            val = result["AAPL"][i]
            if val is not None:
                assert 0.0 <= val <= 1.0

    def test_ts_av_diff(self, wide_df: pl.DataFrame) -> None:
        """Test difference from rolling mean."""
        result = ts_av_diff(wide_df, 3)
        # At idx 2: value=101, mean=(100+102+101)/3=101, diff=0
        assert abs(result["AAPL"][2]) < 0.01

    def test_ts_step(self, wide_df: pl.DataFrame) -> None:
        """Test row counter."""
        result = ts_step(wide_df)
        assert result["AAPL"][0] == 1
        assert result["AAPL"][4] == 5
        assert result["AAPL"][9] == 10

    def test_ts_arg_max(self, wide_df: pl.DataFrame) -> None:
        """Test days since max in window."""
        result = ts_arg_max(wide_df, 3)
        # At idx 2: window [100, 102, 101], max is 102 at window idx 1
        # Days since: (3-1) - 1 = 1 (max was 1 day ago)
        assert result["AAPL"][2] == 1.0

    def test_ts_arg_min(self, wide_df: pl.DataFrame) -> None:
        """Test days since min in window."""
        result = ts_arg_min(wide_df, 3)
        # At idx 2: window [100, 102, 101], min is 100 at window idx 0
        # Days since: (3-1) - 0 = 2 (min was 2 days ago)
        assert result["AAPL"][2] == 2.0

    def test_ts_backfill(self) -> None:
        """Test forward fill with limit."""
        df = pl.DataFrame({
            "timestamp": pl.date_range(date(2024, 1, 1), date(2024, 1, 5), eager=True),
            "AAPL": [100.0, None, None, None, 105.0],
        })
        result = ts_backfill(df, 2)
        # Should fill first 2 nulls
        assert result["AAPL"][1] == 100.0
        assert result["AAPL"][2] == 100.0
        # Third null exceeds limit, stays null
        assert result["AAPL"][3] is None

    def test_kth_element(self, wide_df: pl.DataFrame) -> None:
        """Test k-th element lookback."""
        result = kth_element(wide_df, 5, 2)
        # k=2 means 2 periods ago
        assert result["AAPL"][2] == 100.0
        assert result["AAPL"][3] == 102.0

    def test_last_diff_value(self) -> None:
        """Test finding last different value."""
        df = pl.DataFrame({
            "timestamp": pl.date_range(date(2024, 1, 1), date(2024, 1, 5), eager=True),
            "AAPL": [100.0, 100.0, 102.0, 102.0, 102.0],
        })
        result = last_diff_value(df, 3)
        # At idx 3: window [100, 102, 102], current=102, last diff=100
        assert result["AAPL"][3] == 100.0

    def test_days_from_last_change(self) -> None:
        """Test days since value changed."""
        df = pl.DataFrame({
            "timestamp": pl.date_range(date(2024, 1, 1), date(2024, 1, 5), eager=True),
            "AAPL": [100.0, 100.0, 102.0, 102.0, 102.0],
        })
        result = days_from_last_change(df)
        assert result["AAPL"][0] == 0  # First row
        assert result["AAPL"][1] == 1  # Same as prev
        assert result["AAPL"][2] == 0  # Changed
        assert result["AAPL"][3] == 1  # Same as prev
        assert result["AAPL"][4] == 2  # 2 days since change

    def test_hump(self) -> None:
        """Test hump limiting change magnitude."""
        df = pl.DataFrame({
            "timestamp": pl.date_range(date(2024, 1, 1), date(2024, 1, 3), eager=True),
            "A": [100.0, 200.0, 150.0],
            "B": [50.0, 50.0, 50.0],
        })
        result = hump(df, hump=0.1)
        # Row 1: sum(|values|) = 200+50=250, limit=25
        # A change = 100, capped at prev + 25 = 125
        assert result["A"][1] == 125.0

    def test_ts_decay_linear(self, wide_df: pl.DataFrame) -> None:
        """Test linear decay weighted average."""
        result = ts_decay_linear(wide_df, 3)
        # Weights [1, 2, 3], sum=6
        # At idx 2: (100*1 + 102*2 + 101*3) / 6
        expected = (100 * 1 + 102 * 2 + 101 * 3) / 6
        assert abs(result["AAPL"][2] - expected) < 0.01

    def test_ts_rank(self, wide_df: pl.DataFrame) -> None:
        """Test rank of current value in window."""
        result = ts_rank(wide_df, 3)
        # At idx 2: window [100, 102, 101], current=101
        # Sorted: [100, 101, 102], idx=1, rank=1/2=0.5
        assert abs(result["AAPL"][2] - 0.5) < 0.01

    def test_ts_corr(self, wide_df: pl.DataFrame) -> None:
        """Test rolling correlation."""
        # Correlate with itself should give 1.0
        result = ts_corr(wide_df, wide_df, 3)
        assert abs(result["AAPL"][2] - 1.0) < 0.01

    def test_ts_covariance(self, wide_df: pl.DataFrame) -> None:
        """Test rolling covariance."""
        result = ts_covariance(wide_df, wide_df, 3)
        # Cov with self = variance
        assert result["AAPL"][2] is not None
        assert result["AAPL"][2] > 0

    def test_ts_quantile_gaussian(self, wide_df: pl.DataFrame) -> None:
        """Test rolling quantile with gaussian transform."""
        result = ts_quantile(wide_df, 3, driver="gaussian")
        assert result.columns == wide_df.columns
        # Should produce finite values
        for i in range(2, len(result)):
            val = result["AAPL"][i]
            if val is not None:
                assert not math.isinf(val)

    def test_ts_regression_residual(self, wide_df: pl.DataFrame) -> None:
        """Test regression residual (rettype=0)."""
        result = ts_regression(wide_df, wide_df, 3, rettype=0)
        # Regressing on itself gives perfect fit, residual=0
        for i in range(2, len(result)):
            val = result["AAPL"][i]
            if val is not None:
                assert abs(val) < 0.01

    def test_ts_regression_beta(self, wide_df: pl.DataFrame) -> None:
        """Test regression beta (rettype=1)."""
        result = ts_regression(wide_df, wide_df, 3, rettype=1)
        # Regressing on itself, beta=1
        for i in range(2, len(result)):
            val = result["AAPL"][i]
            if val is not None:
                assert abs(val - 1.0) < 0.01

    def test_ts_regression_rsquared(self, wide_df: pl.DataFrame) -> None:
        """Test regression r-squared (rettype=5)."""
        result = ts_regression(wide_df, wide_df, 3, rettype=5)
        # Regressing on itself, r-squared=1
        for i in range(2, len(result)):
            val = result["AAPL"][i]
            if val is not None:
                assert abs(val - 1.0) < 0.01


# =============================================================================
# CROSS-SECTIONAL OPERATORS
# =============================================================================


class TestCrossSectionalOperators:
    """Cross-sectional operator tests."""

    def test_rank(self, wide_df: pl.DataFrame) -> None:
        """Test cross-sectional rank returns [0,1] floats."""
        result = rank(wide_df, rate=0)  # Precise ranking

        assert result.columns == wide_df.columns

        # At each date, AAPL < GOOGL < MSFT, so ranks should be 0.0, 0.5, 1.0
        # Check first row
        assert result["AAPL"][0] == 0.0  # Smallest
        assert result["GOOGL"][0] == 0.5
        assert result["MSFT"][0] == 1.0  # Largest

    def test_rank_approximate(self, wide_df: pl.DataFrame) -> None:
        """Test approximate ranking with rate>0."""
        result = rank(wide_df, rate=2)  # Bucket-based ranking

        assert result.columns == wide_df.columns
        # Values should be in [0, 1]
        for col in ["AAPL", "MSFT", "GOOGL"]:
            for val in result[col]:
                if val is not None:
                    assert 0.0 <= val <= 1.0

    def test_zscore(self, wide_df: pl.DataFrame) -> None:
        """Test cross-sectional z-score."""
        result = zscore(wide_df)

        assert result.columns == wide_df.columns

        # Z-scores should sum to ~0 for each row
        for i in range(len(result)):
            row_sum = result["AAPL"][i] + result["MSFT"][i] + result["GOOGL"][i]
            assert abs(row_sum) < 0.01

    def test_normalize(self, wide_df: pl.DataFrame) -> None:
        """Test cross-sectional normalize (demean)."""
        result = normalize(wide_df)

        # Normalized values should sum to ~0 for each row
        for i in range(len(result)):
            row_sum = result["AAPL"][i] + result["MSFT"][i] + result["GOOGL"][i]
            assert abs(row_sum) < 0.01

    def test_normalize_with_std(self, wide_df: pl.DataFrame) -> None:
        """Test normalize with std division."""
        result = normalize(wide_df, useStd=True)

        # Should be similar to zscore
        for i in range(len(result)):
            row_sum = result["AAPL"][i] + result["MSFT"][i] + result["GOOGL"][i]
            assert abs(row_sum) < 0.01

    def test_normalize_with_limit(self, wide_df: pl.DataFrame) -> None:
        """Test normalize with clipping."""
        result = normalize(wide_df, useStd=True, limit=0.5)

        # Values should be clipped to [-0.5, 0.5]
        for col in ["AAPL", "MSFT", "GOOGL"]:
            for val in result[col]:
                if val is not None:
                    assert -0.5 <= val <= 0.5

    def test_scale(self, wide_df: pl.DataFrame) -> None:
        """Test scaling to target."""
        result = scale(wide_df, scale=1.0)

        # Sum of absolute values should be ~1.0 for each row
        for i in range(len(result)):
            abs_sum = abs(result["AAPL"][i]) + abs(result["MSFT"][i]) + abs(result["GOOGL"][i])
            assert abs(abs_sum - 1.0) < 0.01

    def test_scale_custom_booksize(self, wide_df: pl.DataFrame) -> None:
        """Test scaling to custom book size."""
        result = scale(wide_df, scale=4.0)

        # Sum of absolute values should be ~4.0 for each row
        for i in range(len(result)):
            abs_sum = abs(result["AAPL"][i]) + abs(result["MSFT"][i]) + abs(result["GOOGL"][i])
            assert abs(abs_sum - 4.0) < 0.01

    def test_scale_longscale_shortscale(self) -> None:
        """Test asymmetric long/short scaling."""
        # Create data with both positive and negative values
        df = pl.DataFrame({
            "timestamp": pl.date_range(date(2024, 1, 1), date(2024, 1, 3), eager=True),
            "A": [10.0, -5.0, 20.0],
            "B": [-20.0, 15.0, -10.0],
            "C": [5.0, -10.0, 30.0],
        })

        result = scale(df, longscale=4.0, shortscale=3.0)

        # Check row 0: longs = [10, 5] = 15, shorts = [|-20|] = 20
        # After scaling: longs sum to 4, shorts sum to 3
        row0_long_sum = max(0, result["A"][0]) + max(0, result["C"][0])
        row0_short_sum = abs(min(0, result["B"][0]))
        assert abs(row0_long_sum - 4.0) < 0.01
        assert abs(row0_short_sum - 3.0) < 0.01

        # Check row 1: longs = [15] = 15, shorts = [|-5|, |-10|] = 15
        row1_long_sum = max(0, result["B"][1])
        row1_short_sum = abs(min(0, result["A"][1])) + abs(min(0, result["C"][1]))
        assert abs(row1_long_sum - 4.0) < 0.01
        assert abs(row1_short_sum - 3.0) < 0.01

    def test_scale_only_longscale(self) -> None:
        """Test scaling only long positions."""
        df = pl.DataFrame({
            "timestamp": pl.date_range(date(2024, 1, 1), date(2024, 1, 2), eager=True),
            "A": [10.0, -5.0],
            "B": [-20.0, 15.0],
            "C": [5.0, -10.0],
        })

        result = scale(df, longscale=2.0, shortscale=0.0)

        # Row 0: longs = [10, 5] = 15, should sum to 2
        row0_long_sum = max(0, result["A"][0]) + max(0, result["C"][0])
        assert abs(row0_long_sum - 2.0) < 0.01
        # Shorts should be 0 (shortscale=0)
        assert result["B"][0] == 0.0

    def test_scale_only_shortscale(self) -> None:
        """Test scaling only short positions."""
        df = pl.DataFrame({
            "timestamp": pl.date_range(date(2024, 1, 1), date(2024, 1, 2), eager=True),
            "A": [10.0, -5.0],
            "B": [-20.0, 15.0],
            "C": [5.0, -10.0],
        })

        result = scale(df, longscale=0.0, shortscale=3.0)

        # Row 0: shorts = [|-20|] = 20, should sum to 3
        row0_short_sum = abs(min(0, result["B"][0]))
        assert abs(row0_short_sum - 3.0) < 0.01
        # Longs should be 0 (longscale=0)
        assert result["A"][0] == 0.0
        assert result["C"][0] == 0.0

    def test_quantile_gaussian(self, wide_df: pl.DataFrame) -> None:
        """Test quantile transformation with gaussian driver."""
        result = quantile(wide_df, driver="gaussian")

        assert result.columns == wide_df.columns
        # Output should be finite for all non-null values
        for col in ["AAPL", "MSFT", "GOOGL"]:
            for val in result[col]:
                if val is not None:
                    assert not math.isnan(val)

    def test_quantile_uniform(self, wide_df: pl.DataFrame) -> None:
        """Test quantile transformation with uniform driver."""
        result = quantile(wide_df, driver="uniform", sigma=2.0)

        assert result.columns == wide_df.columns
        # Uniform output should be in [-sigma, sigma]
        for col in ["AAPL", "MSFT", "GOOGL"]:
            for val in result[col]:
                if val is not None:
                    assert -2.0 <= val <= 2.0

    def test_winsorize(self, wide_df: pl.DataFrame) -> None:
        """Test winsorization."""
        result = winsorize(wide_df, std=1.0)

        assert result.columns == wide_df.columns
        # Winsorized values should not have extreme outliers
        # At least check that values exist
        assert len(result) == len(wide_df)

    def test_winsorize_with_outliers(self) -> None:
        """Test winsorization clips outliers."""
        df = pl.DataFrame({
            "timestamp": pl.date_range(date(2024, 1, 1), date(2024, 1, 1), eager=True),
            "A": [1.0],
            "B": [2.0],
            "C": [100.0],  # Outlier
        })

        result = winsorize(df, std=1.0)
        # The outlier should be clipped
        assert result["C"][0] < 100.0


# =============================================================================
# ARITHMETIC OPERATORS
# =============================================================================


class TestArithmeticOperators:
    """Arithmetic operator tests."""

    @pytest.fixture
    def arith_df(self) -> pl.DataFrame:
        """Create sample wide DataFrame for arithmetic tests."""
        return pl.DataFrame({
            "timestamp": pl.date_range(date(2024, 1, 1), date(2024, 1, 5), eager=True),
            "AAPL": [100.0, -50.0, 25.0, 0.0, -10.0],
            "MSFT": [200.0, 100.0, -50.0, 75.0, 0.0],
            "GOOGL": [-150.0, 0.0, 30.0, -20.0, 40.0],
        })

    @pytest.fixture
    def arith_df2(self) -> pl.DataFrame:
        """Create second sample wide DataFrame for two-input ops."""
        return pl.DataFrame({
            "timestamp": pl.date_range(date(2024, 1, 1), date(2024, 1, 5), eager=True),
            "AAPL": [10.0, 5.0, 5.0, 2.0, -2.0],
            "MSFT": [20.0, 10.0, -10.0, 15.0, 1.0],
            "GOOGL": [-30.0, 1.0, 6.0, -4.0, 8.0],
        })

    def test_abs_basic(self, arith_df: pl.DataFrame) -> None:
        """Test absolute value computation."""
        result = op_abs(arith_df)
        assert result.columns == arith_df.columns
        assert result["AAPL"][0] == 100.0
        assert result["AAPL"][1] == 50.0  # |-50| = 50
        assert result["GOOGL"][0] == 150.0  # |-150| = 150
        assert result["AAPL"][3] == 0.0

    def test_add_two_inputs(self, arith_df: pl.DataFrame, arith_df2: pl.DataFrame) -> None:
        """Test adding two DataFrames."""
        result = add(arith_df, arith_df2)
        assert result.columns == arith_df.columns
        assert result["AAPL"][0] == 110.0  # 100 + 10
        assert result["AAPL"][1] == -45.0  # -50 + 5
        assert result["GOOGL"][0] == -180.0  # -150 + -30

    def test_add_three_inputs(self, arith_df: pl.DataFrame, arith_df2: pl.DataFrame) -> None:
        """Test adding three DataFrames."""
        result = add(arith_df, arith_df2, arith_df)
        assert result["AAPL"][0] == 210.0  # 100 + 10 + 100

    def test_add_filter_null(self) -> None:
        """Test add with filter=True treats null as 0."""
        df1 = pl.DataFrame({"timestamp": [date(2024, 1, 1)], "A": [10.0]})
        df2 = pl.DataFrame({"timestamp": [date(2024, 1, 1)], "A": [None]})
        result = add(df1, df2, filter=True)
        assert result["A"][0] == 10.0  # 10 + 0

    def test_add_without_filter_propagates_null(self) -> None:
        """Test add without filter propagates null."""
        df1 = pl.DataFrame({"timestamp": [date(2024, 1, 1)], "A": [10.0]})
        df2 = pl.DataFrame({"timestamp": [date(2024, 1, 1)], "A": [None]})
        result = add(df1, df2, filter=False)
        assert result["A"][0] is None

    def test_add_requires_two_inputs(self, arith_df: pl.DataFrame) -> None:
        """Test add raises error with less than 2 inputs."""
        with pytest.raises(ValueError, match="at least 2"):
            add(arith_df)

    def test_subtract_basic(self, arith_df: pl.DataFrame, arith_df2: pl.DataFrame) -> None:
        """Test basic subtraction."""
        result = subtract(arith_df, arith_df2)
        assert result["AAPL"][0] == 90.0  # 100 - 10
        assert result["AAPL"][1] == -55.0  # -50 - 5
        assert result["GOOGL"][0] == -120.0  # -150 - -30

    def test_subtract_filter_null(self) -> None:
        """Test subtract with filter=True treats null as 0."""
        df1 = pl.DataFrame({"timestamp": [date(2024, 1, 1)], "A": [10.0]})
        df2 = pl.DataFrame({"timestamp": [date(2024, 1, 1)], "A": [None]})
        result = subtract(df1, df2, filter=True)
        assert result["A"][0] == 10.0  # 10 - 0

    def test_multiply_two_inputs(self, arith_df: pl.DataFrame, arith_df2: pl.DataFrame) -> None:
        """Test multiplying two DataFrames."""
        result = multiply(arith_df, arith_df2)
        assert result["AAPL"][0] == 1000.0  # 100 * 10
        assert result["AAPL"][1] == -250.0  # -50 * 5
        assert result["GOOGL"][0] == 4500.0  # -150 * -30

    def test_multiply_filter_null(self) -> None:
        """Test multiply with filter=True treats null as 1."""
        df1 = pl.DataFrame({"timestamp": [date(2024, 1, 1)], "A": [10.0]})
        df2 = pl.DataFrame({"timestamp": [date(2024, 1, 1)], "A": [None]})
        result = multiply(df1, df2, filter=True)
        assert result["A"][0] == 10.0  # 10 * 1

    def test_multiply_without_filter_propagates_null(self) -> None:
        """Test multiply without filter propagates null."""
        df1 = pl.DataFrame({"timestamp": [date(2024, 1, 1)], "A": [10.0]})
        df2 = pl.DataFrame({"timestamp": [date(2024, 1, 1)], "A": [None]})
        result = multiply(df1, df2, filter=False)
        assert result["A"][0] is None

    def test_multiply_requires_two_inputs(self, arith_df: pl.DataFrame) -> None:
        """Test multiply raises error with less than 2 inputs."""
        with pytest.raises(ValueError, match="at least 2"):
            multiply(arith_df)

    def test_divide_basic(self, arith_df: pl.DataFrame, arith_df2: pl.DataFrame) -> None:
        """Test basic division."""
        result = divide(arith_df, arith_df2)
        assert result["AAPL"][0] == 10.0  # 100 / 10
        assert result["AAPL"][1] == -10.0  # -50 / 5

    def test_divide_by_zero_returns_null(self) -> None:
        """Test division by zero returns null."""
        df1 = pl.DataFrame({"timestamp": [date(2024, 1, 1)], "A": [10.0]})
        df2 = pl.DataFrame({"timestamp": [date(2024, 1, 1)], "A": [0.0]})
        result = divide(df1, df2)
        assert result["A"][0] is None

    def test_divide_zero_by_nonzero(self) -> None:
        """Test 0/x = 0."""
        df1 = pl.DataFrame({"timestamp": [date(2024, 1, 1)], "A": [0.0]})
        df2 = pl.DataFrame({"timestamp": [date(2024, 1, 1)], "A": [5.0]})
        result = divide(df1, df2)
        assert result["A"][0] == 0.0

    def test_inverse_basic(self) -> None:
        """Test basic inverse computation."""
        df = pl.DataFrame({
            "timestamp": [date(2024, 1, 1), date(2024, 1, 2)],
            "A": [2.0, 4.0],
        })
        result = inverse(df)
        assert result["A"][0] == 0.5  # 1/2
        assert result["A"][1] == 0.25  # 1/4

    def test_inverse_of_zero_returns_null(self) -> None:
        """Test 1/0 returns null."""
        df = pl.DataFrame({"timestamp": [date(2024, 1, 1)], "A": [0.0]})
        result = inverse(df)
        assert result["A"][0] is None

    def test_inverse_negative(self) -> None:
        """Test inverse of negative number."""
        df = pl.DataFrame({"timestamp": [date(2024, 1, 1)], "A": [-5.0]})
        result = inverse(df)
        assert result["A"][0] == -0.2

    def test_log_basic(self) -> None:
        """Test basic natural log computation."""
        df = pl.DataFrame({
            "timestamp": [date(2024, 1, 1), date(2024, 1, 2)],
            "A": [math.e, math.e ** 2],
        })
        result = log(df)
        assert abs(result["A"][0] - 1.0) < 0.01
        assert abs(result["A"][1] - 2.0) < 0.01

    def test_log_of_one(self) -> None:
        """Test ln(1) = 0."""
        df = pl.DataFrame({"timestamp": [date(2024, 1, 1)], "A": [1.0]})
        result = log(df)
        assert result["A"][0] == 0.0

    def test_log_of_zero_returns_null(self) -> None:
        """Test ln(0) returns null."""
        df = pl.DataFrame({"timestamp": [date(2024, 1, 1)], "A": [0.0]})
        result = log(df)
        assert result["A"][0] is None

    def test_log_of_negative_returns_null(self) -> None:
        """Test ln(negative) returns null."""
        df = pl.DataFrame({"timestamp": [date(2024, 1, 1)], "A": [-5.0]})
        result = log(df)
        assert result["A"][0] is None

    def test_max_two_inputs(self, arith_df: pl.DataFrame, arith_df2: pl.DataFrame) -> None:
        """Test element-wise max of two DataFrames."""
        result = op_max(arith_df, arith_df2)
        assert result["AAPL"][0] == 100.0  # max(100, 10)
        assert result["AAPL"][1] == 5.0  # max(-50, 5)
        assert result["GOOGL"][0] == -30.0  # max(-150, -30)

    def test_max_three_inputs(self) -> None:
        """Test element-wise max of three DataFrames."""
        df1 = pl.DataFrame({"timestamp": [date(2024, 1, 1)], "A": [1.0]})
        df2 = pl.DataFrame({"timestamp": [date(2024, 1, 1)], "A": [5.0]})
        df3 = pl.DataFrame({"timestamp": [date(2024, 1, 1)], "A": [3.0]})
        result = op_max(df1, df2, df3)
        assert result["A"][0] == 5.0

    def test_max_requires_two_inputs(self, arith_df: pl.DataFrame) -> None:
        """Test max raises error with less than 2 inputs."""
        with pytest.raises(ValueError, match="at least 2"):
            op_max(arith_df)

    def test_min_two_inputs(self, arith_df: pl.DataFrame, arith_df2: pl.DataFrame) -> None:
        """Test element-wise min of two DataFrames."""
        result = op_min(arith_df, arith_df2)
        assert result["AAPL"][0] == 10.0  # min(100, 10)
        assert result["AAPL"][1] == -50.0  # min(-50, 5)
        assert result["GOOGL"][0] == -150.0  # min(-150, -30)

    def test_min_three_inputs(self) -> None:
        """Test element-wise min of three DataFrames."""
        df1 = pl.DataFrame({"timestamp": [date(2024, 1, 1)], "A": [1.0]})
        df2 = pl.DataFrame({"timestamp": [date(2024, 1, 1)], "A": [5.0]})
        df3 = pl.DataFrame({"timestamp": [date(2024, 1, 1)], "A": [3.0]})
        result = op_min(df1, df2, df3)
        assert result["A"][0] == 1.0

    def test_min_requires_two_inputs(self, arith_df: pl.DataFrame) -> None:
        """Test min raises error with less than 2 inputs."""
        with pytest.raises(ValueError, match="at least 2"):
            op_min(arith_df)

    def test_power_basic(self) -> None:
        """Test basic power computation."""
        df_base = pl.DataFrame({
            "timestamp": [date(2024, 1, 1), date(2024, 1, 2)],
            "A": [2.0, 3.0],
        })
        df_exp = pl.DataFrame({
            "timestamp": [date(2024, 1, 1), date(2024, 1, 2)],
            "A": [3.0, 2.0],
        })
        result = power(df_base, df_exp)
        assert result["A"][0] == 8.0  # 2^3
        assert result["A"][1] == 9.0  # 3^2

    def test_power_zero_exponent(self) -> None:
        """Test x^0 = 1."""
        df_base = pl.DataFrame({"timestamp": [date(2024, 1, 1)], "A": [5.0]})
        df_exp = pl.DataFrame({"timestamp": [date(2024, 1, 1)], "A": [0.0]})
        result = power(df_base, df_exp)
        assert result["A"][0] == 1.0

    def test_power_negative_base(self) -> None:
        """Test negative base with integer exponent."""
        df_base = pl.DataFrame({"timestamp": [date(2024, 1, 1)], "A": [-2.0]})
        df_exp = pl.DataFrame({"timestamp": [date(2024, 1, 1)], "A": [3.0]})
        result = power(df_base, df_exp)
        assert result["A"][0] == -8.0  # (-2)^3

    def test_signed_power_positive(self) -> None:
        """Test signed_power with positive base."""
        df_base = pl.DataFrame({"timestamp": [date(2024, 1, 1)], "A": [4.0]})
        df_exp = pl.DataFrame({"timestamp": [date(2024, 1, 1)], "A": [2.0]})
        result = signed_power(df_base, df_exp)
        assert result["A"][0] == 16.0  # sign(4) * |4|^2 = 1 * 16

    def test_signed_power_negative(self) -> None:
        """Test signed_power with negative base."""
        df_base = pl.DataFrame({"timestamp": [date(2024, 1, 1)], "A": [-4.0]})
        df_exp = pl.DataFrame({"timestamp": [date(2024, 1, 1)], "A": [2.0]})
        result = signed_power(df_base, df_exp)
        assert result["A"][0] == -16.0  # sign(-4) * |-4|^2 = -1 * 16

    def test_signed_power_fractional_exp(self) -> None:
        """Test signed_power with fractional exponent preserves sign."""
        df_base = pl.DataFrame({"timestamp": [date(2024, 1, 1)], "A": [-9.0]})
        df_exp = pl.DataFrame({"timestamp": [date(2024, 1, 1)], "A": [0.5]})
        result = signed_power(df_base, df_exp)
        assert result["A"][0] == -3.0  # sign(-9) * |-9|^0.5 = -1 * 3

    def test_sqrt_basic(self) -> None:
        """Test basic square root computation."""
        df = pl.DataFrame({
            "timestamp": [date(2024, 1, 1), date(2024, 1, 2)],
            "A": [4.0, 9.0],
        })
        result = sqrt(df)
        assert result["A"][0] == 2.0
        assert result["A"][1] == 3.0

    def test_sqrt_of_zero(self) -> None:
        """Test sqrt(0) = 0."""
        df = pl.DataFrame({"timestamp": [date(2024, 1, 1)], "A": [0.0]})
        result = sqrt(df)
        assert result["A"][0] == 0.0

    def test_sqrt_of_negative_returns_null(self) -> None:
        """Test sqrt(negative) returns null."""
        df = pl.DataFrame({"timestamp": [date(2024, 1, 1)], "A": [-4.0]})
        result = sqrt(df)
        assert result["A"][0] is None

    def test_sign_positive(self) -> None:
        """Test sign of positive number."""
        df = pl.DataFrame({"timestamp": [date(2024, 1, 1)], "A": [5.0]})
        result = sign(df)
        assert result["A"][0] == 1

    def test_sign_negative(self) -> None:
        """Test sign of negative number."""
        df = pl.DataFrame({"timestamp": [date(2024, 1, 1)], "A": [-5.0]})
        result = sign(df)
        assert result["A"][0] == -1

    def test_sign_zero(self) -> None:
        """Test sign of zero."""
        df = pl.DataFrame({"timestamp": [date(2024, 1, 1)], "A": [0.0]})
        result = sign(df)
        assert result["A"][0] == 0

    def test_sign_null(self) -> None:
        """Test sign of null returns null."""
        df = pl.DataFrame({"timestamp": [date(2024, 1, 1)], "A": pl.Series([None], dtype=pl.Float64)})
        result = sign(df)
        assert result["A"][0] is None

    def test_reverse_basic(self, arith_df: pl.DataFrame) -> None:
        """Test basic negation."""
        result = reverse(arith_df)
        assert result["AAPL"][0] == -100.0
        assert result["AAPL"][1] == 50.0  # -(-50)
        assert result["GOOGL"][0] == 150.0  # -(-150)

    def test_reverse_zero(self) -> None:
        """Test -0 = 0."""
        df = pl.DataFrame({"timestamp": [date(2024, 1, 1)], "A": [0.0]})
        result = reverse(df)
        assert result["A"][0] == 0.0

    def test_densify_basic(self) -> None:
        """Test basic densify remapping."""
        df = pl.DataFrame({
            "timestamp": [date(2024, 1, 1)],
            "A": [10.0],
            "B": [30.0],
            "C": [10.0],  # Same as A
            "D": [20.0],
        })
        result = densify(df)
        # Values should be remapped to 0..n-1 based on unique sorted values
        # Sorted unique: [10, 20, 30] -> ranks: 10->0, 20->1, 30->2
        assert result["A"][0] == 0  # 10 -> rank 0
        assert result["C"][0] == 0  # 10 -> rank 0 (same as A)
        assert result["D"][0] == 1  # 20 -> rank 1
        assert result["B"][0] == 2  # 30 -> rank 2

    def test_densify_all_same(self) -> None:
        """Test densify with all same values."""
        df = pl.DataFrame({
            "timestamp": [date(2024, 1, 1)],
            "A": [5.0],
            "B": [5.0],
            "C": [5.0],
        })
        result = densify(df)
        # All same, should all be 0
        assert result["A"][0] == 0
        assert result["B"][0] == 0
        assert result["C"][0] == 0

    def test_densify_per_row(self) -> None:
        """Test that densify works per row independently."""
        df = pl.DataFrame({
            "timestamp": [date(2024, 1, 1), date(2024, 1, 2)],
            "A": [10.0, 100.0],
            "B": [20.0, 50.0],
        })
        result = densify(df)
        # Row 1: 10->0, 20->1
        # Row 2: 50->0, 100->1
        assert result["A"][0] == 0
        assert result["B"][0] == 1
        assert result["B"][1] == 0  # 50 is smaller in row 2
        assert result["A"][1] == 1  # 100 is larger in row 2

    def test_abs_with_null(self) -> None:
        """Test abs_ preserves nulls."""
        df = pl.DataFrame({"timestamp": [date(2024, 1, 1)], "A": pl.Series([None], dtype=pl.Float64)})
        result = op_abs(df)
        assert result["A"][0] is None

    def test_sqrt_with_null(self) -> None:
        """Test sqrt preserves nulls."""
        df = pl.DataFrame({"timestamp": [date(2024, 1, 1)], "A": pl.Series([None], dtype=pl.Float64)})
        result = sqrt(df)
        assert result["A"][0] is None

    def test_log_with_null(self) -> None:
        """Test log preserves nulls."""
        df = pl.DataFrame({"timestamp": [date(2024, 1, 1)], "A": pl.Series([None], dtype=pl.Float64)})
        result = log(df)
        assert result["A"][0] is None

    def test_reverse_with_null(self) -> None:
        """Test reverse preserves nulls."""
        df = pl.DataFrame({"timestamp": [date(2024, 1, 1)], "A": pl.Series([None], dtype=pl.Float64)})
        result = reverse(df)
        assert result["A"][0] is None


# =============================================================================
# LOGICAL OPERATORS
# =============================================================================


class TestLogicalOperators:
    """Logical operator tests."""

    @pytest.fixture
    def bool_df_a(self) -> pl.DataFrame:
        """Create boolean DataFrame A."""
        return pl.DataFrame({
            "timestamp": pl.date_range(date(2024, 1, 1), date(2024, 1, 5), eager=True),
            "AAPL": [True, False, True, False, True],
            "MSFT": [True, True, False, False, True],
        })

    @pytest.fixture
    def bool_df_b(self) -> pl.DataFrame:
        """Create boolean DataFrame B."""
        return pl.DataFrame({
            "timestamp": pl.date_range(date(2024, 1, 1), date(2024, 1, 5), eager=True),
            "AAPL": [True, True, True, False, False],
            "MSFT": [False, True, False, True, True],
        })

    @pytest.fixture
    def numeric_df(self) -> pl.DataFrame:
        """Create numeric DataFrame for comparisons."""
        return pl.DataFrame({
            "timestamp": pl.date_range(date(2024, 1, 1), date(2024, 1, 5), eager=True),
            "AAPL": [1.0, 2.0, 3.0, 4.0, 5.0],
            "MSFT": [5.0, 4.0, 3.0, 2.0, 1.0],
        })

    def test_and_(self, bool_df_a: pl.DataFrame, bool_df_b: pl.DataFrame) -> None:
        """Test logical AND."""
        result = and_(bool_df_a, bool_df_b)
        assert result.columns == bool_df_a.columns
        # AAPL: [T&T, F&T, T&T, F&F, T&F] = [T, F, T, F, F]
        assert result["AAPL"][0] is True
        assert result["AAPL"][1] is False
        assert result["AAPL"][2] is True
        assert result["AAPL"][3] is False
        assert result["AAPL"][4] is False

    def test_or_(self, bool_df_a: pl.DataFrame, bool_df_b: pl.DataFrame) -> None:
        """Test logical OR."""
        result = or_(bool_df_a, bool_df_b)
        # AAPL: [T|T, F|T, T|T, F|F, T|F] = [T, T, T, F, T]
        assert result["AAPL"][0] is True
        assert result["AAPL"][1] is True
        assert result["AAPL"][2] is True
        assert result["AAPL"][3] is False
        assert result["AAPL"][4] is True

    def test_not_(self, bool_df_a: pl.DataFrame) -> None:
        """Test logical NOT."""
        result = not_(bool_df_a)
        # AAPL: [~T, ~F, ~T, ~F, ~T] = [F, T, F, T, F]
        assert result["AAPL"][0] is False
        assert result["AAPL"][1] is True
        assert result["AAPL"][2] is False
        assert result["AAPL"][3] is True
        assert result["AAPL"][4] is False

    def test_if_else_df_df(self, bool_df_a: pl.DataFrame) -> None:
        """Test if_else with DataFrame then/else."""
        then_df = pl.DataFrame({
            "timestamp": pl.date_range(date(2024, 1, 1), date(2024, 1, 5), eager=True),
            "AAPL": [100.0, 100.0, 100.0, 100.0, 100.0],
            "MSFT": [200.0, 200.0, 200.0, 200.0, 200.0],
        })
        else_df = pl.DataFrame({
            "timestamp": pl.date_range(date(2024, 1, 1), date(2024, 1, 5), eager=True),
            "AAPL": [0.0, 0.0, 0.0, 0.0, 0.0],
            "MSFT": [0.0, 0.0, 0.0, 0.0, 0.0],
        })
        result = if_else(bool_df_a, then_df, else_df)
        # AAPL cond: [T, F, T, F, T] -> [100, 0, 100, 0, 100]
        assert result["AAPL"][0] == 100.0
        assert result["AAPL"][1] == 0.0
        assert result["AAPL"][2] == 100.0
        assert result["AAPL"][3] == 0.0
        assert result["AAPL"][4] == 100.0

    def test_if_else_scalar(self, bool_df_a: pl.DataFrame) -> None:
        """Test if_else with scalar then/else."""
        result = if_else(bool_df_a, 1.0, 0.0)
        # AAPL cond: [T, F, T, F, T] -> [1, 0, 1, 0, 1]
        assert result["AAPL"][0] == 1.0
        assert result["AAPL"][1] == 0.0
        assert result["AAPL"][2] == 1.0

    def test_is_nan_null(self) -> None:
        """Test is_nan with null values."""
        df = pl.DataFrame({
            "timestamp": pl.date_range(date(2024, 1, 1), date(2024, 1, 5), eager=True),
            "AAPL": [1.0, None, 3.0, None, 5.0],
        })
        result = is_nan(df)
        assert result["AAPL"][0] is False
        assert result["AAPL"][1] is True
        assert result["AAPL"][2] is False
        assert result["AAPL"][3] is True
        assert result["AAPL"][4] is False

    def test_is_nan_float_nan(self) -> None:
        """Test is_nan with float NaN values."""
        df = pl.DataFrame({
            "timestamp": pl.date_range(date(2024, 1, 1), date(2024, 1, 3), eager=True),
            "AAPL": [1.0, float("nan"), 3.0],
        })
        result = is_nan(df)
        assert result["AAPL"][0] is False
        assert result["AAPL"][1] is True
        assert result["AAPL"][2] is False

    def test_lt(self, numeric_df: pl.DataFrame) -> None:
        """Test less than comparison."""
        result = lt(numeric_df, 3.0)
        # AAPL: [1<3, 2<3, 3<3, 4<3, 5<3] = [T, T, F, F, F]
        assert result["AAPL"][0] is True
        assert result["AAPL"][1] is True
        assert result["AAPL"][2] is False
        assert result["AAPL"][3] is False
        assert result["AAPL"][4] is False

    def test_le(self, numeric_df: pl.DataFrame) -> None:
        """Test less than or equal comparison."""
        result = le(numeric_df, 3.0)
        # AAPL: [1<=3, 2<=3, 3<=3, 4<=3, 5<=3] = [T, T, T, F, F]
        assert result["AAPL"][0] is True
        assert result["AAPL"][1] is True
        assert result["AAPL"][2] is True
        assert result["AAPL"][3] is False
        assert result["AAPL"][4] is False

    def test_gt(self, numeric_df: pl.DataFrame) -> None:
        """Test greater than comparison."""
        result = gt(numeric_df, 3.0)
        # AAPL: [1>3, 2>3, 3>3, 4>3, 5>3] = [F, F, F, T, T]
        assert result["AAPL"][0] is False
        assert result["AAPL"][1] is False
        assert result["AAPL"][2] is False
        assert result["AAPL"][3] is True
        assert result["AAPL"][4] is True

    def test_ge(self, numeric_df: pl.DataFrame) -> None:
        """Test greater than or equal comparison."""
        result = ge(numeric_df, 3.0)
        # AAPL: [1>=3, 2>=3, 3>=3, 4>=3, 5>=3] = [F, F, T, T, T]
        assert result["AAPL"][0] is False
        assert result["AAPL"][1] is False
        assert result["AAPL"][2] is True
        assert result["AAPL"][3] is True
        assert result["AAPL"][4] is True

    def test_eq(self, numeric_df: pl.DataFrame) -> None:
        """Test equality comparison."""
        result = eq(numeric_df, 3.0)
        # AAPL: [1==3, 2==3, 3==3, 4==3, 5==3] = [F, F, T, F, F]
        assert result["AAPL"][0] is False
        assert result["AAPL"][1] is False
        assert result["AAPL"][2] is True
        assert result["AAPL"][3] is False
        assert result["AAPL"][4] is False

    def test_ne(self, numeric_df: pl.DataFrame) -> None:
        """Test not equal comparison."""
        result = ne(numeric_df, 3.0)
        # AAPL: [1!=3, 2!=3, 3!=3, 4!=3, 5!=3] = [T, T, F, T, T]
        assert result["AAPL"][0] is True
        assert result["AAPL"][1] is True
        assert result["AAPL"][2] is False
        assert result["AAPL"][3] is True
        assert result["AAPL"][4] is True

    def test_comparison_df_vs_df(self, numeric_df: pl.DataFrame) -> None:
        """Test comparison between two DataFrames."""
        other_df = pl.DataFrame({
            "timestamp": pl.date_range(date(2024, 1, 1), date(2024, 1, 5), eager=True),
            "AAPL": [2.0, 2.0, 2.0, 2.0, 2.0],
            "MSFT": [3.0, 3.0, 3.0, 3.0, 3.0],
        })
        result = gt(numeric_df, other_df)
        # AAPL: [1>2, 2>2, 3>2, 4>2, 5>2] = [F, F, T, T, T]
        assert result["AAPL"][0] is False
        assert result["AAPL"][1] is False
        assert result["AAPL"][2] is True
        assert result["AAPL"][3] is True
        assert result["AAPL"][4] is True

    def test_null_propagation(self) -> None:
        """Test that null propagates in comparisons."""
        df = pl.DataFrame({
            "timestamp": pl.date_range(date(2024, 1, 1), date(2024, 1, 3), eager=True),
            "AAPL": [1.0, None, 3.0],
        })
        result = lt(df, 2.0)
        # null comparisons return null
        assert result["AAPL"][0] is True
        assert result["AAPL"][1] is None
        assert result["AAPL"][2] is False


# =============================================================================
# VECTOR OPERATORS
# =============================================================================


class TestVectorOperators:
    """Vector operator tests."""

    @pytest.fixture
    def vector_df(self) -> pl.DataFrame:
        return pl.DataFrame({
            "timestamp": [date(2024, 1, 1), date(2024, 1, 2)],
            "AAPL": [[2.0, 3.0, 5.0, 6.0, 3.0, 8.0, 10.0], [1.0, 2.0, 3.0]],
            "MSFT": [[10.0, 20.0], [5.0, None, 10.0]],
        })

    def test_vec_avg(self, vector_df: pl.DataFrame) -> None:
        """Test vector mean."""
        result = vec_avg(vector_df)
        assert result.columns == vector_df.columns
        assert abs(result["AAPL"][0] - 5.2857) < 0.01  # 37/7
        assert abs(result["AAPL"][1] - 2.0) < 0.01    # 6/3

    def test_vec_sum(self, vector_df: pl.DataFrame) -> None:
        """Test vector sum."""
        result = vec_sum(vector_df)
        assert result["AAPL"][0] == 37.0
        assert result["AAPL"][1] == 6.0

    def test_vec_avg_with_nulls(self, vector_df: pl.DataFrame) -> None:
        """Test vec_avg ignores nulls in list."""
        result = vec_avg(vector_df)
        # MSFT[1] has None in list - Polars list.mean() ignores nulls
        assert abs(result["MSFT"][1] - 7.5) < 0.01


# =============================================================================
# GROUP OPERATORS
# =============================================================================


class TestGroupOperators:
    """Group operator tests."""

    @pytest.fixture
    def group_df(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Create sample data with group assignments."""
        x = pl.DataFrame({
            "timestamp": pl.date_range(date(2024, 1, 1), date(2024, 1, 3), eager=True),
            "A": [10.0, 20.0, 30.0],
            "B": [15.0, 25.0, 35.0],
            "C": [100.0, 200.0, 300.0],
            "D": [150.0, 250.0, 350.0],
        })
        group = pl.DataFrame({
            "timestamp": pl.date_range(date(2024, 1, 1), date(2024, 1, 3), eager=True),
            "A": ["tech", "tech", "tech"],
            "B": ["tech", "tech", "tech"],
            "C": ["fin", "fin", "fin"],
            "D": ["fin", "fin", "fin"],
        })
        return x, group

    def test_group_neutralize(self, group_df: tuple[pl.DataFrame, pl.DataFrame]) -> None:
        """Test group neutralization subtracts group mean."""
        x, group = group_df
        result = group_neutralize(x, group)

        # tech group at row 0: A=10, B=15, mean=12.5
        # A -> 10-12.5 = -2.5, B -> 15-12.5 = 2.5
        assert abs(result["A"][0] - (-2.5)) < 0.01
        assert abs(result["B"][0] - 2.5) < 0.01

        # fin group at row 0: C=100, D=150, mean=125
        # C -> 100-125 = -25, D -> 150-125 = 25
        assert abs(result["C"][0] - (-25.0)) < 0.01
        assert abs(result["D"][0] - 25.0) < 0.01

    def test_group_zscore(self, group_df: tuple[pl.DataFrame, pl.DataFrame]) -> None:
        """Test group z-score."""
        x, group = group_df
        result = group_zscore(x, group)

        # Z-scores within each group should sum to ~0
        for i in range(len(result)):
            tech_sum = result["A"][i] + result["B"][i]
            fin_sum = result["C"][i] + result["D"][i]
            assert abs(tech_sum) < 0.01
            assert abs(fin_sum) < 0.01

    def test_group_scale(self, group_df: tuple[pl.DataFrame, pl.DataFrame]) -> None:
        """Test group min-max scaling to [0, 1]."""
        x, group = group_df
        result = group_scale(x, group)

        # Min should be 0, max should be 1 within each group
        for i in range(len(result)):
            # tech: min(A,B)=0, max(A,B)=1
            assert result["A"][i] == 0.0
            assert result["B"][i] == 1.0
            # fin: min(C,D)=0, max(C,D)=1
            assert result["C"][i] == 0.0
            assert result["D"][i] == 1.0

    def test_group_rank(self, group_df: tuple[pl.DataFrame, pl.DataFrame]) -> None:
        """Test group rank in [0, 1]."""
        x, group = group_df
        result = group_rank(x, group)

        # tech: A < B, ranks: A=0, B=1
        assert result["A"][0] == 0.0
        assert result["B"][0] == 1.0
        # fin: C < D, ranks: C=0, D=1
        assert result["C"][0] == 0.0
        assert result["D"][0] == 1.0

    def test_group_rank_single_member(self) -> None:
        """Test single-member group returns 0.5."""
        x = pl.DataFrame({
            "timestamp": [date(2024, 1, 1)],
            "A": [10.0],
            "B": [20.0],
        })
        group = pl.DataFrame({
            "timestamp": [date(2024, 1, 1)],
            "A": ["grp1"],
            "B": ["grp2"],
        })
        result = group_rank(x, group)
        assert result["A"][0] == 0.5
        assert result["B"][0] == 0.5

    def test_group_mean(self) -> None:
        """Test weighted mean within groups."""
        x = pl.DataFrame({
            "timestamp": [date(2024, 1, 1)],
            "A": [10.0],
            "B": [20.0],
            "C": [100.0],
        })
        weight = pl.DataFrame({
            "timestamp": [date(2024, 1, 1)],
            "A": [1.0],
            "B": [3.0],
            "C": [1.0],
        })
        group = pl.DataFrame({
            "timestamp": [date(2024, 1, 1)],
            "A": ["tech"],
            "B": ["tech"],
            "C": ["fin"],
        })
        result = group_mean(x, weight, group)

        # tech: (10*1 + 20*3) / (1+3) = 70/4 = 17.5
        assert abs(result["A"][0] - 17.5) < 0.01
        assert abs(result["B"][0] - 17.5) < 0.01
        # fin: (100*1) / 1 = 100
        assert abs(result["C"][0] - 100.0) < 0.01

    def test_group_backfill(self) -> None:
        """Test filling NaN with winsorized group mean."""
        x = pl.DataFrame({
            "timestamp": pl.date_range(date(2024, 1, 1), date(2024, 1, 3), eager=True),
            "A": [10.0, None, 30.0],
            "B": [20.0, 25.0, 35.0],
            "C": [100.0, 200.0, None],
        })
        group = pl.DataFrame({
            "timestamp": pl.date_range(date(2024, 1, 1), date(2024, 1, 3), eager=True),
            "A": ["tech", "tech", "tech"],
            "B": ["tech", "tech", "tech"],
            "C": ["fin", "fin", "fin"],
        })
        result = group_backfill(x, group, d=3)

        # A[1] was None, should be filled with tech group mean
        assert result["A"][1] is not None
        assert result["A"][0] == 10.0
        assert result["B"][1] == 25.0

    def test_group_backfill_all_nan(self) -> None:
        """Test all-NaN window keeps NaN."""
        x = pl.DataFrame({
            "timestamp": pl.date_range(date(2024, 1, 1), date(2024, 1, 3), eager=True),
            "A": [None, None, None],
            "B": [None, None, None],
        })
        group = pl.DataFrame({
            "timestamp": pl.date_range(date(2024, 1, 1), date(2024, 1, 3), eager=True),
            "A": ["grp1", "grp1", "grp1"],
            "B": ["grp1", "grp1", "grp1"],
        })
        result = group_backfill(x, group, d=3)

        # All NaN in group, should stay NaN
        assert result["A"][2] is None
        assert result["B"][2] is None

    def test_group_scale_all_same(self) -> None:
        """Test all same values returns NaN for scale."""
        x = pl.DataFrame({
            "timestamp": [date(2024, 1, 1)],
            "A": [5.0],
            "B": [5.0],
        })
        group = pl.DataFrame({
            "timestamp": [date(2024, 1, 1)],
            "A": ["grp1"],
            "B": ["grp1"],
        })
        result = group_scale(x, group)
        # (5-5)/(5-5) = 0/0 = NaN
        assert result["A"][0] is None or math.isnan(result["A"][0])

    def test_group_zscore_all_same(self) -> None:
        """Test all same values returns NaN for zscore."""
        x = pl.DataFrame({
            "timestamp": [date(2024, 1, 1)],
            "A": [5.0],
            "B": [5.0],
        })
        group = pl.DataFrame({
            "timestamp": [date(2024, 1, 1)],
            "A": ["grp1"],
            "B": ["grp1"],
        })
        result = group_zscore(x, group)
        # std=0, (5-5)/0 = NaN
        assert result["A"][0] is None or math.isnan(result["A"][0])

    def test_group_rank_all_same(self) -> None:
        """Test all same values returns values in [0,1]."""
        x = pl.DataFrame({
            "timestamp": [date(2024, 1, 1)],
            "A": [5.0],
            "B": [5.0],
        })
        group = pl.DataFrame({
            "timestamp": [date(2024, 1, 1)],
            "A": ["grp1"],
            "B": ["grp1"],
        })
        result = group_rank(x, group)
        # Two values but same, ordinal rank gives 0.0 and 1.0
        assert 0.0 <= result["A"][0] <= 1.0
        assert 0.0 <= result["B"][0] <= 1.0


# =============================================================================
# OPERATOR COMPOSITION
# =============================================================================


class TestOperatorComposition:
    """Test composing operators."""

    def test_ts_mean_then_rank(self, wide_df: pl.DataFrame) -> None:
        """Test composing time-series and cross-sectional operators."""
        ma = ts_mean(wide_df, 3)
        ranked = rank(ma)

        assert ranked.columns == wide_df.columns
        # With partial windows, all rows have values and can be ranked
        assert 0.0 <= ranked["AAPL"][0] <= 1.0
        assert 0.0 <= ranked["AAPL"][1] <= 1.0

    def test_normalize_then_scale(self, wide_df: pl.DataFrame) -> None:
        """Test composing cross-sectional operators."""
        normalized = normalize(wide_df)
        scaled = scale(normalized, scale=1.0)

        # Should still sum to ~0 (normalize preserved)
        # But absolute sum should be ~1 (scale)
        for i in range(len(scaled)):
            row_sum = scaled["AAPL"][i] + scaled["MSFT"][i] + scaled["GOOGL"][i]
            assert abs(row_sum) < 0.01


# =============================================================================
# EDGE CASES
# =============================================================================


class TestTsRegressionRetTypes:
    """Tests for ts_regression rettype parameter."""

    @pytest.fixture
    def regression_data(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Create y and x DataFrames for regression tests."""
        # y = 2*x + 1 + noise
        x_vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        y_vals = [3.1, 5.0, 7.2, 8.9, 11.1, 13.0, 14.8, 17.1, 19.0, 21.0]
        y = pl.DataFrame({
            "timestamp": pl.date_range(date(2024, 1, 1), date(2024, 1, 10), eager=True),
            "A": y_vals,
        })
        x = pl.DataFrame({
            "timestamp": pl.date_range(date(2024, 1, 1), date(2024, 1, 10), eager=True),
            "A": x_vals,
        })
        return y, x

    def test_ts_regression_alpha(self, regression_data: tuple[pl.DataFrame, pl.DataFrame]) -> None:
        """Test regression alpha (intercept) rettype=2."""
        y, x = regression_data
        result = ts_regression(y, x, 5, rettype=2)
        # Alpha should be ~1 for y=2x+1
        assert result["A"][4] is not None
        assert abs(result["A"][4] - 1.0) < 1.0  # Allow tolerance

    def test_ts_regression_predicted(self, regression_data: tuple[pl.DataFrame, pl.DataFrame]) -> None:
        """Test regression predicted rettype=3."""
        y, x = regression_data
        result = ts_regression(y, x, 5, rettype=3)
        # Predicted values should exist
        assert result["A"][4] is not None

    def test_ts_regression_correlation(self, regression_data: tuple[pl.DataFrame, pl.DataFrame]) -> None:
        """Test regression correlation rettype=4."""
        y, x = regression_data
        result = ts_regression(y, x, 5, rettype=4)
        # Correlation should be close to 1 for linear relationship
        assert result["A"][4] is not None
        assert result["A"][4] > 0.9

    def test_ts_regression_tstat_beta(self, regression_data: tuple[pl.DataFrame, pl.DataFrame]) -> None:
        """Test regression t-stat for beta rettype=6."""
        y, x = regression_data
        result = ts_regression(y, x, 5, rettype=6)
        # t-stat should be large for significant relationship
        assert result["A"][4] is not None
        assert abs(result["A"][4]) > 2.0

    def test_ts_regression_tstat_alpha(self, regression_data: tuple[pl.DataFrame, pl.DataFrame]) -> None:
        """Test regression t-stat for alpha rettype=7."""
        y, x = regression_data
        result = ts_regression(y, x, 5, rettype=7)
        assert result["A"][4] is not None

    def test_ts_regression_stderr_beta(self, regression_data: tuple[pl.DataFrame, pl.DataFrame]) -> None:
        """Test regression std error of beta rettype=8."""
        y, x = regression_data
        result = ts_regression(y, x, 5, rettype=8)
        # Std error should be small for good fit
        assert result["A"][4] is not None
        assert result["A"][4] > 0

    def test_ts_regression_stderr_alpha(self, regression_data: tuple[pl.DataFrame, pl.DataFrame]) -> None:
        """Test regression std error of alpha rettype=9."""
        y, x = regression_data
        result = ts_regression(y, x, 5, rettype=9)
        assert result["A"][4] is not None
        assert result["A"][4] > 0

    def test_ts_regression_invalid_rettype(self, regression_data: tuple[pl.DataFrame, pl.DataFrame]) -> None:
        """Test invalid rettype returns None."""
        y, x = regression_data
        result = ts_regression(y, x, 5, rettype=99)
        # Invalid rettype should give None
        assert result["A"][4] is None

    def test_ts_regression_with_nulls(self) -> None:
        """Test regression filters out null pairs and computes with available data."""
        y = pl.DataFrame({
            "timestamp": pl.date_range(date(2024, 1, 1), date(2024, 1, 5), eager=True),
            "A": [1.0, None, 3.0, 4.0, 5.0],
        })
        x = pl.DataFrame({
            "timestamp": pl.date_range(date(2024, 1, 1), date(2024, 1, 5), eager=True),
            "A": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        result = ts_regression(y, x, 3, rettype=1)
        # At idx 2: pairs [(1,1), (3,3)] after filtering null, beta = 1.0
        assert result["A"][2] == 1.0

    def test_ts_regression_zero_variance(self) -> None:
        """Test regression with zero variance in x (ss_xx=0)."""
        y = pl.DataFrame({
            "timestamp": pl.date_range(date(2024, 1, 1), date(2024, 1, 5), eager=True),
            "A": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        x = pl.DataFrame({
            "timestamp": pl.date_range(date(2024, 1, 1), date(2024, 1, 5), eager=True),
            "A": [5.0, 5.0, 5.0, 5.0, 5.0],  # Constant x
        })
        result = ts_regression(y, x, 3, rettype=1)
        # Zero variance should return None
        assert result["A"][2] is None


class TestTsQuantileEdgeCases:
    """Tests for ts_quantile edge cases."""

    def test_ts_quantile_uniform(self) -> None:
        """Test ts_quantile with uniform driver."""
        df = pl.DataFrame({
            "timestamp": pl.date_range(date(2024, 1, 1), date(2024, 1, 5), eager=True),
            "A": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        result = ts_quantile(df, 3, driver="uniform")
        # Uniform output should be in [-1, 1]
        for i in range(2, len(result)):
            val = result["A"][i]
            if val is not None:
                assert -1.0 <= val <= 1.0

    def test_ts_quantile_single_unique_value(self) -> None:
        """Test ts_quantile with single unique value in window."""
        df = pl.DataFrame({
            "timestamp": pl.date_range(date(2024, 1, 1), date(2024, 1, 5), eager=True),
            "A": [5.0, 5.0, 5.0, 5.0, 5.0],
        })
        result = ts_quantile(df, 3, driver="gaussian")
        # All same values: idx=0, rank_pct=0.5/3, inv_norm(0.166) ~ -0.97
        assert result["A"][2] is not None
        assert not math.isnan(result["A"][2])

    def test_ts_quantile_with_nulls(self) -> None:
        """Test ts_quantile with null values."""
        df = pl.DataFrame({
            "timestamp": pl.date_range(date(2024, 1, 1), date(2024, 1, 5), eager=True),
            "A": [1.0, None, 3.0, 4.0, 5.0],
        })
        result = ts_quantile(df, 3, driver="gaussian")
        # Should handle nulls gracefully
        assert result is not None


class TestTsCorrCovarianceEdgeCases:
    """Tests for ts_corr and ts_covariance edge cases."""

    def test_ts_corr_with_nulls(self) -> None:
        """Test ts_corr with null values."""
        x = pl.DataFrame({
            "timestamp": pl.date_range(date(2024, 1, 1), date(2024, 1, 5), eager=True),
            "A": [1.0, None, 3.0, 4.0, 5.0],
        })
        y = pl.DataFrame({
            "timestamp": pl.date_range(date(2024, 1, 1), date(2024, 1, 5), eager=True),
            "A": [2.0, 4.0, 6.0, 8.0, 10.0],
        })
        result = ts_corr(x, y, 3)
        # Window with null should return None
        assert result["A"][2] is None

    def test_ts_corr_zero_std(self) -> None:
        """Test ts_corr with zero standard deviation."""
        x = pl.DataFrame({
            "timestamp": pl.date_range(date(2024, 1, 1), date(2024, 1, 5), eager=True),
            "A": [5.0, 5.0, 5.0, 5.0, 5.0],  # Constant
        })
        y = pl.DataFrame({
            "timestamp": pl.date_range(date(2024, 1, 1), date(2024, 1, 5), eager=True),
            "A": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        result = ts_corr(x, y, 3)
        # Zero std should return None
        assert result["A"][2] is None

    def test_ts_covariance_with_nulls(self) -> None:
        """Test ts_covariance with null values."""
        x = pl.DataFrame({
            "timestamp": pl.date_range(date(2024, 1, 1), date(2024, 1, 5), eager=True),
            "A": [1.0, 2.0, None, 4.0, 5.0],
        })
        y = pl.DataFrame({
            "timestamp": pl.date_range(date(2024, 1, 1), date(2024, 1, 5), eager=True),
            "A": [2.0, 4.0, 6.0, 8.0, 10.0],
        })
        result = ts_covariance(x, y, 3)
        # Window with null should return None
        assert result["A"][3] is None


class TestTsRankEdgeCases:
    """Tests for ts_rank edge cases."""

    def test_ts_rank_with_nulls(self) -> None:
        """Test ts_rank with null current value."""
        df = pl.DataFrame({
            "timestamp": pl.date_range(date(2024, 1, 1), date(2024, 1, 5), eager=True),
            "A": [1.0, 2.0, None, 4.0, 5.0],
        })
        result = ts_rank(df, 3)
        # Null current value should return None
        assert result["A"][2] is None

    def test_ts_rank_single_unique(self) -> None:
        """Test ts_rank with single unique non-null value."""
        df = pl.DataFrame({
            "timestamp": pl.date_range(date(2024, 1, 1), date(2024, 1, 5), eager=True),
            "A": [5.0, 5.0, 5.0, 5.0, 5.0],
        })
        result = ts_rank(df, 3)
        # All same values: current is at idx 0, len=3, rank=0/(3-1)=0
        assert result["A"][2] == 0.0


class TestTsDecayLinearEdgeCases:
    """Tests for ts_decay_linear edge cases."""

    def test_ts_decay_linear_dense_true(self) -> None:
        """Test ts_decay_linear with dense=True (skip nulls)."""
        df = pl.DataFrame({
            "timestamp": pl.date_range(date(2024, 1, 1), date(2024, 1, 5), eager=True),
            "A": [1.0, None, 3.0, 4.0, 5.0],
        })
        result = ts_decay_linear(df, 3, dense=True)
        # Dense mode skips nulls, should return value
        assert result["A"][4] is not None

    def test_ts_decay_linear_with_nulls_dense_false(self) -> None:
        """Test ts_decay_linear with nulls and dense=False."""
        df = pl.DataFrame({
            "timestamp": pl.date_range(date(2024, 1, 1), date(2024, 1, 5), eager=True),
            "A": [1.0, None, 3.0, 4.0, 5.0],
        })
        result = ts_decay_linear(df, 3, dense=False)
        # Non-dense mode: window with null returns None
        assert result["A"][2] is None


class TestOtherOperatorEdgeCases:
    """Tests for other operator edge cases."""

    def test_hump_with_none(self) -> None:
        """Test hump when previous value is None."""
        df = pl.DataFrame({
            "timestamp": pl.date_range(date(2024, 1, 1), date(2024, 1, 3), eager=True),
            "A": [None, 100.0, 150.0],
        })
        result = hump(df, hump=0.1)
        # When prev is None, curr should pass through
        assert result["A"][1] == 100.0

    def test_last_diff_value_all_same(self) -> None:
        """Test last_diff_value when all values are same."""
        df = pl.DataFrame({
            "timestamp": pl.date_range(date(2024, 1, 1), date(2024, 1, 5), eager=True),
            "A": [5.0, 5.0, 5.0, 5.0, 5.0],
        })
        result = last_diff_value(df, 3)
        # No different value exists, should return None
        assert result["A"][4] is None

    def test_ts_arg_max_short_window(self) -> None:
        """Test ts_arg_max when window not filled."""
        df = pl.DataFrame({
            "timestamp": pl.date_range(date(2024, 1, 1), date(2024, 1, 3), eager=True),
            "A": [1.0, 2.0, 3.0],
        })
        result = ts_arg_max(df, 5)
        # Window size 5, only 3 values, should be None
        assert result["A"][2] is None

    def test_ts_arg_min_short_window(self) -> None:
        """Test ts_arg_min when window not filled."""
        df = pl.DataFrame({
            "timestamp": pl.date_range(date(2024, 1, 1), date(2024, 1, 3), eager=True),
            "A": [3.0, 2.0, 1.0],
        })
        result = ts_arg_min(df, 5)
        # Window size 5, only 3 values, should be None
        assert result["A"][2] is None


class TestEdgeCases:
    """Edge case tests."""

    def test_single_column(self) -> None:
        """Test operators with single symbol column."""
        df = pl.DataFrame({
            "timestamp": pl.date_range(date(2024, 1, 1), date(2024, 1, 5), eager=True),
            "AAPL": [100.0, 102.0, 101.0, 103.0, 105.0],
        })

        result = ts_mean(df, 3)
        assert result.columns == df.columns

    def test_with_nulls(self) -> None:
        """Test operators handle nulls correctly."""
        df = pl.DataFrame({
            "timestamp": pl.date_range(date(2024, 1, 1), date(2024, 1, 5), eager=True),
            "AAPL": [100.0, None, 101.0, 103.0, 105.0],
            "MSFT": [200.0, 202.0, None, 203.0, 205.0],
        })

        result = ts_mean(df, 3)
        # Should not raise, nulls propagate
        assert result is not None


# =============================================================================
# COVERAGE BOOST TESTS
# =============================================================================


class TestBucketOperator:
    """Tests for bucket operator."""

    def test_bucket_basic(self) -> None:
        """Test basic bucketing."""
        df = pl.DataFrame({
            "timestamp": [date(2024, 1, 1)],
            "A": [0.1],
            "B": [0.3],
            "C": [0.6],
            "D": [0.9],
        })
        result = bucket(df, "0,1,0.25")
        assert result["A"][0] == 0.0   # 0.1 -> bucket [0, 0.25)
        assert result["B"][0] == 0.25  # 0.3 -> bucket [0.25, 0.5)
        assert result["C"][0] == 0.5   # 0.6 -> bucket [0.5, 0.75)
        assert result["D"][0] == 0.75  # 0.9 -> bucket [0.75, 1.0]

    def test_bucket_clipping(self) -> None:
        """Test bucket clips values outside range."""
        df = pl.DataFrame({
            "timestamp": [date(2024, 1, 1)],
            "A": [-0.5],  # Below start
            "B": [1.5],   # Above end
        })
        result = bucket(df, "0,1,0.25")
        assert result["A"][0] == 0.0   # Clipped to start
        assert result["B"][0] == 0.75  # Clipped to end - step

    def test_bucket_invalid_spec(self) -> None:
        """Test bucket raises on invalid range spec."""
        df = pl.DataFrame({"timestamp": [date(2024, 1, 1)], "A": [0.5]})
        with pytest.raises(ValueError, match="range_spec"):
            bucket(df, "0,1")

    def test_bucket_invalid_step(self) -> None:
        """Test bucket raises on non-positive step."""
        df = pl.DataFrame({"timestamp": [date(2024, 1, 1)], "A": [0.5]})
        with pytest.raises(ValueError, match="step must be positive"):
            bucket(df, "0,1,-0.25")

    def test_bucket_multiple_rows(self) -> None:
        """Test bucket works across multiple rows."""
        df = pl.DataFrame({
            "timestamp": pl.date_range(date(2024, 1, 1), date(2024, 1, 3), eager=True),
            "A": [0.1, 0.5, 0.9],
        })
        result = bucket(df, "0,1,0.5")
        assert result["A"][0] == 0.0
        assert result["A"][1] == 0.5
        assert result["A"][2] == 0.5


class TestQuantileCauchy:
    """Test quantile with cauchy driver."""

    def test_quantile_cauchy(self) -> None:
        """Test quantile transformation with cauchy driver."""
        df = pl.DataFrame({
            "timestamp": [date(2024, 1, 1)],
            "A": [1.0],
            "B": [2.0],
            "C": [3.0],
        })
        result = quantile(df, driver="cauchy")
        assert result.columns == df.columns
        for col in ["A", "B", "C"]:
            val = result[col][0]
            if val is not None:
                assert not math.isnan(val)

    def test_quantile_unknown_driver_raises(self) -> None:
        """Test quantile raises on unknown driver."""
        df = pl.DataFrame({
            "timestamp": [date(2024, 1, 1)],
            "A": [1.0],
            "B": [2.0],
            "C": [3.0],
        })
        with pytest.raises(ValueError, match="Unknown driver"):
            quantile(df, driver="invalid_driver")


class TestArithmeticBuiltinFallbacks:
    """Tests for arithmetic operator built-in fallbacks."""

    def test_abs_scalar(self) -> None:
        """Test abs() with scalar falls back to built-in."""
        assert op_abs(-5) == 5
        assert op_abs(3) == 3
        assert op_abs(0) == 0

    def test_max_builtin_scalar(self) -> None:
        """Test max() with scalars falls back to built-in."""
        assert op_max(1, 2, 3) == 3
        assert op_max([5, 2, 8]) == 8

    def test_max_builtin_kwargs(self) -> None:
        """Test max() with kwargs falls back to built-in."""
        assert op_max([1, 2, 3], key=lambda x: -x) == 1

    def test_min_builtin_scalar(self) -> None:
        """Test min() with scalars falls back to built-in."""
        assert op_min(1, 2, 3) == 1
        assert op_min([5, 2, 8]) == 2

    def test_min_builtin_kwargs(self) -> None:
        """Test min() with kwargs falls back to built-in."""
        assert op_min([1, 2, 3], key=lambda x: -x) == 3

    def test_multiply_all_scalars(self) -> None:
        """Test multiply with all scalar inputs."""
        assert multiply(2, 3) == 6
        assert multiply(2, 3, 4) == 24

    def test_multiply_scalar_first(self) -> None:
        """Test multiply when first arg is scalar, second is DataFrame."""
        df = pl.DataFrame({"timestamp": [date(2024, 1, 1)], "A": [5.0]})
        result = multiply(3, df)
        assert result["A"][0] == 15.0

    def test_multiply_scalar_in_loop(self) -> None:
        """Test multiply with DataFrame then scalar."""
        df = pl.DataFrame({"timestamp": [date(2024, 1, 1)], "A": [5.0]})
        result = multiply(df, 3)
        assert result["A"][0] == 15.0

    def test_power_both_scalars(self) -> None:
        """Test power with both scalars."""
        assert power(2, 3) == 8
        assert power(3, 0) == 1

    def test_power_scalar_base_df_exp(self) -> None:
        """Test power with scalar base and DataFrame exponent."""
        df = pl.DataFrame({"timestamp": [date(2024, 1, 1)], "A": [3.0]})
        result = power(2, df)
        assert result["A"][0] == 8.0  # 2^3

    def test_signed_power_both_scalars(self) -> None:
        """Test signed_power with both scalars."""
        assert signed_power(4, 2) == 16
        assert signed_power(-4, 2) == -16

    def test_signed_power_scalar_df(self) -> None:
        """Test signed_power with scalar base and DataFrame exponent."""
        df = pl.DataFrame({"timestamp": [date(2024, 1, 1)], "A": [2.0]})
        result = signed_power(3, df)
        assert result["A"][0] == 9.0  # sign(3) * |3|^2


class TestLookbackOperators:
    """Tests for time-series operators with lookback parameter."""

    def test_ts_delta_with_lookback(self) -> None:
        """Test ts_delta with lookback DataFrame."""
        lookback = pl.DataFrame({
            "timestamp": [date(2024, 1, 1), date(2024, 1, 2)],
            "A": [100.0, 102.0],
        })
        x = pl.DataFrame({
            "timestamp": [date(2024, 1, 3), date(2024, 1, 4)],
            "A": [105.0, 108.0],
        })
        result = ts_delta(x, 1, lookback=lookback)
        assert len(result) == 2
        # First value should be 105 - 102 = 3 (using lookback)
        assert result["A"][0] == 3.0
        assert result["A"][1] == 3.0  # 108 - 105

    def test_ts_delay_with_lookback(self) -> None:
        """Test ts_delay with lookback DataFrame."""
        lookback = pl.DataFrame({
            "timestamp": [date(2024, 1, 1), date(2024, 1, 2)],
            "A": [100.0, 102.0],
        })
        x = pl.DataFrame({
            "timestamp": [date(2024, 1, 3), date(2024, 1, 4)],
            "A": [105.0, 108.0],
        })
        result = ts_delay(x, 1, lookback=lookback)
        assert len(result) == 2
        # First value should be 102 (from lookback)
        assert result["A"][0] == 102.0
        assert result["A"][1] == 105.0


class TestTsQuantileUniform:
    """Test ts_quantile uniform path."""

    def test_ts_quantile_uniform_path(self) -> None:
        """Test ts_quantile with uniform driver exercises else branch."""
        df = pl.DataFrame({
            "timestamp": pl.date_range(date(2024, 1, 1), date(2024, 1, 5), eager=True),
            "A": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        result = ts_quantile(df, 3, driver="uniform")
        # Uniform output in [-1, 1]
        for i in range(2, len(result)):
            val = result["A"][i]
            if val is not None:
                assert -1.0 <= val <= 1.0


class TestRankLargeUniverse:
    """Test rank with large universe to trigger bucket-based ranking."""

    def test_rank_bucket_based(self) -> None:
        """Test rank with 33+ symbols to trigger bucket-based path."""
        n_symbols = 40
        data = {"timestamp": [date(2024, 1, 1), date(2024, 1, 2)]}
        for i in range(n_symbols):
            data[f"S{i:03d}"] = [float(i), float(n_symbols - i)]
        df = pl.DataFrame(data)
        result = rank(df, rate=2)  # rate>0 with n_symbols>=32 triggers bucket path
        assert result.columns == df.columns
        # Values should be in [0, 1]
        for col in df.columns[1:]:
            for val in result[col]:
                if val is not None:
                    assert 0.0 <= val <= 1.0
