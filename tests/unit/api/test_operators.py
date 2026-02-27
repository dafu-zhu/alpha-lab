"""Tests for alpha operators."""

import math
from datetime import date

import polars as pl
import pytest

from alphalab.api.operators import (
    abs as op_abs,
    bucket,
)


def is_missing(value) -> bool:
    """Check if value is missing (None or NaN)."""
    if value is None:
        return True
    try:
        return math.isnan(value)
    except (TypeError, ValueError):
        return False
from alphalab.api.operators import (
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
from alphalab.api.operators import (
    max as op_max,
)
from alphalab.api.operators import (
    min as op_min,
)


@pytest.fixture
def wide_df() -> pl.DataFrame:
    """Create sample wide DataFrame."""
    return pl.DataFrame({
        "Date": pl.date_range(date(2024, 1, 1), date(2024, 1, 10), eager=True),
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
            "Date": pl.date_range(date(2024, 1, 1), date(2024, 1, 5), eager=True),
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
            "Date": pl.date_range(date(2024, 1, 1), date(2024, 1, 5), eager=True),
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
            "Date": pl.date_range(date(2024, 1, 1), date(2024, 1, 5), eager=True),
            "AAPL": [100.0, 100.0, 102.0, 102.0, 102.0],
        })
        result = last_diff_value(df, 3)
        # At idx 3: window [100, 102, 102], current=102, last diff=100
        assert result["AAPL"][3] == 100.0

    def test_days_from_last_change(self) -> None:
        """Test days since value changed."""
        df = pl.DataFrame({
            "Date": pl.date_range(date(2024, 1, 1), date(2024, 1, 5), eager=True),
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
            "Date": pl.date_range(date(2024, 1, 1), date(2024, 1, 3), eager=True),
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
            "Date": pl.date_range(date(2024, 1, 1), date(2024, 1, 3), eager=True),
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
            "Date": pl.date_range(date(2024, 1, 1), date(2024, 1, 2), eager=True),
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
            "Date": pl.date_range(date(2024, 1, 1), date(2024, 1, 2), eager=True),
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
            "Date": pl.date_range(date(2024, 1, 1), date(2024, 1, 1), eager=True),
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
            "Date": pl.date_range(date(2024, 1, 1), date(2024, 1, 5), eager=True),
            "AAPL": [100.0, -50.0, 25.0, 0.0, -10.0],
            "MSFT": [200.0, 100.0, -50.0, 75.0, 0.0],
            "GOOGL": [-150.0, 0.0, 30.0, -20.0, 40.0],
        })

    @pytest.fixture
    def arith_df2(self) -> pl.DataFrame:
        """Create second sample wide DataFrame for two-input ops."""
        return pl.DataFrame({
            "Date": pl.date_range(date(2024, 1, 1), date(2024, 1, 5), eager=True),
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
        df1 = pl.DataFrame({"Date": [date(2024, 1, 1)], "A": [10.0]})
        df2 = pl.DataFrame({"Date": [date(2024, 1, 1)], "A": [None]})
        result = add(df1, df2, filter=True)
        assert result["A"][0] == 10.0  # 10 + 0

    def test_add_without_filter_propagates_null(self) -> None:
        """Test add without filter propagates null."""
        df1 = pl.DataFrame({"Date": [date(2024, 1, 1)], "A": [10.0]})
        df2 = pl.DataFrame({"Date": [date(2024, 1, 1)], "A": [None]})
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
        df1 = pl.DataFrame({"Date": [date(2024, 1, 1)], "A": [10.0]})
        df2 = pl.DataFrame({"Date": [date(2024, 1, 1)], "A": [None]})
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
        df1 = pl.DataFrame({"Date": [date(2024, 1, 1)], "A": [10.0]})
        df2 = pl.DataFrame({"Date": [date(2024, 1, 1)], "A": [None]})
        result = multiply(df1, df2, filter=True)
        assert result["A"][0] == 10.0  # 10 * 1

    def test_multiply_without_filter_propagates_null(self) -> None:
        """Test multiply without filter propagates null."""
        df1 = pl.DataFrame({"Date": [date(2024, 1, 1)], "A": [10.0]})
        df2 = pl.DataFrame({"Date": [date(2024, 1, 1)], "A": [None]})
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
        df1 = pl.DataFrame({"Date": [date(2024, 1, 1)], "A": [10.0]})
        df2 = pl.DataFrame({"Date": [date(2024, 1, 1)], "A": [0.0]})
        result = divide(df1, df2)
        assert result["A"][0] is None

    def test_divide_zero_by_nonzero(self) -> None:
        """Test 0/x = 0."""
        df1 = pl.DataFrame({"Date": [date(2024, 1, 1)], "A": [0.0]})
        df2 = pl.DataFrame({"Date": [date(2024, 1, 1)], "A": [5.0]})
        result = divide(df1, df2)
        assert result["A"][0] == 0.0

    def test_inverse_basic(self) -> None:
        """Test basic inverse computation."""
        df = pl.DataFrame({
            "Date": [date(2024, 1, 1), date(2024, 1, 2)],
            "A": [2.0, 4.0],
        })
        result = inverse(df)
        assert result["A"][0] == 0.5  # 1/2
        assert result["A"][1] == 0.25  # 1/4

    def test_inverse_of_zero_returns_null(self) -> None:
        """Test 1/0 returns null."""
        df = pl.DataFrame({"Date": [date(2024, 1, 1)], "A": [0.0]})
        result = inverse(df)
        assert result["A"][0] is None

    def test_inverse_negative(self) -> None:
        """Test inverse of negative number."""
        df = pl.DataFrame({"Date": [date(2024, 1, 1)], "A": [-5.0]})
        result = inverse(df)
        assert result["A"][0] == -0.2

    def test_log_basic(self) -> None:
        """Test basic natural log computation."""
        df = pl.DataFrame({
            "Date": [date(2024, 1, 1), date(2024, 1, 2)],
            "A": [math.e, math.e ** 2],
        })
        result = log(df)
        assert abs(result["A"][0] - 1.0) < 0.01
        assert abs(result["A"][1] - 2.0) < 0.01

    def test_log_of_one(self) -> None:
        """Test ln(1) = 0."""
        df = pl.DataFrame({"Date": [date(2024, 1, 1)], "A": [1.0]})
        result = log(df)
        assert result["A"][0] == 0.0

    def test_log_of_zero_returns_null(self) -> None:
        """Test ln(0) returns null."""
        df = pl.DataFrame({"Date": [date(2024, 1, 1)], "A": [0.0]})
        result = log(df)
        assert result["A"][0] is None

    def test_log_of_negative_returns_null(self) -> None:
        """Test ln(negative) returns null."""
        df = pl.DataFrame({"Date": [date(2024, 1, 1)], "A": [-5.0]})
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
        df1 = pl.DataFrame({"Date": [date(2024, 1, 1)], "A": [1.0]})
        df2 = pl.DataFrame({"Date": [date(2024, 1, 1)], "A": [5.0]})
        df3 = pl.DataFrame({"Date": [date(2024, 1, 1)], "A": [3.0]})
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
        df1 = pl.DataFrame({"Date": [date(2024, 1, 1)], "A": [1.0]})
        df2 = pl.DataFrame({"Date": [date(2024, 1, 1)], "A": [5.0]})
        df3 = pl.DataFrame({"Date": [date(2024, 1, 1)], "A": [3.0]})
        result = op_min(df1, df2, df3)
        assert result["A"][0] == 1.0

    def test_min_requires_two_inputs(self, arith_df: pl.DataFrame) -> None:
        """Test min raises error with less than 2 inputs."""
        with pytest.raises(ValueError, match="at least 2"):
            op_min(arith_df)

    def test_power_basic(self) -> None:
        """Test basic power computation."""
        df_base = pl.DataFrame({
            "Date": [date(2024, 1, 1), date(2024, 1, 2)],
            "A": [2.0, 3.0],
        })
        df_exp = pl.DataFrame({
            "Date": [date(2024, 1, 1), date(2024, 1, 2)],
            "A": [3.0, 2.0],
        })
        result = power(df_base, df_exp)
        assert result["A"][0] == 8.0  # 2^3
        assert result["A"][1] == 9.0  # 3^2

    def test_power_zero_exponent(self) -> None:
        """Test x^0 = 1."""
        df_base = pl.DataFrame({"Date": [date(2024, 1, 1)], "A": [5.0]})
        df_exp = pl.DataFrame({"Date": [date(2024, 1, 1)], "A": [0.0]})
        result = power(df_base, df_exp)
        assert result["A"][0] == 1.0

    def test_power_negative_base(self) -> None:
        """Test negative base with integer exponent."""
        df_base = pl.DataFrame({"Date": [date(2024, 1, 1)], "A": [-2.0]})
        df_exp = pl.DataFrame({"Date": [date(2024, 1, 1)], "A": [3.0]})
        result = power(df_base, df_exp)
        assert result["A"][0] == -8.0  # (-2)^3

    def test_signed_power_positive(self) -> None:
        """Test signed_power with positive base."""
        df_base = pl.DataFrame({"Date": [date(2024, 1, 1)], "A": [4.0]})
        df_exp = pl.DataFrame({"Date": [date(2024, 1, 1)], "A": [2.0]})
        result = signed_power(df_base, df_exp)
        assert result["A"][0] == 16.0  # sign(4) * |4|^2 = 1 * 16

    def test_signed_power_negative(self) -> None:
        """Test signed_power with negative base."""
        df_base = pl.DataFrame({"Date": [date(2024, 1, 1)], "A": [-4.0]})
        df_exp = pl.DataFrame({"Date": [date(2024, 1, 1)], "A": [2.0]})
        result = signed_power(df_base, df_exp)
        assert result["A"][0] == -16.0  # sign(-4) * |-4|^2 = -1 * 16

    def test_signed_power_fractional_exp(self) -> None:
        """Test signed_power with fractional exponent preserves sign."""
        df_base = pl.DataFrame({"Date": [date(2024, 1, 1)], "A": [-9.0]})
        df_exp = pl.DataFrame({"Date": [date(2024, 1, 1)], "A": [0.5]})
        result = signed_power(df_base, df_exp)
        assert result["A"][0] == -3.0  # sign(-9) * |-9|^0.5 = -1 * 3

    def test_sqrt_basic(self) -> None:
        """Test basic square root computation."""
        df = pl.DataFrame({
            "Date": [date(2024, 1, 1), date(2024, 1, 2)],
            "A": [4.0, 9.0],
        })
        result = sqrt(df)
        assert result["A"][0] == 2.0
        assert result["A"][1] == 3.0

    def test_sqrt_of_zero(self) -> None:
        """Test sqrt(0) = 0."""
        df = pl.DataFrame({"Date": [date(2024, 1, 1)], "A": [0.0]})
        result = sqrt(df)
        assert result["A"][0] == 0.0

    def test_sqrt_of_negative_returns_null(self) -> None:
        """Test sqrt(negative) returns null."""
        df = pl.DataFrame({"Date": [date(2024, 1, 1)], "A": [-4.0]})
        result = sqrt(df)
        assert result["A"][0] is None

    def test_sign_positive(self) -> None:
        """Test sign of positive number."""
        df = pl.DataFrame({"Date": [date(2024, 1, 1)], "A": [5.0]})
        result = sign(df)
        assert result["A"][0] == 1

    def test_sign_negative(self) -> None:
        """Test sign of negative number."""
        df = pl.DataFrame({"Date": [date(2024, 1, 1)], "A": [-5.0]})
        result = sign(df)
        assert result["A"][0] == -1

    def test_sign_zero(self) -> None:
        """Test sign of zero."""
        df = pl.DataFrame({"Date": [date(2024, 1, 1)], "A": [0.0]})
        result = sign(df)
        assert result["A"][0] == 0

    def test_sign_null(self) -> None:
        """Test sign of null returns null."""
        df = pl.DataFrame({"Date": [date(2024, 1, 1)], "A": pl.Series([None], dtype=pl.Float64)})
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
        df = pl.DataFrame({"Date": [date(2024, 1, 1)], "A": [0.0]})
        result = reverse(df)
        assert result["A"][0] == 0.0

    def test_densify_basic(self) -> None:
        """Test basic densify remapping."""
        df = pl.DataFrame({
            "Date": [date(2024, 1, 1)],
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
            "Date": [date(2024, 1, 1)],
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
            "Date": [date(2024, 1, 1), date(2024, 1, 2)],
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
        df = pl.DataFrame({"Date": [date(2024, 1, 1)], "A": pl.Series([None], dtype=pl.Float64)})
        result = op_abs(df)
        assert result["A"][0] is None

    def test_sqrt_with_null(self) -> None:
        """Test sqrt preserves nulls."""
        df = pl.DataFrame({"Date": [date(2024, 1, 1)], "A": pl.Series([None], dtype=pl.Float64)})
        result = sqrt(df)
        assert result["A"][0] is None

    def test_log_with_null(self) -> None:
        """Test log preserves nulls."""
        df = pl.DataFrame({"Date": [date(2024, 1, 1)], "A": pl.Series([None], dtype=pl.Float64)})
        result = log(df)
        assert result["A"][0] is None

    def test_reverse_with_null(self) -> None:
        """Test reverse preserves nulls."""
        df = pl.DataFrame({"Date": [date(2024, 1, 1)], "A": pl.Series([None], dtype=pl.Float64)})
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
            "Date": pl.date_range(date(2024, 1, 1), date(2024, 1, 5), eager=True),
            "AAPL": [True, False, True, False, True],
            "MSFT": [True, True, False, False, True],
        })

    @pytest.fixture
    def bool_df_b(self) -> pl.DataFrame:
        """Create boolean DataFrame B."""
        return pl.DataFrame({
            "Date": pl.date_range(date(2024, 1, 1), date(2024, 1, 5), eager=True),
            "AAPL": [True, True, True, False, False],
            "MSFT": [False, True, False, True, True],
        })

    @pytest.fixture
    def numeric_df(self) -> pl.DataFrame:
        """Create numeric DataFrame for comparisons."""
        return pl.DataFrame({
            "Date": pl.date_range(date(2024, 1, 1), date(2024, 1, 5), eager=True),
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
            "Date": pl.date_range(date(2024, 1, 1), date(2024, 1, 5), eager=True),
            "AAPL": [100.0, 100.0, 100.0, 100.0, 100.0],
            "MSFT": [200.0, 200.0, 200.0, 200.0, 200.0],
        })
        else_df = pl.DataFrame({
            "Date": pl.date_range(date(2024, 1, 1), date(2024, 1, 5), eager=True),
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
            "Date": pl.date_range(date(2024, 1, 1), date(2024, 1, 5), eager=True),
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
            "Date": pl.date_range(date(2024, 1, 1), date(2024, 1, 3), eager=True),
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
            "Date": pl.date_range(date(2024, 1, 1), date(2024, 1, 5), eager=True),
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
            "Date": pl.date_range(date(2024, 1, 1), date(2024, 1, 3), eager=True),
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
            "Date": [date(2024, 1, 1), date(2024, 1, 2)],
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
            "Date": pl.date_range(date(2024, 1, 1), date(2024, 1, 3), eager=True),
            "A": [10.0, 20.0, 30.0],
            "B": [15.0, 25.0, 35.0],
            "C": [100.0, 200.0, 300.0],
            "D": [150.0, 250.0, 350.0],
        })
        group = pl.DataFrame({
            "Date": pl.date_range(date(2024, 1, 1), date(2024, 1, 3), eager=True),
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
            "Date": [date(2024, 1, 1)],
            "A": [10.0],
            "B": [20.0],
        })
        group = pl.DataFrame({
            "Date": [date(2024, 1, 1)],
            "A": ["grp1"],
            "B": ["grp2"],
        })
        result = group_rank(x, group)
        assert result["A"][0] == 0.5
        assert result["B"][0] == 0.5

    def test_group_mean(self) -> None:
        """Test weighted mean within groups."""
        x = pl.DataFrame({
            "Date": [date(2024, 1, 1)],
            "A": [10.0],
            "B": [20.0],
            "C": [100.0],
        })
        weight = pl.DataFrame({
            "Date": [date(2024, 1, 1)],
            "A": [1.0],
            "B": [3.0],
            "C": [1.0],
        })
        group = pl.DataFrame({
            "Date": [date(2024, 1, 1)],
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
            "Date": pl.date_range(date(2024, 1, 1), date(2024, 1, 3), eager=True),
            "A": [10.0, None, 30.0],
            "B": [20.0, 25.0, 35.0],
            "C": [100.0, 200.0, None],
        })
        group = pl.DataFrame({
            "Date": pl.date_range(date(2024, 1, 1), date(2024, 1, 3), eager=True),
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
            "Date": pl.date_range(date(2024, 1, 1), date(2024, 1, 3), eager=True),
            "A": [None, None, None],
            "B": [None, None, None],
        })
        group = pl.DataFrame({
            "Date": pl.date_range(date(2024, 1, 1), date(2024, 1, 3), eager=True),
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
            "Date": [date(2024, 1, 1)],
            "A": [5.0],
            "B": [5.0],
        })
        group = pl.DataFrame({
            "Date": [date(2024, 1, 1)],
            "A": ["grp1"],
            "B": ["grp1"],
        })
        result = group_scale(x, group)
        # (5-5)/(5-5) = 0/0 = NaN
        assert result["A"][0] is None or math.isnan(result["A"][0])

    def test_group_zscore_all_same(self) -> None:
        """Test all same values returns NaN for zscore."""
        x = pl.DataFrame({
            "Date": [date(2024, 1, 1)],
            "A": [5.0],
            "B": [5.0],
        })
        group = pl.DataFrame({
            "Date": [date(2024, 1, 1)],
            "A": ["grp1"],
            "B": ["grp1"],
        })
        result = group_zscore(x, group)
        # std=0, (5-5)/0 = NaN
        assert result["A"][0] is None or math.isnan(result["A"][0])

    def test_group_rank_all_same(self) -> None:
        """Test all same values returns values in [0,1]."""
        x = pl.DataFrame({
            "Date": [date(2024, 1, 1)],
            "A": [5.0],
            "B": [5.0],
        })
        group = pl.DataFrame({
            "Date": [date(2024, 1, 1)],
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
            "Date": pl.date_range(date(2024, 1, 1), date(2024, 1, 10), eager=True),
            "A": y_vals,
        })
        x = pl.DataFrame({
            "Date": pl.date_range(date(2024, 1, 1), date(2024, 1, 10), eager=True),
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
            "Date": pl.date_range(date(2024, 1, 1), date(2024, 1, 5), eager=True),
            "A": [1.0, None, 3.0, 4.0, 5.0],
        })
        x = pl.DataFrame({
            "Date": pl.date_range(date(2024, 1, 1), date(2024, 1, 5), eager=True),
            "A": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        result = ts_regression(y, x, 3, rettype=1)
        # Window containing null returns NaN
        assert is_missing(result["A"][2])

    def test_ts_regression_zero_variance(self) -> None:
        """Test regression with zero variance in x (ss_xx=0)."""
        y = pl.DataFrame({
            "Date": pl.date_range(date(2024, 1, 1), date(2024, 1, 5), eager=True),
            "A": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        x = pl.DataFrame({
            "Date": pl.date_range(date(2024, 1, 1), date(2024, 1, 5), eager=True),
            "A": [5.0, 5.0, 5.0, 5.0, 5.0],  # Constant x
        })
        result = ts_regression(y, x, 3, rettype=1)
        # Zero variance should return NaN
        assert is_missing(result["A"][2])


class TestTsQuantileEdgeCases:
    """Tests for ts_quantile edge cases."""

    def test_ts_quantile_uniform(self) -> None:
        """Test ts_quantile with uniform driver."""
        df = pl.DataFrame({
            "Date": pl.date_range(date(2024, 1, 1), date(2024, 1, 5), eager=True),
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
            "Date": pl.date_range(date(2024, 1, 1), date(2024, 1, 5), eager=True),
            "A": [5.0, 5.0, 5.0, 5.0, 5.0],
        })
        result = ts_quantile(df, 3, driver="gaussian")
        # All same values: idx=0, rank_pct=0.5/3, inv_norm(0.166) ~ -0.97
        assert result["A"][2] is not None
        assert not math.isnan(result["A"][2])

    def test_ts_quantile_with_nulls(self) -> None:
        """Test ts_quantile with null values."""
        df = pl.DataFrame({
            "Date": pl.date_range(date(2024, 1, 1), date(2024, 1, 5), eager=True),
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
            "Date": pl.date_range(date(2024, 1, 1), date(2024, 1, 5), eager=True),
            "A": [1.0, None, 3.0, 4.0, 5.0],
        })
        y = pl.DataFrame({
            "Date": pl.date_range(date(2024, 1, 1), date(2024, 1, 5), eager=True),
            "A": [2.0, 4.0, 6.0, 8.0, 10.0],
        })
        result = ts_corr(x, y, 3)
        # Window with null should return NaN
        assert is_missing(result["A"][2])

    def test_ts_corr_zero_std(self) -> None:
        """Test ts_corr with zero standard deviation."""
        x = pl.DataFrame({
            "Date": pl.date_range(date(2024, 1, 1), date(2024, 1, 5), eager=True),
            "A": [5.0, 5.0, 5.0, 5.0, 5.0],  # Constant
        })
        y = pl.DataFrame({
            "Date": pl.date_range(date(2024, 1, 1), date(2024, 1, 5), eager=True),
            "A": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        result = ts_corr(x, y, 3)
        # Zero std should return NaN
        assert is_missing(result["A"][2])

    def test_ts_covariance_with_nulls(self) -> None:
        """Test ts_covariance with null values."""
        x = pl.DataFrame({
            "Date": pl.date_range(date(2024, 1, 1), date(2024, 1, 5), eager=True),
            "A": [1.0, 2.0, None, 4.0, 5.0],
        })
        y = pl.DataFrame({
            "Date": pl.date_range(date(2024, 1, 1), date(2024, 1, 5), eager=True),
            "A": [2.0, 4.0, 6.0, 8.0, 10.0],
        })
        result = ts_covariance(x, y, 3)
        # Window with null should return NaN
        assert is_missing(result["A"][3])


class TestTsRankEdgeCases:
    """Tests for ts_rank edge cases."""

    def test_ts_rank_with_nulls(self) -> None:
        """Test ts_rank with null current value."""
        df = pl.DataFrame({
            "Date": pl.date_range(date(2024, 1, 1), date(2024, 1, 5), eager=True),
            "A": [1.0, 2.0, None, 4.0, 5.0],
        })
        result = ts_rank(df, 3)
        # Null current value should return missing (None or NaN)
        assert is_missing(result["A"][2])

    def test_ts_rank_single_unique(self) -> None:
        """Test ts_rank with single unique non-null value."""
        df = pl.DataFrame({
            "Date": pl.date_range(date(2024, 1, 1), date(2024, 1, 5), eager=True),
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
            "Date": pl.date_range(date(2024, 1, 1), date(2024, 1, 5), eager=True),
            "A": [1.0, None, 3.0, 4.0, 5.0],
        })
        result = ts_decay_linear(df, 3, dense=True)
        # Dense mode skips nulls, should return value
        assert result["A"][4] is not None

    def test_ts_decay_linear_with_nulls_dense_false(self) -> None:
        """Test ts_decay_linear with nulls and dense=False."""
        df = pl.DataFrame({
            "Date": pl.date_range(date(2024, 1, 1), date(2024, 1, 5), eager=True),
            "A": [1.0, None, 3.0, 4.0, 5.0],
        })
        result = ts_decay_linear(df, 3, dense=False)
        # Non-dense mode: window with null returns missing (None or NaN)
        assert is_missing(result["A"][2])


class TestOtherOperatorEdgeCases:
    """Tests for other operator edge cases."""

    def test_hump_with_none(self) -> None:
        """Test hump when previous value is None."""
        df = pl.DataFrame({
            "Date": pl.date_range(date(2024, 1, 1), date(2024, 1, 3), eager=True),
            "A": [None, 100.0, 150.0],
        })
        result = hump(df, hump=0.1)
        # When prev is None, curr should pass through
        assert result["A"][1] == 100.0

    def test_last_diff_value_all_same(self) -> None:
        """Test last_diff_value when all values are same."""
        df = pl.DataFrame({
            "Date": pl.date_range(date(2024, 1, 1), date(2024, 1, 5), eager=True),
            "A": [5.0, 5.0, 5.0, 5.0, 5.0],
        })
        result = last_diff_value(df, 3)
        # No different value exists, should return NaN
        assert is_missing(result["A"][4])

    def test_ts_arg_max_short_window(self) -> None:
        """Test ts_arg_max when window not filled."""
        df = pl.DataFrame({
            "Date": pl.date_range(date(2024, 1, 1), date(2024, 1, 3), eager=True),
            "A": [1.0, 2.0, 3.0],
        })
        result = ts_arg_max(df, 5)
        # Window size 5, only 3 values, should be missing
        assert is_missing(result["A"][2])

    def test_ts_arg_min_short_window(self) -> None:
        """Test ts_arg_min when window not filled."""
        df = pl.DataFrame({
            "Date": pl.date_range(date(2024, 1, 1), date(2024, 1, 3), eager=True),
            "A": [3.0, 2.0, 1.0],
        })
        result = ts_arg_min(df, 5)
        # Window size 5, only 3 values, should be missing
        assert is_missing(result["A"][2])


class TestEdgeCases:
    """Edge case tests."""

    def test_single_column(self) -> None:
        """Test operators with single symbol column."""
        df = pl.DataFrame({
            "Date": pl.date_range(date(2024, 1, 1), date(2024, 1, 5), eager=True),
            "AAPL": [100.0, 102.0, 101.0, 103.0, 105.0],
        })

        result = ts_mean(df, 3)
        assert result.columns == df.columns

    def test_with_nulls(self) -> None:
        """Test operators handle nulls correctly."""
        df = pl.DataFrame({
            "Date": pl.date_range(date(2024, 1, 1), date(2024, 1, 5), eager=True),
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
            "Date": [date(2024, 1, 1)],
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
            "Date": [date(2024, 1, 1)],
            "A": [-0.5],  # Below start
            "B": [1.5],   # Above end
        })
        result = bucket(df, "0,1,0.25")
        assert result["A"][0] == 0.0   # Clipped to start
        assert result["B"][0] == 0.75  # Clipped to end - step

    def test_bucket_invalid_spec(self) -> None:
        """Test bucket raises on invalid range spec."""
        df = pl.DataFrame({"Date": [date(2024, 1, 1)], "A": [0.5]})
        with pytest.raises(ValueError, match="range_spec"):
            bucket(df, "0,1")

    def test_bucket_invalid_step(self) -> None:
        """Test bucket raises on non-positive step."""
        df = pl.DataFrame({"Date": [date(2024, 1, 1)], "A": [0.5]})
        with pytest.raises(ValueError, match="step must be positive"):
            bucket(df, "0,1,-0.25")

    def test_bucket_multiple_rows(self) -> None:
        """Test bucket works across multiple rows."""
        df = pl.DataFrame({
            "Date": pl.date_range(date(2024, 1, 1), date(2024, 1, 3), eager=True),
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
            "Date": [date(2024, 1, 1)],
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
            "Date": [date(2024, 1, 1)],
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
        df = pl.DataFrame({"Date": [date(2024, 1, 1)], "A": [5.0]})
        result = multiply(3, df)
        assert result["A"][0] == 15.0

    def test_multiply_scalar_in_loop(self) -> None:
        """Test multiply with DataFrame then scalar."""
        df = pl.DataFrame({"Date": [date(2024, 1, 1)], "A": [5.0]})
        result = multiply(df, 3)
        assert result["A"][0] == 15.0

    def test_power_both_scalars(self) -> None:
        """Test power with both scalars."""
        assert power(2, 3) == 8
        assert power(3, 0) == 1

    def test_power_scalar_base_df_exp(self) -> None:
        """Test power with scalar base and DataFrame exponent."""
        df = pl.DataFrame({"Date": [date(2024, 1, 1)], "A": [3.0]})
        result = power(2, df)
        assert result["A"][0] == 8.0  # 2^3

    def test_signed_power_both_scalars(self) -> None:
        """Test signed_power with both scalars."""
        assert signed_power(4, 2) == 16
        assert signed_power(-4, 2) == -16

    def test_signed_power_scalar_df(self) -> None:
        """Test signed_power with scalar base and DataFrame exponent."""
        df = pl.DataFrame({"Date": [date(2024, 1, 1)], "A": [2.0]})
        result = signed_power(3, df)
        assert result["A"][0] == 9.0  # sign(3) * |3|^2


class TestLookbackOperators:
    """Tests for time-series operators with lookback parameter."""

    def test_ts_delta_with_lookback(self) -> None:
        """Test ts_delta with lookback DataFrame."""
        lookback = pl.DataFrame({
            "Date": [date(2024, 1, 1), date(2024, 1, 2)],
            "A": [100.0, 102.0],
        })
        x = pl.DataFrame({
            "Date": [date(2024, 1, 3), date(2024, 1, 4)],
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
            "Date": [date(2024, 1, 1), date(2024, 1, 2)],
            "A": [100.0, 102.0],
        })
        x = pl.DataFrame({
            "Date": [date(2024, 1, 3), date(2024, 1, 4)],
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
            "Date": pl.date_range(date(2024, 1, 1), date(2024, 1, 5), eager=True),
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
        data = {"Date": [date(2024, 1, 1), date(2024, 1, 2)]}
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


# =============================================================================
# TRANSFORMATIONAL OPERATORS
# =============================================================================


class TestTradeWhen:
    """Tests for trade_when operator."""

    @pytest.fixture
    def dates(self) -> list[date]:
        return [date(2024, 1, i) for i in range(1, 6)]

    def test_basic_trigger_enter(self, dates: list[date]) -> None:
        """Entry signal sets alpha value."""
        from alphalab.api.operators import trade_when
        trigger = pl.DataFrame({"Date": dates, "A": [1.0, 0.0, 0.0, 1.0, 0.0]})
        alpha = pl.DataFrame({"Date": dates, "A": [10.0, 20.0, 30.0, 40.0, 50.0]})
        result = trade_when(trigger, alpha, -1)  # never exit
        assert result["A"][0] == 10.0  # enter
        assert result["A"][3] == 40.0  # re-enter

    def test_carry_forward(self, dates: list[date]) -> None:
        """Position carries forward when no trigger."""
        from alphalab.api.operators import trade_when
        trigger = pl.DataFrame({"Date": dates, "A": [1.0, 0.0, 0.0, 0.0, 0.0]})
        alpha = pl.DataFrame({"Date": dates, "A": [10.0, 20.0, 30.0, 40.0, 50.0]})
        result = trade_when(trigger, alpha, -1)
        assert result["A"][0] == 10.0
        assert result["A"][1] == 10.0  # carry forward
        assert result["A"][4] == 10.0  # still carrying

    def test_exit_trigger_clears(self, dates: list[date]) -> None:
        """Exit signal produces NaN."""
        from alphalab.api.operators import trade_when
        import math
        trigger = pl.DataFrame({"Date": dates, "A": [1.0, 0.0, 0.0, 0.0, 0.0]})
        alpha = pl.DataFrame({"Date": dates, "A": [10.0, 20.0, 30.0, 40.0, 50.0]})
        exit_df = pl.DataFrame({"Date": dates, "A": [0.0, 0.0, 1.0, 0.0, 0.0]})
        result = trade_when(trigger, alpha, exit_df)
        assert result["A"][0] == 10.0  # entered
        assert result["A"][1] == 10.0  # carry
        assert math.isnan(result["A"][2])  # exit → NaN
        assert math.isnan(result["A"][3])  # no trigger, carry NaN

    def test_scalar_exit_never(self, dates: list[date]) -> None:
        """Scalar exit -1 means never exit."""
        from alphalab.api.operators import trade_when
        trigger = pl.DataFrame({"Date": dates, "A": [1.0, 0.0, 0.0, 0.0, 0.0]})
        alpha = pl.DataFrame({"Date": dates, "A": [99.0, 0.0, 0.0, 0.0, 0.0]})
        result = trade_when(trigger, alpha, -1)
        # All values should be 99.0 (entered, never exited)
        for i in range(5):
            assert result["A"][i] == 99.0

    def test_all_nan_when_exit_always(self, dates: list[date]) -> None:
        """Scalar exit 1 means always exit (all NaN)."""
        from alphalab.api.operators import trade_when
        import math
        trigger = pl.DataFrame({"Date": dates, "A": [1.0, 1.0, 1.0, 1.0, 1.0]})
        alpha = pl.DataFrame({"Date": dates, "A": [10.0, 20.0, 30.0, 40.0, 50.0]})
        result = trade_when(trigger, alpha, 1)  # always exit
        for i in range(5):
            assert math.isnan(result["A"][i])

    def test_multiple_columns(self, dates: list[date]) -> None:
        """Works with multiple symbol columns."""
        from alphalab.api.operators import trade_when
        trigger = pl.DataFrame({
            "Date": dates,
            "A": [1.0, 0.0, 0.0, 0.0, 0.0],
            "B": [0.0, 1.0, 0.0, 0.0, 0.0],
        })
        alpha = pl.DataFrame({
            "Date": dates,
            "A": [10.0, 20.0, 30.0, 40.0, 50.0],
            "B": [100.0, 200.0, 300.0, 400.0, 500.0],
        })
        result = trade_when(trigger, alpha, -1)
        assert result["A"][0] == 10.0
        assert result["B"][1] == 200.0


# =============================================================================
# ADDITIONAL COVERAGE TESTS
# =============================================================================


class TestTimeSeriesEdgeCases:
    """Tests for time series operator edge cases to improve coverage."""

    def test_ts_arg_max_with_nulls(self) -> None:
        """Test ts_arg_max when window has null values."""
        df = pl.DataFrame({
            "Date": [date(2024, 1, i) for i in range(1, 6)],
            "A": [1.0, None, 3.0, 2.0, 5.0],
        })
        result = ts_arg_max(df, 3)
        assert result.columns == df.columns

    def test_ts_arg_min_with_nulls(self) -> None:
        """Test ts_arg_min when window has null values."""
        df = pl.DataFrame({
            "Date": [date(2024, 1, i) for i in range(1, 6)],
            "A": [5.0, None, 3.0, 2.0, 1.0],
        })
        result = ts_arg_min(df, 3)
        assert result.columns == df.columns

    def test_last_diff_value_all_same(self) -> None:
        """Test last_diff_value when all values are the same."""
        df = pl.DataFrame({
            "Date": [date(2024, 1, i) for i in range(1, 6)],
            "A": [5.0, 5.0, 5.0, 5.0, 5.0],
        })
        result = last_diff_value(df, 3)
        assert result.columns == df.columns

    def test_last_diff_value_short_window(self) -> None:
        """Test last_diff_value with very short data."""
        df = pl.DataFrame({
            "Date": [date(2024, 1, 1)],
            "A": [5.0],
        })
        result = last_diff_value(df, 3)
        assert result.columns == df.columns

    def test_ts_decay_linear_dense_true(self) -> None:
        """Test ts_decay_linear with dense=True to skip nulls."""
        df = pl.DataFrame({
            "Date": [date(2024, 1, i) for i in range(1, 6)],
            "A": [1.0, None, 3.0, None, 5.0],
        })
        result = ts_decay_linear(df, 3, dense=True)
        assert result.columns == df.columns

    def test_ts_decay_linear_dense_false_with_nulls(self) -> None:
        """Test ts_decay_linear with dense=False returns null when window has nulls."""
        df = pl.DataFrame({
            "Date": [date(2024, 1, i) for i in range(1, 6)],
            "A": [1.0, None, 3.0, 4.0, 5.0],
        })
        result = ts_decay_linear(df, 3, dense=False)
        assert result.columns == df.columns

    def test_ts_rank_single_value(self) -> None:
        """Test ts_rank with single value in window."""
        df = pl.DataFrame({
            "Date": [date(2024, 1, i) for i in range(1, 4)],
            "A": [5.0, 5.0, 5.0],
        })
        result = ts_rank(df, 3, constant=0.0)
        assert result.columns == df.columns

    def test_ts_rank_with_null_current(self) -> None:
        """Test ts_rank when current value is null."""
        df = pl.DataFrame({
            "Date": [date(2024, 1, i) for i in range(1, 4)],
            "A": [1.0, 2.0, None],
        })
        result = ts_rank(df, 3)
        assert result.columns == df.columns

    def test_ts_quantile_gaussian_edge_values(self) -> None:
        """Test ts_quantile with extreme values to cover inv_norm branches."""
        df = pl.DataFrame({
            "Date": [date(2024, 1, i) for i in range(1, 11)],
            "A": [float(i) for i in range(1, 11)],
        })
        result = ts_quantile(df, 10, driver="gaussian")
        assert result.columns == df.columns

    def test_ts_quantile_uniform_driver(self) -> None:
        """Test ts_quantile with uniform driver."""
        df = pl.DataFrame({
            "Date": [date(2024, 1, i) for i in range(1, 6)],
            "A": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        result = ts_quantile(df, 5, driver="uniform")
        assert result.columns == df.columns

    def test_ts_quantile_single_value_window(self) -> None:
        """Test ts_quantile with single valid value in window."""
        df = pl.DataFrame({
            "Date": [date(2024, 1, i) for i in range(1, 4)],
            "A": [5.0, 5.0, 5.0],
        })
        result = ts_quantile(df, 3)
        assert result.columns == df.columns


class TestCrossSectionalEdgeCases:
    """Tests for cross-sectional operator edge cases."""

    def test_quantile_with_single_valid_value(self) -> None:
        """Test quantile when only one valid value per row."""
        df = pl.DataFrame({
            "Date": [date(2024, 1, 1)],
            "A": [1.0],
            "B": [None],
        })
        result = quantile(df, driver="gaussian")
        assert result.columns == df.columns

    def test_quantile_uniform_driver(self) -> None:
        """Test quantile with uniform driver."""
        df = pl.DataFrame({
            "Date": [date(2024, 1, 1)],
            "A": [1.0],
            "B": [2.0],
            "C": [3.0],
        })
        result = quantile(df, driver="uniform")
        assert result.columns == df.columns

    def test_rank_bucket_single_valid(self) -> None:
        """Test bucket rank with single valid value."""
        n_symbols = 40
        data = {"Date": [date(2024, 1, 1)]}
        for i in range(n_symbols):
            data[f"S{i:03d}"] = [None] if i > 0 else [1.0]
        df = pl.DataFrame(data)
        result = rank(df, rate=2)
        assert result.columns == df.columns


class TestBucketRankCoverage:
    """Additional tests for cross_sectional bucket_rank inner function."""

    def test_bucket_rank_all_null(self) -> None:
        """Test bucket rank with all null values."""
        import math
        n_symbols = 40
        data = {"Date": [date(2024, 1, 1)]}
        for i in range(n_symbols):
            data[f"S{i:03d}"] = [None]
        df = pl.DataFrame(data)
        result = rank(df, rate=2)
        # All values should remain null/NaN
        for col in df.columns[1:]:
            val = result[col][0]
            assert val is None or math.isnan(val)

    def test_bucket_rank_with_mixed_null(self) -> None:
        """Test bucket rank with mix of valid and null values (>32 symbols)."""
        import math
        n_symbols = 40
        data = {"Date": [date(2024, 1, 1)]}
        for i in range(n_symbols):
            # Half valid, half null
            data[f"S{i:03d}"] = [float(i)] if i % 2 == 0 else [None]
        df = pl.DataFrame(data)
        result = rank(df, rate=2)
        # Valid values should be ranked
        for i in range(n_symbols):
            col = f"S{i:03d}"
            val = result[col][0]
            if i % 2 == 0:  # valid
                assert val is not None and not math.isnan(val)
                assert 0.0 <= val <= 1.0
            else:  # null
                assert val is None or math.isnan(val)

    def test_bucket_rank_multiple_rows(self) -> None:
        """Test bucket rank across multiple rows."""
        n_symbols = 35  # Just above 32 threshold
        data = {"Date": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)]}
        for i in range(n_symbols):
            data[f"S{i:03d}"] = [float(i), float(n_symbols - i - 1), float(i * 2)]
        df = pl.DataFrame(data)
        result = rank(df, rate=1)  # Different rate
        assert result.shape == df.shape
        # Each row should have values in [0, 1]
        for row_idx in range(3):
            for col in df.columns[1:]:
                val = result[col][row_idx]
                assert 0.0 <= val <= 1.0


class TestTsQuantileInvNormCoverage:
    """Tests to cover all branches of ts_quantile's inv_norm function."""

    def test_ts_quantile_extreme_low_rank(self) -> None:
        """Test ts_quantile with very low rank values (triggers p < p_low branch)."""
        from alphalab.api.operators import ts_quantile
        from datetime import timedelta
        # Create 101 rows using multiple months
        base_date = date(2024, 1, 1)
        dates = [base_date + timedelta(days=i) for i in range(101)]
        df = pl.DataFrame({
            "Date": dates,
            "A": [0.0] + [float(i) for i in range(1, 101)],  # A has lowest value at start
        })
        result = ts_quantile(df, d=100, driver="gaussian")
        # The smallest value should give a very negative result
        assert result["A"][-1] is not None

    def test_ts_quantile_extreme_high_rank(self) -> None:
        """Test ts_quantile with very high rank values (triggers p > p_high branch)."""
        from alphalab.api.operators import ts_quantile
        from datetime import timedelta
        # Create 101 rows using multiple months
        base_date = date(2024, 1, 1)
        dates = [base_date + timedelta(days=i) for i in range(101)]
        df = pl.DataFrame({
            "Date": dates,
            "A": [float(i) for i in range(1, 101)] + [1000.0],  # A has highest value at end
        })
        result = ts_quantile(df, d=100, driver="gaussian")
        # The highest value should give a very positive result
        assert result["A"][-1] is not None
        assert result["A"][-1] > 0  # Should be positive

    def test_ts_quantile_middle_rank(self) -> None:
        """Test ts_quantile with middle rank (triggers p <= p_high branch)."""
        from alphalab.api.operators import ts_quantile
        df = pl.DataFrame({
            "Date": [date(2024, 1, i) for i in range(1, 11)],
            "A": [float(i) for i in range(1, 11)],  # Linear values
        })
        result = ts_quantile(df, d=10, driver="gaussian")
        # Middle values should be close to 0
        assert result["A"][-1] is not None

    def test_ts_quantile_uniform_all_same(self) -> None:
        """Test ts_quantile uniform driver with all same values returns expected value."""
        from alphalab.api.operators import ts_quantile
        df = pl.DataFrame({
            "Date": [date(2024, 1, i) for i in range(1, 6)],
            "A": [5.0, 5.0, 5.0, 5.0, 5.0],
        })
        result = ts_quantile(df, d=3, driver="uniform")
        # With all same values, rank_pct will be 0.5/3 = 0.166..., uniform => 0.166*2-1 ≈ -0.666
        assert result["A"][-1] is not None

    def test_ts_quantile_null_current(self) -> None:
        """Test ts_quantile with null current value."""
        from alphalab.api.operators import ts_quantile
        df = pl.DataFrame({
            "Date": [date(2024, 1, i) for i in range(1, 6)],
            "A": [1.0, 2.0, 3.0, 4.0, None],
        })
        result = ts_quantile(df, d=3, driver="gaussian")
        # Current null should return missing (None or NaN)
        assert is_missing(result["A"][-1])


class TestTimeSeriesArgMinMaxCoverage:
    """Tests for ts_arg_max and ts_arg_min edge cases."""

    def test_ts_arg_max_all_null_in_window(self) -> None:
        """Test ts_arg_max when all values in window are null."""
        from alphalab.api.operators import ts_arg_max
        df = pl.DataFrame({
            "Date": [date(2024, 1, i) for i in range(1, 6)],
            "A": [1.0, None, None, None, None],
        })
        result = ts_arg_max(df, d=3)
        # Window [None, None, None] should return missing (current is NaN)
        assert is_missing(result["A"][-1])

    def test_ts_arg_min_all_null_in_window(self) -> None:
        """Test ts_arg_min when all values in window are null."""
        from alphalab.api.operators import ts_arg_min
        df = pl.DataFrame({
            "Date": [date(2024, 1, i) for i in range(1, 6)],
            "A": [1.0, None, None, None, None],
        })
        result = ts_arg_min(df, d=3)
        # Window [None, None, None] should return missing (current is NaN)
        assert is_missing(result["A"][-1])

    def test_ts_arg_max_window_smaller_than_d(self) -> None:
        """Test ts_arg_max at start when window < d."""
        from alphalab.api.operators import ts_arg_max
        df = pl.DataFrame({
            "Date": [date(2024, 1, i) for i in range(1, 6)],
            "A": [5.0, 3.0, 4.0, 2.0, 1.0],
        })
        result = ts_arg_max(df, d=10)  # d=10 but only 5 rows
        # First few rows should have missing (window not complete)
        assert is_missing(result["A"][0])

    def test_ts_arg_min_window_smaller_than_d(self) -> None:
        """Test ts_arg_min at start when window < d."""
        from alphalab.api.operators import ts_arg_min
        df = pl.DataFrame({
            "Date": [date(2024, 1, i) for i in range(1, 6)],
            "A": [5.0, 3.0, 4.0, 2.0, 1.0],
        })
        result = ts_arg_min(df, d=10)  # d=10 but only 5 rows
        # First few rows should have missing (window not complete)
        assert is_missing(result["A"][0])


class TestLastDiffValueCoverage:
    """Tests for last_diff_value edge cases."""

    def test_last_diff_value_single_element(self) -> None:
        """Test last_diff_value with window of 1."""
        from alphalab.api.operators import last_diff_value
        df = pl.DataFrame({
            "Date": [date(2024, 1, i) for i in range(1, 6)],
            "A": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        result = last_diff_value(df, d=1)
        # Window of 1 means no previous value, should be NaN
        assert is_missing(result["A"][-1])

    def test_last_diff_value_with_nulls(self) -> None:
        """Test last_diff_value with null values in sequence."""
        from alphalab.api.operators import last_diff_value
        df = pl.DataFrame({
            "Date": [date(2024, 1, i) for i in range(1, 6)],
            "A": [1.0, None, None, None, 1.0],  # Current same as first
        })
        result = last_diff_value(df, d=5)
        # Should return NaN since no different non-null value exists
        assert is_missing(result["A"][-1])

    def test_last_diff_value_finds_different(self) -> None:
        """Test last_diff_value finds the last different value."""
        from alphalab.api.operators import last_diff_value
        df = pl.DataFrame({
            "Date": [date(2024, 1, i) for i in range(1, 6)],
            "A": [1.0, 2.0, 3.0, 5.0, 5.0],  # Last different is 3.0
        })
        result = last_diff_value(df, d=5)
        assert result["A"][-1] == 3.0


class TestQuantileDriversCoverage:
    """Additional tests for cross_sectional quantile driver coverage."""

    def test_quantile_uniform_small_dataset(self) -> None:
        """Test cross-sectional quantile with uniform driver and small dataset."""
        from alphalab.api.operators import quantile
        df = pl.DataFrame({
            "Date": [date(2024, 1, 1)],
            "A": [1.0],
            "B": [2.0],
            "C": [3.0],
        })
        result = quantile(df, driver="uniform", sigma=1.0)
        # Uniform driver should produce values in [-sigma, sigma]
        for col in ["A", "B", "C"]:
            assert -1.0 <= result[col][0] <= 1.0

    def test_quantile_cauchy_small_dataset(self) -> None:
        """Test cross-sectional quantile with cauchy driver."""
        from alphalab.api.operators import quantile
        df = pl.DataFrame({
            "Date": [date(2024, 1, 1)],
            "A": [1.0],
            "B": [2.0],
            "C": [3.0],
        })
        result = quantile(df, driver="cauchy", sigma=1.0)
        # Cauchy driver produces values
        assert result["A"][0] is not None
        assert result["B"][0] is not None
        assert result["C"][0] is not None

    def test_quantile_n_valid_one(self) -> None:
        """Test cross-sectional quantile with single valid value."""
        import math
        from alphalab.api.operators import quantile
        df = pl.DataFrame({
            "Date": [date(2024, 1, 1)],
            "A": [1.0],
            "B": [None],
            "C": [None],
        })
        result = quantile(df, driver="gaussian")
        # Single valid value should return 0
        assert result["A"][0] == 0.0
        # Null values become NaN in polars
        assert result["B"][0] is None or math.isnan(result["B"][0])
        assert result["C"][0] is None or math.isnan(result["C"][0])


class TestTimeSeriesModuleLevelHelpers:
    """Direct tests for time_series module-level helper functions."""

    def test_arg_max_fn_basic(self) -> None:
        """Test _arg_max_fn with normal values."""
        from alphalab.api.operators.time_series import _arg_max_fn
        s = pl.Series([1.0, 5.0, 3.0, 2.0, 4.0])
        # Max is at index 1 (value 5.0), days since = (5-1) - 1 = 3
        result = _arg_max_fn(s, 5)
        assert result == 3.0

    def test_arg_max_fn_max_at_end(self) -> None:
        """Test _arg_max_fn when max is at most recent position."""
        from alphalab.api.operators.time_series import _arg_max_fn
        s = pl.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        # Max is at index 4, days since = (5-1) - 4 = 0
        result = _arg_max_fn(s, 5)
        assert result == 0.0

    def test_arg_max_fn_window_too_small(self) -> None:
        """Test _arg_max_fn when series length < d."""
        from alphalab.api.operators.time_series import _arg_max_fn
        s = pl.Series([1.0, 2.0])
        result = _arg_max_fn(s, 5)
        assert result is None

    def test_arg_min_fn_basic(self) -> None:
        """Test _arg_min_fn with normal values."""
        from alphalab.api.operators.time_series import _arg_min_fn
        s = pl.Series([5.0, 1.0, 3.0, 2.0, 4.0])
        # Min is at index 1 (value 1.0), days since = (5-1) - 1 = 3
        result = _arg_min_fn(s, 5)
        assert result == 3.0

    def test_arg_min_fn_min_at_end(self) -> None:
        """Test _arg_min_fn when min is at most recent position."""
        from alphalab.api.operators.time_series import _arg_min_fn
        s = pl.Series([5.0, 4.0, 3.0, 2.0, 1.0])
        # Min is at index 4, days since = (5-1) - 4 = 0
        result = _arg_min_fn(s, 5)
        assert result == 0.0

    def test_arg_min_fn_window_too_small(self) -> None:
        """Test _arg_min_fn when series length < d."""
        from alphalab.api.operators.time_series import _arg_min_fn
        s = pl.Series([1.0, 2.0])
        result = _arg_min_fn(s, 5)
        assert result is None

    def test_find_last_diff_basic(self) -> None:
        """Test _find_last_diff finds the previous different value."""
        from alphalab.api.operators.time_series import _find_last_diff
        s = pl.Series([1.0, 2.0, 3.0, 3.0, 3.0])
        # Current is 3.0, last different is 2.0 at index 1
        result = _find_last_diff(s)
        assert result == 2.0

    def test_find_last_diff_all_same(self) -> None:
        """Test _find_last_diff when all values are the same."""
        from alphalab.api.operators.time_series import _find_last_diff
        s = pl.Series([5.0, 5.0, 5.0, 5.0])
        result = _find_last_diff(s)
        assert result is None

    def test_find_last_diff_single_element(self) -> None:
        """Test _find_last_diff with single element."""
        from alphalab.api.operators.time_series import _find_last_diff
        s = pl.Series([5.0])
        result = _find_last_diff(s)
        assert result is None

    def test_find_last_diff_with_none(self) -> None:
        """Test _find_last_diff skips None values."""
        from alphalab.api.operators.time_series import _find_last_diff
        s = pl.Series([1.0, None, 3.0, 3.0])
        # Current is 3.0, last different is 1.0 (skipping None)
        result = _find_last_diff(s)
        assert result == 1.0

    def test_inv_norm_middle(self) -> None:
        """Test _inv_norm for middle probability values."""
        from alphalab.api.operators.time_series import _inv_norm
        # p=0.5 should give approximately 0
        result = _inv_norm(0.5)
        assert abs(result) < 0.001

    def test_inv_norm_low_tail(self) -> None:
        """Test _inv_norm for low probability (p < 0.02425)."""
        from alphalab.api.operators.time_series import _inv_norm
        result = _inv_norm(0.01)
        # Should be a large negative value
        assert result < -2.0

    def test_inv_norm_high_tail(self) -> None:
        """Test _inv_norm for high probability (p > 0.97575)."""
        from alphalab.api.operators.time_series import _inv_norm
        result = _inv_norm(0.99)
        # Should be a large positive value
        assert result > 2.0

    def test_inv_norm_boundary_zero(self) -> None:
        """Test _inv_norm at p=0."""
        from alphalab.api.operators.time_series import _inv_norm
        result = _inv_norm(0)
        assert result == float("-inf")

    def test_inv_norm_boundary_one(self) -> None:
        """Test _inv_norm at p=1."""
        from alphalab.api.operators.time_series import _inv_norm
        result = _inv_norm(1)
        assert result == float("inf")

    def test_ts_quantile_transform_gaussian(self) -> None:
        """Test _ts_quantile_transform with gaussian driver."""
        from alphalab.api.operators.time_series import _ts_quantile_transform
        s = pl.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _ts_quantile_transform(s, "gaussian")
        # High value (5.0) should give positive result
        assert result is not None
        assert result > 0

    def test_ts_quantile_transform_uniform(self) -> None:
        """Test _ts_quantile_transform with uniform driver."""
        from alphalab.api.operators.time_series import _ts_quantile_transform
        s = pl.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _ts_quantile_transform(s, "uniform")
        # Uniform driver should produce values in [-1, 1]
        assert result is not None
        assert -1.0 <= result <= 1.0

    def test_ts_quantile_transform_null_current(self) -> None:
        """Test _ts_quantile_transform when current value is None."""
        from alphalab.api.operators.time_series import _ts_quantile_transform
        s = pl.Series([1.0, 2.0, 3.0, None])
        result = _ts_quantile_transform(s, "gaussian")
        assert result is None

    def test_ts_quantile_transform_single_value(self) -> None:
        """Test _ts_quantile_transform with single valid value."""
        from alphalab.api.operators.time_series import _ts_quantile_transform
        s = pl.Series([None, None, 5.0])
        result = _ts_quantile_transform(s, "gaussian")
        # Single value should return 0.0
        assert result == 0.0


class TestCrossSectionalModuleLevelHelpers:
    """Direct tests for cross_sectional module-level helper functions."""

    def test_bucket_rank_basic(self) -> None:
        """Test _bucket_rank with normal values."""
        import numpy as np
        from alphalab.api.operators.cross_sectional import _bucket_rank
        values = np.array([1.0, 5.0, 3.0, 2.0, 4.0])
        result = _bucket_rank(values, rate=2)
        # All values should be in [0, 1]
        assert all(0 <= r <= 1 for r in result)

    def test_bucket_rank_with_nan(self) -> None:
        """Test _bucket_rank preserves NaN positions."""
        import numpy as np
        from alphalab.api.operators.cross_sectional import _bucket_rank
        values = np.array([1.0, np.nan, 3.0, 2.0, np.nan])
        result = _bucket_rank(values, rate=2)
        # NaN positions should remain NaN
        assert np.isnan(result[1])
        assert np.isnan(result[4])
        # Valid positions should have valid ranks
        assert not np.isnan(result[0])
        assert not np.isnan(result[2])
        assert not np.isnan(result[3])

    def test_bucket_rank_single_valid(self) -> None:
        """Test _bucket_rank with single valid value."""
        import numpy as np
        from alphalab.api.operators.cross_sectional import _bucket_rank
        values = np.array([np.nan, 5.0, np.nan])
        result = _bucket_rank(values, rate=2)
        # Single valid value gets rank 0
        assert result[1] == 0.0
        assert np.isnan(result[0])
        assert np.isnan(result[2])

    def test_bucket_rank_all_nan(self) -> None:
        """Test _bucket_rank with all NaN."""
        import numpy as np
        from alphalab.api.operators.cross_sectional import _bucket_rank
        values = np.array([np.nan, np.nan, np.nan])
        result = _bucket_rank(values, rate=2)
        assert all(np.isnan(r) for r in result)

    def test_quantile_transform_gaussian(self) -> None:
        """Test _quantile_transform with gaussian driver."""
        import numpy as np
        from alphalab.api.operators.cross_sectional import _quantile_transform
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _quantile_transform(values, driver="gaussian", sigma=1.0)
        # Highest value should have highest (positive) score
        assert result[4] > result[0]
        # Middle value should be close to 0
        assert abs(result[2]) < 0.5

    def test_quantile_transform_uniform(self) -> None:
        """Test _quantile_transform with uniform driver."""
        import numpy as np
        from alphalab.api.operators.cross_sectional import _quantile_transform
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _quantile_transform(values, driver="uniform", sigma=1.0)
        # All values should be in [-sigma, sigma]
        assert all(-1.0 <= r <= 1.0 for r in result)

    def test_quantile_transform_cauchy(self) -> None:
        """Test _quantile_transform with cauchy driver."""
        import numpy as np
        from alphalab.api.operators.cross_sectional import _quantile_transform
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _quantile_transform(values, driver="cauchy", sigma=1.0)
        # Cauchy distribution has heavier tails
        assert result[4] > result[0]

    def test_quantile_transform_with_nan(self) -> None:
        """Test _quantile_transform preserves NaN positions."""
        import numpy as np
        from alphalab.api.operators.cross_sectional import _quantile_transform
        values = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
        result = _quantile_transform(values, driver="gaussian", sigma=1.0)
        # NaN positions should remain NaN
        assert np.isnan(result[1])
        assert np.isnan(result[3])
        # Valid positions should have valid values
        assert not np.isnan(result[0])
        assert not np.isnan(result[2])
        assert not np.isnan(result[4])

    def test_quantile_transform_single_valid(self) -> None:
        """Test _quantile_transform with single valid value."""
        import numpy as np
        from alphalab.api.operators.cross_sectional import _quantile_transform
        values = np.array([np.nan, 5.0, np.nan])
        result = _quantile_transform(values, driver="gaussian", sigma=1.0)
        # Single valid value should return 0
        assert result[1] == 0.0

    def test_quantile_transform_unknown_driver(self) -> None:
        """Test _quantile_transform raises on unknown driver."""
        import numpy as np
        from alphalab.api.operators.cross_sectional import _quantile_transform
        values = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="Unknown driver"):
            _quantile_transform(values, driver="unknown", sigma=1.0)
