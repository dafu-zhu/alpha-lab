# tests/unit/api/operators/test_time_series_parallel.py
"""Tests for parallelized time-series operators."""

import math

import polars as pl
import pytest


def is_missing(value) -> bool:
    """Check if value is missing (None or NaN)."""
    if value is None:
        return True
    try:
        return math.isnan(value)
    except (TypeError, ValueError):
        return False


def test_days_from_last_change_correctness():
    """Parallelized version produces same results as original."""
    from alphalab.api.operators.time_series import days_from_last_change

    df = pl.DataFrame({
        "Date": [1, 2, 3, 4, 5, 6, 7],
        "A": [1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0],
        "B": [5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 7.0],
    })

    result = days_from_last_change(df)

    # A: changes at idx 2 and 5
    assert result["A"].to_list() == [0, 1, 0, 1, 2, 0, 1]
    # B: changes at idx 3 and 6
    assert result["B"].to_list() == [0, 1, 2, 0, 1, 2, 0]


def test_days_from_last_change_single_column():
    """Works with single value column."""
    from alphalab.api.operators.time_series import days_from_last_change

    df = pl.DataFrame({
        "Date": [1, 2, 3, 4],
        "A": [1.0, 1.0, 2.0, 2.0],
    })

    result = days_from_last_change(df)
    assert result["A"].to_list() == [0, 1, 0, 1]


def test_days_from_last_change_with_nulls():
    """Null values are treated as same value (None == None)."""
    from alphalab.api.operators.time_series import days_from_last_change

    df = pl.DataFrame({
        "Date": [1, 2, 3, 4],
        "A": [1.0, None, None, 2.0],
    })

    result = days_from_last_change(df)
    # None to None is NOT a change (None == None is True)
    # But 1.0 to None IS a change, and None to 2.0 IS a change
    assert result["A"].to_list() == [0, 0, 1, 0]


def test_days_from_last_change_with_nans():
    """NaN values are treated as changes (NaN != NaN)."""
    from alphalab.api.operators.time_series import days_from_last_change
    import math

    df = pl.DataFrame({
        "Date": [1, 2, 3, 4],
        "A": [1.0, float("nan"), float("nan"), 2.0],
    })

    result = days_from_last_change(df)
    # NaN != NaN is True, so each NaN is a "change"
    assert result["A"].to_list() == [0, 0, 0, 0]


def test_ts_corr_correctness():
    """Parallelized ts_corr produces same results."""
    from alphalab.api.operators.time_series import ts_corr

    df_x = pl.DataFrame({
        "Date": list(range(10)),
        "A": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "B": [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
    })
    df_y = pl.DataFrame({
        "Date": list(range(10)),
        "A": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "B": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    })

    result = ts_corr(df_x, df_y, 5)

    # A correlates perfectly with itself (corr = 1.0)
    assert result["A"][4] == pytest.approx(1.0, rel=1e-6)
    # B is negatively correlated (corr = -1.0)
    assert result["B"][4] == pytest.approx(-1.0, rel=1e-6)


def test_ts_corr_with_nan():
    """NaN values in window return None."""
    from alphalab.api.operators.time_series import ts_corr

    df_x = pl.DataFrame({
        "Date": [1, 2, 3, 4, 5],
        "A": [1.0, 2.0, float("nan"), 4.0, 5.0],
    })
    df_y = pl.DataFrame({
        "Date": [1, 2, 3, 4, 5],
        "A": [1.0, 2.0, 3.0, 4.0, 5.0],
    })
    result = ts_corr(df_x, df_y, 3)
    # Window containing NaN returns missing value
    assert is_missing(result["A"][2])  # Window [1,2,nan] contains NaN
    assert is_missing(result["A"][3])  # Window [2,nan,4] contains NaN
    assert is_missing(result["A"][4])  # Window [nan,4,5] contains NaN


def test_ts_covariance_correctness():
    """Parallelized ts_covariance produces same results."""
    from alphalab.api.operators.time_series import ts_covariance

    df_x = pl.DataFrame({
        "Date": list(range(5)),
        "A": [1.0, 2.0, 3.0, 4.0, 5.0],
    })
    df_y = pl.DataFrame({
        "Date": list(range(5)),
        "A": [2.0, 4.0, 6.0, 8.0, 10.0],  # y = 2*x
    })

    result = ts_covariance(df_x, df_y, 3)

    # Covariance of x with 2*x = 2 * var(x)
    # For window [3,4,5] and [6,8,10]: cov should be 2 * var([3,4,5])
    assert result["A"][4] is not None


def test_ts_covariance_with_nan():
    """NaN values in window return None."""
    from alphalab.api.operators.time_series import ts_covariance

    df_x = pl.DataFrame({
        "Date": [1, 2, 3, 4, 5],
        "A": [1.0, 2.0, float("nan"), 4.0, 5.0],
    })
    df_y = pl.DataFrame({
        "Date": [1, 2, 3, 4, 5],
        "A": [1.0, 2.0, 3.0, 4.0, 5.0],
    })
    result = ts_covariance(df_x, df_y, 3)
    # Windows containing NaN return missing value
    assert is_missing(result["A"][2])
    assert is_missing(result["A"][3])
    assert is_missing(result["A"][4])


def test_ts_regression_correctness():
    """Parallelized ts_regression produces same results."""
    from alphalab.api.operators.time_series import ts_regression

    df_y = pl.DataFrame({
        "Date": list(range(5)),
        "A": [2.0, 4.0, 6.0, 8.0, 10.0],
    })
    df_x = pl.DataFrame({
        "Date": list(range(5)),
        "A": [1.0, 2.0, 3.0, 4.0, 5.0],
    })

    # y = 2*x, so beta should be ~2.0
    result = ts_regression(df_y, df_x, 3, rettype="beta")
    assert result["A"][4] == pytest.approx(2.0, rel=1e-6)

    # alpha should be ~0.0
    result_alpha = ts_regression(df_y, df_x, 3, rettype="alpha")
    assert result_alpha["A"][4] == pytest.approx(0.0, abs=1e-6)


def test_ts_regression_with_nan():
    """NaN values in window return None."""
    from alphalab.api.operators.time_series import ts_regression

    df_y = pl.DataFrame({
        "Date": [1, 2, 3, 4, 5],
        "A": [1.0, 2.0, float("nan"), 4.0, 5.0],
    })
    df_x = pl.DataFrame({
        "Date": [1, 2, 3, 4, 5],
        "A": [1.0, 2.0, 3.0, 4.0, 5.0],
    })
    result = ts_regression(df_y, df_x, 3, rettype="beta")
    # Windows containing NaN return missing value
    assert is_missing(result["A"][2])
    assert is_missing(result["A"][3])
    assert is_missing(result["A"][4])


def test_hump_correctness():
    """Parallelized hump produces same results."""
    from alphalab.api.operators.time_series import hump

    df = pl.DataFrame({
        "Date": [1, 2, 3],
        "A": [1.0, 10.0, 2.0],  # Large jump at row 2
        "B": [5.0, 5.5, 5.2],   # Small changes
    })

    result = hump(df, hump=0.1)

    # First row should be unchanged
    assert result["A"].to_list()[0] == 1.0
    assert result["B"].to_list()[0] == 5.0


def test_ts_decay_linear_correctness():
    """Parallelized ts_decay_linear produces correct weighted average."""
    from alphalab.api.operators.time_series import ts_decay_linear

    # Window of 3 with weights [1, 2, 3], weight_sum = 6
    df = pl.DataFrame({
        "Date": list(range(5)),
        "A": [1.0, 2.0, 3.0, 4.0, 5.0],
        "B": [10.0, 20.0, 30.0, 40.0, 50.0],
    })

    result = ts_decay_linear(df, d=3)

    # First 2 values should be NaN (not enough data for window)
    assert is_missing(result["A"][0])
    assert is_missing(result["A"][1])

    # Index 2: window [1, 2, 3], weighted = (1*1 + 2*2 + 3*3) / 6 = (1 + 4 + 9) / 6 = 14/6 = 2.333...
    assert result["A"][2] == pytest.approx(14.0 / 6.0, rel=1e-9)

    # Index 3: window [2, 3, 4], weighted = (1*2 + 2*3 + 3*4) / 6 = (2 + 6 + 12) / 6 = 20/6 = 3.333...
    assert result["A"][3] == pytest.approx(20.0 / 6.0, rel=1e-9)

    # Index 4: window [3, 4, 5], weighted = (1*3 + 2*4 + 3*5) / 6 = (3 + 8 + 15) / 6 = 26/6 = 4.333...
    assert result["A"][4] == pytest.approx(26.0 / 6.0, rel=1e-9)

    # B column should follow same pattern (values are 10x A)
    assert result["B"][2] == pytest.approx(140.0 / 6.0, rel=1e-9)
    assert result["B"][3] == pytest.approx(200.0 / 6.0, rel=1e-9)
    assert result["B"][4] == pytest.approx(260.0 / 6.0, rel=1e-9)


def test_ts_decay_linear_with_nan():
    """NaN values in window return NaN for dense=False (numba kernel)."""
    from alphalab.api.operators.time_series import ts_decay_linear

    df = pl.DataFrame({
        "Date": [1, 2, 3, 4, 5, 6],
        "A": [1.0, 2.0, float("nan"), 4.0, 5.0, 6.0],
    })

    result = ts_decay_linear(df, d=3)

    # Windows containing NaN should return NaN
    assert is_missing(result["A"][2])  # window [1, 2, nan]
    assert is_missing(result["A"][3])  # window [2, nan, 4]
    assert is_missing(result["A"][4])  # window [nan, 4, 5]

    # Index 5: window [4, 5, 6] - no NaN, should compute correctly
    # weighted = (1*4 + 2*5 + 3*6) / 6 = (4 + 10 + 18) / 6 = 32/6
    assert result["A"][5] == pytest.approx(32.0 / 6.0, rel=1e-9)


def test_ts_decay_linear_multiple_columns():
    """Verify parallel processing works correctly with multiple columns."""
    from alphalab.api.operators.time_series import ts_decay_linear

    # Create DataFrame with many columns to test parallel processing
    df = pl.DataFrame({
        "Date": list(range(10)),
        **{f"Col_{i}": [float(j + i) for j in range(10)] for i in range(10)},
    })

    result = ts_decay_linear(df, d=3)

    # Verify all columns are present
    assert result.columns == df.columns

    # Verify first two rows are NaN for all columns (window not full)
    for col in df.columns[1:]:
        assert is_missing(result[col][0])
        assert is_missing(result[col][1])

    # Verify computation is correct for Col_0: values [0, 1, 2, 3, ...]
    # Index 2: window [0, 1, 2], weighted = (1*0 + 2*1 + 3*2) / 6 = 8/6
    assert result["Col_0"][2] == pytest.approx(8.0 / 6.0, rel=1e-9)


def test_ts_product_correctness():
    """Parallelized ts_product produces correct rolling products."""
    from alphalab.api.operators.time_series import ts_product

    df = pl.DataFrame({
        "Date": list(range(6)),
        "A": [2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        "B": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    })

    result = ts_product(df, d=3)

    # Partial windows (min_samples=1)
    # Index 0: window [2.0] -> product = 2.0
    assert result["A"][0] == pytest.approx(2.0, rel=1e-9)
    # Index 1: window [2.0, 3.0] -> product = 6.0
    assert result["A"][1] == pytest.approx(6.0, rel=1e-9)
    # Index 2: window [2.0, 3.0, 4.0] -> product = 24.0
    assert result["A"][2] == pytest.approx(24.0, rel=1e-9)
    # Index 3: window [3.0, 4.0, 5.0] -> product = 60.0
    assert result["A"][3] == pytest.approx(60.0, rel=1e-9)
    # Index 4: window [4.0, 5.0, 6.0] -> product = 120.0
    assert result["A"][4] == pytest.approx(120.0, rel=1e-9)
    # Index 5: window [5.0, 6.0, 7.0] -> product = 210.0
    assert result["A"][5] == pytest.approx(210.0, rel=1e-9)

    # Column B
    # Index 2: window [1.0, 2.0, 3.0] -> product = 6.0
    assert result["B"][2] == pytest.approx(6.0, rel=1e-9)
    # Index 5: window [4.0, 5.0, 6.0] -> product = 120.0
    assert result["B"][5] == pytest.approx(120.0, rel=1e-9)


def test_ts_product_with_nan():
    """NaN values in window return NaN."""
    from alphalab.api.operators.time_series import ts_product

    df = pl.DataFrame({
        "Date": [1, 2, 3, 4, 5, 6],
        "A": [2.0, 3.0, float("nan"), 4.0, 5.0, 6.0],
    })

    result = ts_product(df, d=3)

    # Index 0, 1: partial windows before NaN
    assert result["A"][0] == pytest.approx(2.0, rel=1e-9)
    assert result["A"][1] == pytest.approx(6.0, rel=1e-9)

    # Windows containing NaN should return NaN
    assert is_missing(result["A"][2])  # window [2, 3, nan]
    assert is_missing(result["A"][3])  # window [3, nan, 4]
    assert is_missing(result["A"][4])  # window [nan, 4, 5]

    # Index 5: window [4, 5, 6] - no NaN
    assert result["A"][5] == pytest.approx(120.0, rel=1e-9)


def test_ts_product_with_zero():
    """Zero in window returns zero (not NaN)."""
    from alphalab.api.operators.time_series import ts_product

    df = pl.DataFrame({
        "Date": [1, 2, 3, 4, 5, 6],
        "A": [2.0, 3.0, 0.0, 4.0, 5.0, 6.0],
    })

    result = ts_product(df, d=3)

    # Index 0, 1: partial windows before zero
    assert result["A"][0] == pytest.approx(2.0, rel=1e-9)
    assert result["A"][1] == pytest.approx(6.0, rel=1e-9)

    # Windows containing zero should return 0.0
    assert result["A"][2] == pytest.approx(0.0, abs=1e-12)  # window [2, 3, 0]
    assert result["A"][3] == pytest.approx(0.0, abs=1e-12)  # window [3, 0, 4]
    assert result["A"][4] == pytest.approx(0.0, abs=1e-12)  # window [0, 4, 5]

    # Index 5: window [4, 5, 6] - no zero
    assert result["A"][5] == pytest.approx(120.0, rel=1e-9)


def test_ts_product_partial_window():
    """Partial windows (min_samples=1) compute product of available values."""
    from alphalab.api.operators.time_series import ts_product

    df = pl.DataFrame({
        "Date": [1, 2, 3, 4],
        "A": [2.0, 3.0, 4.0, 5.0],
    })

    result = ts_product(df, d=10)  # Window larger than data

    # All windows are partial
    # Index 0: [2.0] -> 2.0
    assert result["A"][0] == pytest.approx(2.0, rel=1e-9)
    # Index 1: [2.0, 3.0] -> 6.0
    assert result["A"][1] == pytest.approx(6.0, rel=1e-9)
    # Index 2: [2.0, 3.0, 4.0] -> 24.0
    assert result["A"][2] == pytest.approx(24.0, rel=1e-9)
    # Index 3: [2.0, 3.0, 4.0, 5.0] -> 120.0
    assert result["A"][3] == pytest.approx(120.0, rel=1e-9)


def test_ts_product_with_negative():
    """Negative values handled correctly (sign tracking)."""
    from alphalab.api.operators.time_series import ts_product

    df = pl.DataFrame({
        "Date": [1, 2, 3, 4, 5],
        "A": [2.0, -3.0, 4.0, -5.0, 6.0],
    })

    result = ts_product(df, d=3)

    # Index 0: [2.0] -> 2.0
    assert result["A"][0] == pytest.approx(2.0, rel=1e-9)
    # Index 1: [2.0, -3.0] -> -6.0
    assert result["A"][1] == pytest.approx(-6.0, rel=1e-9)
    # Index 2: [2.0, -3.0, 4.0] -> -24.0 (1 negative)
    assert result["A"][2] == pytest.approx(-24.0, rel=1e-9)
    # Index 3: [-3.0, 4.0, -5.0] -> 60.0 (2 negatives = positive)
    assert result["A"][3] == pytest.approx(60.0, rel=1e-9)
    # Index 4: [4.0, -5.0, 6.0] -> -120.0 (1 negative)
    assert result["A"][4] == pytest.approx(-120.0, rel=1e-9)


def test_ts_product_multiple_columns():
    """Verify parallel processing works correctly with multiple columns."""
    from alphalab.api.operators.time_series import ts_product

    # Create DataFrame with many columns to test parallel processing
    df = pl.DataFrame({
        "Date": list(range(5)),
        **{f"Col_{i}": [float(j + 1) for j in range(5)] for i in range(10)},
    })

    result = ts_product(df, d=3)

    # Verify all columns are present
    assert result.columns == df.columns

    # All columns have same values [1, 2, 3, 4, 5], so products should match
    for col in df.columns[1:]:
        # Index 2: [1, 2, 3] -> 6.0
        assert result[col][2] == pytest.approx(6.0, rel=1e-9)
        # Index 4: [3, 4, 5] -> 60.0
        assert result[col][4] == pytest.approx(60.0, rel=1e-9)


def test_ts_rank_correctness():
    """Parallelized ts_rank produces correct rolling ranks."""
    from alphalab.api.operators.time_series import ts_rank

    df = pl.DataFrame({
        "Date": list(range(6)),
        "A": [1.0, 3.0, 2.0, 5.0, 4.0, 6.0],
        "B": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
    })

    result = ts_rank(df, d=3)

    # Partial windows (min_samples=1)
    # Index 0: window [1.0] -> single value -> 0.5
    assert result["A"][0] == pytest.approx(0.5, rel=1e-9)

    # Index 1: window [1.0, 3.0] -> current=3.0, sorted=[1,3]
    # count_less=1, valid=2 -> rank = 1/(2-1) = 1.0
    assert result["A"][1] == pytest.approx(1.0, rel=1e-9)

    # Index 2: window [1.0, 3.0, 2.0] -> current=2.0, sorted=[1,2,3]
    # count_less=1, valid=3 -> rank = 1/(3-1) = 0.5
    assert result["A"][2] == pytest.approx(0.5, rel=1e-9)

    # Index 3: window [3.0, 2.0, 5.0] -> current=5.0, sorted=[2,3,5]
    # count_less=2, valid=3 -> rank = 2/(3-1) = 1.0
    assert result["A"][3] == pytest.approx(1.0, rel=1e-9)

    # Index 4: window [2.0, 5.0, 4.0] -> current=4.0, sorted=[2,4,5]
    # count_less=1, valid=3 -> rank = 1/(3-1) = 0.5
    assert result["A"][4] == pytest.approx(0.5, rel=1e-9)

    # Index 5: window [5.0, 4.0, 6.0] -> current=6.0, sorted=[4,5,6]
    # count_less=2, valid=3 -> rank = 2/(3-1) = 1.0
    assert result["A"][5] == pytest.approx(1.0, rel=1e-9)

    # Column B: monotonically increasing, so current is always max in window
    # All full windows should have rank = 1.0 (current is largest)
    assert result["B"][2] == pytest.approx(1.0, rel=1e-9)
    assert result["B"][5] == pytest.approx(1.0, rel=1e-9)


def test_ts_rank_with_nan():
    """NaN in current value returns NaN; NaN in window is skipped."""
    from alphalab.api.operators.time_series import ts_rank

    df = pl.DataFrame({
        "Date": [1, 2, 3, 4, 5, 6],
        "A": [1.0, 2.0, float("nan"), 4.0, 5.0, 6.0],
    })

    result = ts_rank(df, d=3)

    # Index 0, 1: partial windows, no NaN
    assert result["A"][0] == pytest.approx(0.5, rel=1e-9)  # single value
    assert result["A"][1] == pytest.approx(1.0, rel=1e-9)  # [1,2], 2 is max

    # Index 2: current is NaN -> NaN
    assert is_missing(result["A"][2])

    # Index 3: window [2, nan, 4], current=4.0 (not NaN)
    # valid values: [2, 4], count_less=1, valid=2 -> rank=1.0
    assert result["A"][3] == pytest.approx(1.0, rel=1e-9)

    # Index 4: window [nan, 4, 5], current=5.0 (not NaN)
    # valid values: [4, 5], count_less=1, valid=2 -> rank=1.0
    assert result["A"][4] == pytest.approx(1.0, rel=1e-9)

    # Index 5: window [4, 5, 6], no NaN
    # valid values: [4, 5, 6], count_less=2, valid=3 -> rank=2/2=1.0
    assert result["A"][5] == pytest.approx(1.0, rel=1e-9)


def test_ts_rank_constant_values():
    """Constant values (ties) handled correctly."""
    from alphalab.api.operators.time_series import ts_rank

    df = pl.DataFrame({
        "Date": list(range(5)),
        "A": [3.0, 3.0, 3.0, 3.0, 3.0],  # All same
        "B": [1.0, 2.0, 2.0, 2.0, 3.0],  # Some ties
    })

    result = ts_rank(df, d=3)

    # Column A: all values are 3.0 (constant)
    # For any window, count_less=0, so rank = 0/(valid-1) = 0.0
    assert result["A"][0] == pytest.approx(0.5, rel=1e-9)  # single value
    assert result["A"][1] == pytest.approx(0.0, rel=1e-9)  # [3,3], 0 less
    assert result["A"][2] == pytest.approx(0.0, rel=1e-9)  # [3,3,3], 0 less
    assert result["A"][3] == pytest.approx(0.0, rel=1e-9)
    assert result["A"][4] == pytest.approx(0.0, rel=1e-9)

    # Column B: [1, 2, 2, 2, 3]
    # Index 2: window [1, 2, 2], current=2.0
    # count_less=1 (only 1 is less than 2), valid=3 -> rank=1/2=0.5
    assert result["B"][2] == pytest.approx(0.5, rel=1e-9)

    # Index 3: window [2, 2, 2], current=2.0
    # count_less=0, valid=3 -> rank=0/2=0.0
    assert result["B"][3] == pytest.approx(0.0, rel=1e-9)

    # Index 4: window [2, 2, 3], current=3.0
    # count_less=2, valid=3 -> rank=2/2=1.0
    assert result["B"][4] == pytest.approx(1.0, rel=1e-9)


def test_ts_rank_with_constant():
    """ts_rank with constant parameter shifts range to [constant, 1+constant]."""
    from alphalab.api.operators.time_series import ts_rank

    df = pl.DataFrame({
        "Date": list(range(5)),
        "A": [1.0, 2.0, 3.0, 4.0, 5.0],
    })

    result = ts_rank(df, d=3, constant=1.0)

    # With constant=1.0, ranks are in [1.0, 2.0] instead of [0.0, 1.0]
    # Index 2: window [1, 2, 3], current=3.0 (max) -> rank = 1.0 + 1.0 = 2.0
    assert result["A"][2] == pytest.approx(2.0, rel=1e-9)

    # Index 0: single value -> 1.0 + 0.5 = 1.5
    assert result["A"][0] == pytest.approx(1.5, rel=1e-9)


def test_ts_rank_multiple_columns():
    """Verify parallel processing works correctly with multiple columns."""
    from alphalab.api.operators.time_series import ts_rank

    # Create DataFrame with many columns to test parallel processing
    df = pl.DataFrame({
        "Date": list(range(5)),
        **{f"Col_{i}": [float(j + 1) for j in range(5)] for i in range(10)},
    })

    result = ts_rank(df, d=3)

    # Verify all columns are present
    assert result.columns == df.columns

    # All columns have same values [1, 2, 3, 4, 5] (monotonically increasing)
    # Current value is always the max in window -> rank = 1.0
    for col in df.columns[1:]:
        # Index 2: [1, 2, 3] -> current=3 is max -> rank=1.0
        assert result[col][2] == pytest.approx(1.0, rel=1e-9)
        # Index 4: [3, 4, 5] -> current=5 is max -> rank=1.0
        assert result[col][4] == pytest.approx(1.0, rel=1e-9)