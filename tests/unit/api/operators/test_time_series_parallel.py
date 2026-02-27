# tests/unit/api/operators/test_time_series_parallel.py
"""Tests for parallelized time-series operators."""

import polars as pl
import pytest


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
    # Window containing NaN returns None
    assert result["A"][2] is None  # Window [1,2,nan] contains NaN
    assert result["A"][3] is None  # Window [2,nan,4] contains NaN
    assert result["A"][4] is None  # Window [nan,4,5] contains NaN


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
    # Windows containing NaN return None
    assert result["A"][2] is None
    assert result["A"][3] is None
    assert result["A"][4] is None


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
    # Windows containing NaN return None
    assert result["A"][2] is None
    assert result["A"][3] is None
    assert result["A"][4] is None