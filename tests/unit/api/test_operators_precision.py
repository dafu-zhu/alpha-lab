"""Numerical precision tests for alpha operators.

Tests numerical stability for edge cases that could cause precision loss:
1. Large value accumulation (1e15 scale)
2. Near-zero variance (constant values + epsilon)
3. Catastrophic cancellation (subtract similar values)
4. Online algorithm stability (numba vs naive over 10k rows)
5. Edge cases: inf, -inf, NaN propagation
"""

from datetime import date

import numpy as np
import polars as pl
import pytest

from alphalab.api.operators import (
    rank,
    ts_corr,
    ts_covariance,
    ts_mean,
    ts_std,
    ts_sum,
    zscore,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def large_value_df() -> pl.DataFrame:
    """Create DataFrame with very large values (1e15 scale).

    Tests accumulation precision in rolling operations.
    """
    np.random.seed(42)
    n_rows = 100
    dates = pl.date_range(date(2024, 1, 1), date(2024, 4, 10), eager=True)[:n_rows]

    # Large base values with small variations
    base = 1e15
    return pl.DataFrame({
        "Date": dates,
        "A": base + np.random.randn(n_rows) * 1e12,
        "B": base + np.random.randn(n_rows) * 1e12,
        "C": base + np.random.randn(n_rows) * 1e12,
    })


@pytest.fixture
def near_constant_df() -> pl.DataFrame:
    """Create DataFrame with near-constant values (constant + epsilon).

    Tests variance calculations with very small variance.
    """
    n_rows = 50
    dates = pl.date_range(date(2024, 1, 1), date(2024, 2, 19), eager=True)[:n_rows]

    base = 100.0
    epsilon = 1e-10
    return pl.DataFrame({
        "Date": dates,
        "A": [base + epsilon * i for i in range(n_rows)],
        "B": [base + epsilon * np.sin(i) for i in range(n_rows)],
        "C": [base] * n_rows,  # Exactly constant
    })


@pytest.fixture
def similar_values_df() -> pl.DataFrame:
    """Create DataFrame with similar values for cancellation tests.

    Tests catastrophic cancellation when subtracting similar values.
    """
    n_rows = 50
    dates = pl.date_range(date(2024, 1, 1), date(2024, 2, 19), eager=True)[:n_rows]

    # Values that are very similar (differ by small amounts)
    base = 1e10
    diff = 1e-5
    return pl.DataFrame({
        "Date": dates,
        "A": [base + diff * i for i in range(n_rows)],
        "B": [base + diff * np.cos(i / 10) for i in range(n_rows)],
        "C": [base + diff * np.sin(i / 10) for i in range(n_rows)],
    })


@pytest.fixture
def long_series_df() -> pl.DataFrame:
    """Create DataFrame with 10k rows for stability testing.

    Tests online algorithm drift over many iterations.
    """
    np.random.seed(42)
    n_rows = 10000
    dates = pl.date_range(date(2000, 1, 1), date(2038, 1, 1), eager=True)[:n_rows]

    return pl.DataFrame({
        "Date": dates,
        "A": np.cumsum(np.random.randn(n_rows)) + 100,
        "B": np.cumsum(np.random.randn(n_rows)) + 200,
        "C": np.cumsum(np.random.randn(n_rows)) + 150,
    })


@pytest.fixture
def edge_case_df() -> pl.DataFrame:
    """Create DataFrame with inf, -inf, and NaN values.

    Tests edge case handling and NaN propagation.
    """
    n_rows = 20
    dates = pl.date_range(date(2024, 1, 1), date(2024, 1, 20), eager=True)[:n_rows]

    values_a = list(np.random.randn(n_rows) * 10 + 100)
    values_b = list(np.random.randn(n_rows) * 10 + 100)
    values_c = list(np.random.randn(n_rows) * 10 + 100)

    # Inject special values
    values_a[5] = float('inf')
    values_a[10] = float('-inf')
    values_b[7] = float('nan')
    values_c[3] = float('inf')
    values_c[15] = float('nan')

    return pl.DataFrame({
        "Date": dates,
        "A": values_a,
        "B": values_b,
        "C": values_c,
    })


# =============================================================================
# TEST CLASS 1: LARGE VALUE ACCUMULATION
# =============================================================================


class TestLargeValueAccumulation:
    """Test numerical stability with large values (1e15 scale).

    Large values can cause precision loss due to floating-point
    representation limits when accumulated.
    """

    def test_ts_sum_large_values(self, large_value_df: pl.DataFrame) -> None:
        """ts_sum should maintain precision with large values."""
        window = 10
        result = ts_sum(large_value_df, window)

        # All results should be finite
        for col in ["A", "B", "C"]:
            values = result[col].to_numpy()
            valid_mask = ~np.isnan(values)
            assert np.all(np.isfinite(values[valid_mask])), \
                f"ts_sum produced non-finite values for {col}"

        # Verify sums are in expected range (window * base value)
        # Note: partial windows will have smaller sums initially
        expected_magnitude = window * 1e15
        for col in ["A", "B", "C"]:
            values = result[col].to_numpy()
            valid = values[~np.isnan(values)]
            # Sums should be within reasonable range of expected
            assert np.all(np.abs(valid) < expected_magnitude * 2), \
                f"ts_sum values out of expected range for {col}"
            # Check that later values (full windows) are near expected
            later_values = valid[window:]
            assert np.all(np.abs(later_values) > expected_magnitude * 0.9), \
                f"ts_sum full window values too small for {col}"

    def test_ts_mean_large_values(self, large_value_df: pl.DataFrame) -> None:
        """ts_mean should maintain precision with large values."""
        window = 10
        result = ts_mean(large_value_df, window)

        # All results should be finite
        for col in ["A", "B", "C"]:
            values = result[col].to_numpy()
            valid_mask = ~np.isnan(values)
            assert np.all(np.isfinite(values[valid_mask])), \
                f"ts_mean produced non-finite values for {col}"

        # Mean should be close to base value
        expected_magnitude = 1e15
        for col in ["A", "B", "C"]:
            values = result[col].to_numpy()
            valid = values[~np.isnan(values)]
            assert np.all(np.abs(valid - expected_magnitude) < 0.1 * expected_magnitude), \
                f"ts_mean values deviated too far from expected for {col}"

    def test_ts_std_large_values(self, large_value_df: pl.DataFrame) -> None:
        """ts_std should maintain precision with large values."""
        window = 10
        result = ts_std(large_value_df, window)

        # All results should be non-negative and finite
        for col in ["A", "B", "C"]:
            values = result[col].to_numpy()
            valid_mask = ~np.isnan(values)
            valid = values[valid_mask]

            assert np.all(np.isfinite(valid)), \
                f"ts_std produced non-finite values for {col}"
            assert np.all(valid >= 0), \
                f"ts_std produced negative values for {col}"

        # Std should be proportional to variation magnitude (1e12)
        expected_magnitude = 1e12
        for col in ["A", "B", "C"]:
            values = result[col].to_numpy()
            valid = values[~np.isnan(values)]
            # Std should be within order of magnitude
            assert np.all(valid < expected_magnitude * 10), \
                f"ts_std values too large for {col}"


# =============================================================================
# TEST CLASS 2: NEAR-ZERO VARIANCE
# =============================================================================


class TestNearZeroVariance:
    """Test numerical stability with near-constant values.

    Near-constant data can cause issues in variance and correlation
    calculations due to division by very small numbers.
    """

    def test_ts_std_near_constant(self, near_constant_df: pl.DataFrame) -> None:
        """ts_std should handle near-constant values gracefully."""
        window = 10
        result = ts_std(near_constant_df, window)

        # For columns with epsilon variation, std should be very small but finite
        for col in ["A", "B"]:
            values = result[col].to_numpy()
            valid_mask = ~np.isnan(values)
            valid = values[valid_mask]

            assert np.all(np.isfinite(valid)), \
                f"ts_std produced non-finite values for {col}"
            assert np.all(valid >= 0), \
                f"ts_std produced negative values for {col}"

        # For exactly constant column, std should be 0
        c_values = result["C"].to_numpy()
        valid_c = c_values[~np.isnan(c_values)]
        assert np.allclose(valid_c, 0.0), \
            "ts_std should be 0 for constant values"

    def test_zscore_near_constant(self, near_constant_df: pl.DataFrame) -> None:
        """zscore should handle near-constant rows gracefully."""
        # Create a df where each row has very similar values
        n_rows = 10
        dates = pl.date_range(date(2024, 1, 1), date(2024, 1, 10), eager=True)[:n_rows]

        # Each row has values differing by epsilon
        epsilon = 1e-12
        df = pl.DataFrame({
            "Date": dates,
            "A": [100.0 + epsilon * i for i in range(n_rows)],
            "B": [100.0 + 2 * epsilon * i for i in range(n_rows)],
            "C": [100.0 + 3 * epsilon * i for i in range(n_rows)],
        })

        result = zscore(df)

        # Zscores should be finite where std > 0
        for col in ["A", "B", "C"]:
            values = result[col].to_numpy()
            finite_mask = np.isfinite(values)
            # Either all finite, or NaN where std would be zero
            assert np.all(finite_mask | np.isnan(values)), \
                f"zscore produced inf values for {col}"

    def test_ts_corr_near_constant(self) -> None:
        """ts_corr should return NaN for near-constant series."""
        n_rows = 20
        dates = pl.date_range(date(2024, 1, 1), date(2024, 1, 20), eager=True)[:n_rows]

        # Nearly constant x, varying y
        df_x = pl.DataFrame({
            "Date": dates,
            "A": [100.0] * n_rows,  # Exactly constant
        })
        df_y = pl.DataFrame({
            "Date": dates,
            "A": np.arange(n_rows, dtype=float),  # Varying
        })

        result = ts_corr(df_x, df_y, 10)

        # Correlation with constant should be NaN (undefined)
        values = result["A"].to_numpy()
        # After window fills, should be NaN due to zero variance in x
        assert np.all(np.isnan(values[9:])), \
            "ts_corr should return NaN when one series is constant"

    def test_ts_covariance_near_constant(self) -> None:
        """ts_covariance should handle constant series."""
        n_rows = 20
        dates = pl.date_range(date(2024, 1, 1), date(2024, 1, 20), eager=True)[:n_rows]

        # Constant x, varying y
        df_x = pl.DataFrame({
            "Date": dates,
            "A": [100.0] * n_rows,
        })
        df_y = pl.DataFrame({
            "Date": dates,
            "A": np.arange(n_rows, dtype=float),
        })

        result = ts_covariance(df_x, df_y, 10)

        # Covariance with constant should be ~0
        values = result["A"].to_numpy()
        valid_mask = ~np.isnan(values)
        valid = values[valid_mask]

        assert np.all(np.isfinite(valid)), \
            "ts_covariance should produce finite values"
        # Cov(constant, y) = E[(c-c)(y-y_bar)] = 0
        assert np.allclose(valid, 0.0, atol=1e-10), \
            "ts_covariance with constant should be 0"


# =============================================================================
# TEST CLASS 3: CATASTROPHIC CANCELLATION
# =============================================================================


class TestCatastrophicCancellation:
    """Test for catastrophic cancellation when subtracting similar values.

    Occurs when subtracting two similar large numbers, losing precision
    in the difference.
    """

    def test_ts_std_cancellation(self, similar_values_df: pl.DataFrame) -> None:
        """ts_std should avoid cancellation with similar values."""
        window = 10
        result = ts_std(similar_values_df, window)

        # All results should be finite and non-negative
        for col in ["A", "B", "C"]:
            values = result[col].to_numpy()
            valid_mask = ~np.isnan(values)
            valid = values[valid_mask]

            assert np.all(np.isfinite(valid)), \
                f"ts_std produced non-finite values for {col}"
            assert np.all(valid >= 0), \
                f"ts_std produced negative values for {col}"

    def test_variance_computation_stability(self) -> None:
        """Test that variance computation doesn't go negative due to cancellation.

        Naive variance: var = E[x^2] - E[x]^2 can be negative due to cancellation.
        """
        n_rows = 50
        dates = pl.date_range(date(2024, 1, 1), date(2024, 2, 19), eager=True)[:n_rows]

        # Create data that would cause naive variance formula to fail
        # Large mean with small variance
        base = 1e8
        df = pl.DataFrame({
            "Date": dates,
            "A": [base + 0.001 * i for i in range(n_rows)],
        })

        result = ts_std(df, 10)
        values = result["A"].to_numpy()
        valid = values[~np.isnan(values)]

        # Should all be non-negative (no negative variance artifacts)
        assert np.all(valid >= 0), \
            "ts_std produced negative values (variance cancellation bug)"

    def test_covariance_cancellation(self, similar_values_df: pl.DataFrame) -> None:
        """ts_covariance should handle similar values without cancellation errors."""
        window = 10

        result = ts_covariance(
            similar_values_df,
            similar_values_df.select(
                similar_values_df.columns[0],
                *[pl.col(c) + 1e-8 for c in similar_values_df.columns[1:]]
            ),
            window
        )

        # All results should be finite
        for col in ["A", "B", "C"]:
            values = result[col].to_numpy()
            valid_mask = ~np.isnan(values)
            valid = values[valid_mask]

            assert np.all(np.isfinite(valid)), \
                f"ts_covariance produced non-finite values for {col}"


# =============================================================================
# TEST CLASS 4: ONLINE ALGORITHM STABILITY
# =============================================================================


class TestOnlineAlgorithmStability:
    """Test online algorithms don't drift over long series.

    Online algorithms accumulate state and can drift from true values
    over many iterations due to rounding errors.
    """

    def test_ts_mean_long_series(self, long_series_df: pl.DataFrame) -> None:
        """ts_mean should match naive implementation over 10k rows."""
        window = 100
        result = ts_mean(long_series_df, window)

        # Compare to naive rolling mean at various points
        for col in ["A", "B", "C"]:
            values = long_series_df[col].to_numpy()
            result_values = result[col].to_numpy()

            # Check at regular intervals
            check_points = [1000, 5000, 9000]
            for i in check_points:
                expected = np.mean(values[max(0, i-window+1):i+1])
                actual = result_values[i]

                # Should match to high precision
                np.testing.assert_allclose(
                    actual, expected,
                    rtol=1e-10,
                    err_msg=f"ts_mean drift at row {i} for {col}"
                )

    def test_ts_sum_long_series(self, long_series_df: pl.DataFrame) -> None:
        """ts_sum should match naive implementation over 10k rows."""
        window = 100
        result = ts_sum(long_series_df, window)

        # Compare to naive rolling sum at various points
        for col in ["A", "B", "C"]:
            values = long_series_df[col].to_numpy()
            result_values = result[col].to_numpy()

            check_points = [1000, 5000, 9000]
            for i in check_points:
                expected = np.sum(values[max(0, i-window+1):i+1])
                actual = result_values[i]

                np.testing.assert_allclose(
                    actual, expected,
                    rtol=1e-10,
                    err_msg=f"ts_sum drift at row {i} for {col}"
                )

    def test_ts_std_long_series(self, long_series_df: pl.DataFrame) -> None:
        """ts_std should match naive implementation over 10k rows."""
        window = 100
        result = ts_std(long_series_df, window)

        # Compare to naive rolling std at various points
        for col in ["A", "B", "C"]:
            values = long_series_df[col].to_numpy()
            result_values = result[col].to_numpy()

            check_points = [1000, 5000, 9000]
            for i in check_points:
                window_vals = values[max(0, i-window+1):i+1]
                expected = np.std(window_vals, ddof=0)
                actual = result_values[i]

                np.testing.assert_allclose(
                    actual, expected,
                    rtol=1e-9,
                    err_msg=f"ts_std drift at row {i} for {col}"
                )

    def test_ts_corr_long_series(self, long_series_df: pl.DataFrame) -> None:
        """ts_corr should match naive implementation over 10k rows."""
        window = 100

        # Create a correlated series
        np.random.seed(42)
        n_rows = len(long_series_df)
        dates = long_series_df["Date"]

        df_x = long_series_df
        df_y = pl.DataFrame({
            "Date": dates,
            "A": df_x["A"].to_numpy() * 0.8 + np.random.randn(n_rows) * 5,
            "B": df_x["B"].to_numpy() * 0.8 + np.random.randn(n_rows) * 5,
            "C": df_x["C"].to_numpy() * 0.8 + np.random.randn(n_rows) * 5,
        })

        result = ts_corr(df_x, df_y, window)

        # Compare at various points
        check_points = [1000, 5000, 9000]
        for col in ["A", "B", "C"]:
            x_vals = df_x[col].to_numpy()
            y_vals = df_y[col].to_numpy()
            result_values = result[col].to_numpy()

            for i in check_points:
                x_window = x_vals[max(0, i-window+1):i+1]
                y_window = y_vals[max(0, i-window+1):i+1]
                expected = np.corrcoef(x_window, y_window)[0, 1]
                actual = result_values[i]

                np.testing.assert_allclose(
                    actual, expected,
                    rtol=1e-9,
                    err_msg=f"ts_corr drift at row {i} for {col}"
                )


# =============================================================================
# TEST CLASS 5: EDGE CASES - INF, -INF, NAN PROPAGATION
# =============================================================================


class TestEdgeCases:
    """Test edge case handling: inf, -inf, NaN propagation."""

    def test_ts_mean_inf_propagation(self, edge_case_df: pl.DataFrame) -> None:
        """ts_mean should propagate inf through window."""
        window = 5
        result = ts_mean(edge_case_df, window)

        # Check column A which has inf at position 5
        values_a = result["A"].to_numpy()

        # Rows 5-9 should have inf (while inf is in window)
        for i in range(5, min(10, len(values_a))):
            assert np.isinf(values_a[i]) or np.isnan(values_a[i]), \
                f"Expected inf at row {i}, got {values_a[i]}"

    def test_ts_sum_inf_propagation(self, edge_case_df: pl.DataFrame) -> None:
        """ts_sum should propagate inf through window."""
        window = 5
        result = ts_sum(edge_case_df, window)

        values_a = result["A"].to_numpy()

        # Rows with inf in window should have inf result
        for i in range(5, min(10, len(values_a))):
            assert np.isinf(values_a[i]) or np.isnan(values_a[i]), \
                f"Expected inf at row {i}, got {values_a[i]}"

    def test_ts_std_nan_propagation(self, edge_case_df: pl.DataFrame) -> None:
        """ts_std should handle NaN in window appropriately."""
        window = 5
        result = ts_std(edge_case_df, window)

        # Column B has NaN at position 7
        values_b = result["B"].to_numpy()

        # Check that computation continues (doesn't crash)
        assert len(values_b) == len(edge_case_df), \
            "ts_std should return same number of rows"

    def test_zscore_with_inf(self) -> None:
        """zscore should handle inf values gracefully."""
        n_rows = 10
        dates = pl.date_range(date(2024, 1, 1), date(2024, 1, 10), eager=True)[:n_rows]

        df = pl.DataFrame({
            "Date": dates,
            "A": [1.0, 2.0, float('inf'), 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "B": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "C": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        })

        # zscore should not crash
        result = zscore(df)

        # Row 2 has inf, so that row's zscore might have NaN/inf
        assert len(result) == n_rows, "zscore should return same number of rows"

    def test_rank_with_nan(self) -> None:
        """rank should handle NaN values correctly."""
        n_rows = 10
        dates = pl.date_range(date(2024, 1, 1), date(2024, 1, 10), eager=True)[:n_rows]

        df = pl.DataFrame({
            "Date": dates,
            "A": [1.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "B": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "C": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        })

        result = rank(df, rate=0)

        # Row with NaN should have NaN in that column's rank
        values_a = result["A"].to_numpy()
        assert np.isnan(values_a[1]), "rank should return NaN for NaN input"

        # Other values should be in [0, 1]
        valid = values_a[~np.isnan(values_a)]
        assert np.all(valid >= 0) and np.all(valid <= 1), \
            "rank should be in [0, 1]"

    def test_rank_with_inf(self) -> None:
        """rank should handle inf values correctly."""
        n_rows = 10
        dates = pl.date_range(date(2024, 1, 1), date(2024, 1, 10), eager=True)[:n_rows]

        df = pl.DataFrame({
            "Date": dates,
            "A": [1.0, float('inf'), 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "B": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "C": [1.0, 2.0, 3.0, float('-inf'), 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        })

        result = rank(df, rate=0)

        # inf should have highest rank (1.0)
        values_a = result["A"].to_numpy()
        assert values_a[1] == 1.0, "inf should have rank 1.0"

        # -inf should have lowest rank (0.0)
        values_c = result["C"].to_numpy()
        assert values_c[3] == 0.0, "-inf should have rank 0.0"

    def test_ts_corr_mixed_special_values(self) -> None:
        """ts_corr should handle mixed special values."""
        n_rows = 20
        dates = pl.date_range(date(2024, 1, 1), date(2024, 1, 20), eager=True)[:n_rows]

        df_x = pl.DataFrame({
            "Date": dates,
            "A": [float(i) for i in range(n_rows)],
        })

        values_y = [float(i) for i in range(n_rows)]
        values_y[5] = float('nan')
        df_y = pl.DataFrame({
            "Date": dates,
            "A": values_y,
        })

        result = ts_corr(df_x, df_y, 10)

        # Result should not crash and should have appropriate NaN handling
        assert len(result) == n_rows, "ts_corr should return same number of rows"


# =============================================================================
# ADDITIONAL STABILITY TESTS
# =============================================================================


class TestOutputBounds:
    """Test that outputs stay within expected bounds."""

    def test_rank_bounds(self) -> None:
        """rank output should be in [0, 1]."""
        np.random.seed(42)
        n_rows = 100
        dates = pl.date_range(date(2024, 1, 1), date(2024, 4, 10), eager=True)[:n_rows]

        df = pl.DataFrame({
            "Date": dates,
            "A": np.random.randn(n_rows) * 100,
            "B": np.random.randn(n_rows) * 100,
            "C": np.random.randn(n_rows) * 100,
        })

        result = rank(df, rate=0)

        for col in ["A", "B", "C"]:
            values = result[col].to_numpy()
            valid = values[~np.isnan(values)]
            assert np.all(valid >= 0.0), f"rank below 0 for {col}"
            assert np.all(valid <= 1.0), f"rank above 1 for {col}"

    def test_ts_corr_bounds(self) -> None:
        """ts_corr output should be in [-1, 1]."""
        np.random.seed(42)
        n_rows = 100
        dates = pl.date_range(date(2024, 1, 1), date(2024, 4, 10), eager=True)[:n_rows]

        df_x = pl.DataFrame({
            "Date": dates,
            "A": np.cumsum(np.random.randn(n_rows)),
            "B": np.cumsum(np.random.randn(n_rows)),
        })
        df_y = pl.DataFrame({
            "Date": dates,
            "A": np.cumsum(np.random.randn(n_rows)),
            "B": np.cumsum(np.random.randn(n_rows)),
        })

        result = ts_corr(df_x, df_y, 20)

        for col in ["A", "B"]:
            values = result[col].to_numpy()
            valid = values[~np.isnan(values)]
            assert np.all(valid >= -1.0 - 1e-10), f"ts_corr below -1 for {col}"
            assert np.all(valid <= 1.0 + 1e-10), f"ts_corr above 1 for {col}"

    def test_zscore_has_zero_mean(self) -> None:
        """zscore output should have ~0 mean per row."""
        np.random.seed(42)
        n_rows = 50
        dates = pl.date_range(date(2024, 1, 1), date(2024, 2, 19), eager=True)[:n_rows]

        df = pl.DataFrame({
            "Date": dates,
            "A": np.random.randn(n_rows) * 100,
            "B": np.random.randn(n_rows) * 50 + 1000,
            "C": np.random.randn(n_rows) * 200 - 500,
            "D": np.random.randn(n_rows) * 10,
            "E": np.random.randn(n_rows) * 1000,
        })

        result = zscore(df)

        value_cols = ["A", "B", "C", "D", "E"]
        values = result.select(value_cols).to_numpy()

        # Each row should have mean ~0
        row_means = np.nanmean(values, axis=1)
        assert np.allclose(row_means, 0.0, atol=1e-10), \
            "zscore rows should have mean 0"

    def test_zscore_has_unit_std(self) -> None:
        """zscore output should have ~1 std per row."""
        np.random.seed(42)
        n_rows = 50
        dates = pl.date_range(date(2024, 1, 1), date(2024, 2, 19), eager=True)[:n_rows]

        df = pl.DataFrame({
            "Date": dates,
            "A": np.random.randn(n_rows) * 100,
            "B": np.random.randn(n_rows) * 50 + 1000,
            "C": np.random.randn(n_rows) * 200 - 500,
            "D": np.random.randn(n_rows) * 10,
            "E": np.random.randn(n_rows) * 1000,
        })

        result = zscore(df)

        value_cols = ["A", "B", "C", "D", "E"]
        values = result.select(value_cols).to_numpy()

        # Each row should have std ~1
        row_stds = np.nanstd(values, axis=1, ddof=0)
        assert np.allclose(row_stds, 1.0, rtol=1e-10), \
            "zscore rows should have std 1"
