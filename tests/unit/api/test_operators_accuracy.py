"""Accuracy tests for alpha operators against reference implementations.

Compares operator outputs to trusted implementations from pandas, numpy, and scipy
to verify numerical correctness.
"""

from datetime import date

import numpy as np
import pandas as pd
import polars as pl
import pytest
from scipy import stats
from scipy.stats import mstats

from alphalab.api.operators import (
    quantile,
    rank,
    ts_corr,
    ts_covariance,
    ts_mean,
    ts_regression,
    ts_std,
    ts_sum,
    winsorize,
    zscore,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def accuracy_df() -> pl.DataFrame:
    """Create DataFrame with varied data for accuracy testing.

    Uses larger window sizes and more diverse values than basic tests
    to better exercise numerical accuracy.
    """
    np.random.seed(42)
    n_rows = 50
    dates = pl.date_range(date(2024, 1, 1), date(2024, 2, 19), eager=True)[:n_rows]

    # Generate data with varying characteristics
    return pl.DataFrame({
        "Date": dates,
        "AAPL": np.cumsum(np.random.randn(n_rows)) + 100,
        "MSFT": np.cumsum(np.random.randn(n_rows)) + 200,
        "GOOGL": np.cumsum(np.random.randn(n_rows)) + 150,
        "TSLA": np.cumsum(np.random.randn(n_rows)) + 180,
        "NVDA": np.cumsum(np.random.randn(n_rows)) + 120,
    })


@pytest.fixture
def accuracy_df_with_nans() -> pl.DataFrame:
    """Create DataFrame with NaN values for edge case testing."""
    np.random.seed(42)
    n_rows = 30
    dates = pl.date_range(date(2024, 1, 1), date(2024, 1, 30), eager=True)[:n_rows]

    data = {
        "Date": dates,
        "AAPL": np.cumsum(np.random.randn(n_rows)) + 100,
        "MSFT": np.cumsum(np.random.randn(n_rows)) + 200,
        "GOOGL": np.cumsum(np.random.randn(n_rows)) + 150,
    }

    # Inject some NaN values
    data["AAPL"][5] = np.nan
    data["AAPL"][15] = np.nan
    data["MSFT"][10] = np.nan

    return pl.DataFrame(data)


@pytest.fixture
def two_series_df() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Create two correlated DataFrames for two-variable operator tests."""
    np.random.seed(42)
    n_rows = 50
    dates = pl.date_range(date(2024, 1, 1), date(2024, 2, 19), eager=True)[:n_rows]

    # Create correlated series
    x_base = np.cumsum(np.random.randn(n_rows))
    noise = np.random.randn(n_rows) * 0.5

    df_x = pl.DataFrame({
        "Date": dates,
        "AAPL": x_base + 100,
        "MSFT": x_base * 1.5 + 200,
        "GOOGL": x_base * 0.8 + 150,
    })

    df_y = pl.DataFrame({
        "Date": dates,
        "AAPL": x_base * 0.9 + noise + 100,
        "MSFT": x_base * 1.2 + noise + 200,
        "GOOGL": x_base * 0.6 + noise + 150,
    })

    return df_x, df_y


@pytest.fixture
def cross_sectional_df() -> pl.DataFrame:
    """Create DataFrame for cross-sectional operator tests."""
    np.random.seed(42)
    n_rows = 20
    dates = pl.date_range(date(2024, 1, 1), date(2024, 1, 20), eager=True)[:n_rows]

    return pl.DataFrame({
        "Date": dates,
        "AAPL": np.random.randn(n_rows) * 10 + 100,
        "MSFT": np.random.randn(n_rows) * 15 + 200,
        "GOOGL": np.random.randn(n_rows) * 12 + 150,
        "TSLA": np.random.randn(n_rows) * 20 + 180,
        "NVDA": np.random.randn(n_rows) * 8 + 120,
    })


# =============================================================================
# TIME-SERIES OPERATOR ACCURACY TESTS
# =============================================================================


class TestTsMeanAccuracy:
    """Test ts_mean against pandas rolling().mean()."""

    def test_ts_mean_matches_pandas(self, accuracy_df: pl.DataFrame) -> None:
        """ts_mean should match pandas rolling mean."""
        window = 10
        result = ts_mean(accuracy_df, window)

        # Compare each column
        for col in ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]:
            pd_series = pd.Series(accuracy_df[col].to_numpy())
            expected = pd_series.rolling(window=window, min_periods=1).mean()

            actual = result[col].to_numpy()

            np.testing.assert_allclose(
                actual, expected.values,
                rtol=1e-10,
                err_msg=f"ts_mean mismatch for {col}"
            )

    def test_ts_mean_various_windows(self, accuracy_df: pl.DataFrame) -> None:
        """Test ts_mean accuracy across different window sizes."""
        for window in [3, 5, 10, 20]:
            result = ts_mean(accuracy_df, window)

            col = "AAPL"
            pd_series = pd.Series(accuracy_df[col].to_numpy())
            expected = pd_series.rolling(window=window, min_periods=1).mean()

            np.testing.assert_allclose(
                result[col].to_numpy(), expected.values,
                rtol=1e-10,
                err_msg=f"ts_mean mismatch for window={window}"
            )


class TestTsStdAccuracy:
    """Test ts_std against pandas rolling().std(ddof=0)."""

    def test_ts_std_matches_pandas(self, accuracy_df: pl.DataFrame) -> None:
        """ts_std should match pandas rolling std with ddof=0."""
        window = 10
        result = ts_std(accuracy_df, window)

        for col in ["AAPL", "MSFT", "GOOGL"]:
            pd_series = pd.Series(accuracy_df[col].to_numpy())
            # Spec requires ddof=0 (population std)
            expected = pd_series.rolling(window=window, min_periods=2).std(ddof=0)

            actual = result[col].to_numpy()

            # Compare only non-NaN values
            mask = ~np.isnan(expected.values) & ~np.isnan(actual)
            np.testing.assert_allclose(
                actual[mask], expected.values[mask],
                rtol=1e-10,
                err_msg=f"ts_std mismatch for {col}"
            )


class TestTsSumAccuracy:
    """Test ts_sum against pandas rolling().sum()."""

    def test_ts_sum_matches_pandas(self, accuracy_df: pl.DataFrame) -> None:
        """ts_sum should match pandas rolling sum."""
        window = 10
        result = ts_sum(accuracy_df, window)

        for col in ["AAPL", "MSFT", "GOOGL"]:
            pd_series = pd.Series(accuracy_df[col].to_numpy())
            expected = pd_series.rolling(window=window, min_periods=1).sum()

            np.testing.assert_allclose(
                result[col].to_numpy(), expected.values,
                rtol=1e-10,
                err_msg=f"ts_sum mismatch for {col}"
            )


class TestTsCorrAccuracy:
    """Test ts_corr against pandas rolling().corr()."""

    def test_ts_corr_matches_pandas(self, two_series_df: tuple[pl.DataFrame, pl.DataFrame]) -> None:
        """ts_corr should match pandas rolling correlation."""
        df_x, df_y = two_series_df
        window = 10
        result = ts_corr(df_x, df_y, window)

        for col in ["AAPL", "MSFT", "GOOGL"]:
            pd_x = pd.Series(df_x[col].to_numpy())
            pd_y = pd.Series(df_y[col].to_numpy())
            expected = pd_x.rolling(window=window, min_periods=2).corr(pd_y)

            actual = result[col].to_numpy()

            # Compare only non-NaN values (first window-1 may be NaN)
            mask = ~np.isnan(expected.values) & ~np.isnan(actual)
            np.testing.assert_allclose(
                actual[mask], expected.values[mask],
                rtol=1e-9,
                err_msg=f"ts_corr mismatch for {col}"
            )

    def test_ts_corr_perfect_correlation(self) -> None:
        """Perfectly correlated series should have correlation = 1."""
        dates = pl.date_range(date(2024, 1, 1), date(2024, 1, 20), eager=True)
        values = np.arange(20).astype(float)

        df_x = pl.DataFrame({"Date": dates, "A": values})
        df_y = pl.DataFrame({"Date": dates, "A": values * 2 + 5})  # Perfect linear relationship

        result = ts_corr(df_x, df_y, 5)

        # After first few rows, should be 1.0
        actual = result["A"].to_numpy()
        mask = ~np.isnan(actual)
        np.testing.assert_allclose(
            actual[mask], np.ones(mask.sum()),
            rtol=1e-10,
            err_msg="Perfect correlation should be 1.0"
        )


class TestTsCovarianceAccuracy:
    """Test ts_covariance against pandas rolling().cov().

    Note: ts_covariance uses population covariance (ddof=0, divides by N),
    while pandas uses sample covariance (ddof=1, divides by N-1).
    We test by comparing our result * (d-1)/d to pandas result.
    """

    def test_ts_covariance_matches_pandas(self, two_series_df: tuple[pl.DataFrame, pl.DataFrame]) -> None:
        """ts_covariance should match pandas rolling covariance (after ddof correction)."""
        df_x, df_y = two_series_df
        window = 10
        result = ts_covariance(df_x, df_y, window)

        for col in ["AAPL", "MSFT", "GOOGL"]:
            pd_x = pd.Series(df_x[col].to_numpy())
            pd_y = pd.Series(df_y[col].to_numpy())
            # Pandas uses sample covariance (ddof=1)
            pandas_cov = pd_x.rolling(window=window, min_periods=2).cov(pd_y)

            actual = result[col].to_numpy()

            # Our implementation uses population covariance (divides by N)
            # Pandas uses sample covariance (divides by N-1)
            # To compare: actual * (N)/(N-1) should match pandas
            # Or equivalently: actual should match pandas * (N-1)/N
            expected = pandas_cov.values * (window - 1) / window

            # Compare only non-NaN values
            mask = ~np.isnan(expected) & ~np.isnan(actual)
            np.testing.assert_allclose(
                actual[mask], expected[mask],
                rtol=1e-9,
                err_msg=f"ts_covariance mismatch for {col}"
            )


class TestTsRegressionAccuracy:
    """Test ts_regression against scipy.stats.linregress."""

    def test_ts_regression_beta_matches_scipy(self, two_series_df: tuple[pl.DataFrame, pl.DataFrame]) -> None:
        """ts_regression beta should match scipy linregress slope."""
        df_x, df_y = two_series_df
        window = 10

        # Get beta (slope) from ts_regression
        result = ts_regression(df_y, df_x, window, rettype=1)  # rettype=1 is beta

        col = "AAPL"
        x_vals = df_x[col].to_numpy()
        y_vals = df_y[col].to_numpy()

        # Compute expected using scipy for each window
        expected_beta = []
        for i in range(len(y_vals)):
            start = max(0, i - window + 1)
            x_win = x_vals[start:i + 1]
            y_win = y_vals[start:i + 1]

            if len(x_win) >= 2:
                slope, _, _, _, _ = stats.linregress(x_win, y_win)
                expected_beta.append(slope)
            else:
                expected_beta.append(np.nan)

        actual = result[col].to_numpy()
        expected = np.array(expected_beta)

        mask = ~np.isnan(expected) & ~np.isnan(actual)
        np.testing.assert_allclose(
            actual[mask], expected[mask],
            rtol=1e-9,
            err_msg="ts_regression beta mismatch"
        )

    def test_ts_regression_alpha_matches_scipy(self, two_series_df: tuple[pl.DataFrame, pl.DataFrame]) -> None:
        """ts_regression alpha should match scipy linregress intercept."""
        df_x, df_y = two_series_df
        window = 10

        result = ts_regression(df_y, df_x, window, rettype=2)  # rettype=2 is alpha

        col = "AAPL"
        x_vals = df_x[col].to_numpy()
        y_vals = df_y[col].to_numpy()

        expected_alpha = []
        for i in range(len(y_vals)):
            start = max(0, i - window + 1)
            x_win = x_vals[start:i + 1]
            y_win = y_vals[start:i + 1]

            if len(x_win) >= 2:
                _, intercept, _, _, _ = stats.linregress(x_win, y_win)
                expected_alpha.append(intercept)
            else:
                expected_alpha.append(np.nan)

        actual = result[col].to_numpy()
        expected = np.array(expected_alpha)

        mask = ~np.isnan(expected) & ~np.isnan(actual)
        np.testing.assert_allclose(
            actual[mask], expected[mask],
            rtol=1e-9,
            err_msg="ts_regression alpha mismatch"
        )

    def test_ts_regression_r_squared_matches_scipy(self, two_series_df: tuple[pl.DataFrame, pl.DataFrame]) -> None:
        """ts_regression r_squared should match scipy linregress r-value squared."""
        df_x, df_y = two_series_df
        window = 10

        result = ts_regression(df_y, df_x, window, rettype=5)  # rettype=5 is r_squared

        col = "AAPL"
        x_vals = df_x[col].to_numpy()
        y_vals = df_y[col].to_numpy()

        expected_r2 = []
        for i in range(len(y_vals)):
            start = max(0, i - window + 1)
            x_win = x_vals[start:i + 1]
            y_win = y_vals[start:i + 1]

            if len(x_win) >= 2:
                _, _, r_value, _, _ = stats.linregress(x_win, y_win)
                expected_r2.append(r_value ** 2)
            else:
                expected_r2.append(np.nan)

        actual = result[col].to_numpy()
        expected = np.array(expected_r2)

        mask = ~np.isnan(expected) & ~np.isnan(actual)
        np.testing.assert_allclose(
            actual[mask], expected[mask],
            rtol=1e-9,
            err_msg="ts_regression r_squared mismatch"
        )


# =============================================================================
# CROSS-SECTIONAL OPERATOR ACCURACY TESTS
# =============================================================================


class TestRankAccuracy:
    """Test rank against scipy.stats.rankdata."""

    def test_rank_matches_scipy(self, cross_sectional_df: pl.DataFrame) -> None:
        """rank should produce values consistent with scipy rankdata."""
        result = rank(cross_sectional_df, rate=0)  # Use precise ranking

        value_cols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
        values = cross_sectional_df.select(value_cols).to_numpy()

        for i in range(len(cross_sectional_df)):
            row = values[i]
            # scipy rankdata with 'ordinal' method, normalized to [0, 1]
            scipy_ranks = stats.rankdata(row, method='ordinal')
            n = len(row)
            expected = (scipy_ranks - 1) / (n - 1)  # Normalize to [0, 1]

            actual = np.array([result[col][i] for col in value_cols])

            # Our rank uses argsort which gives ordinal ranks
            np.testing.assert_allclose(
                actual, expected,
                rtol=1e-10,
                err_msg=f"rank mismatch at row {i}"
            )

    def test_rank_preserves_order(self, cross_sectional_df: pl.DataFrame) -> None:
        """rank should preserve the relative order of values."""
        result = rank(cross_sectional_df, rate=0)

        value_cols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
        values = cross_sectional_df.select(value_cols).to_numpy()

        for i in range(len(cross_sectional_df)):
            row = values[i]
            ranked_row = np.array([result[col][i] for col in value_cols])

            # Higher values should have higher ranks
            order_original = np.argsort(row)
            order_ranked = np.argsort(ranked_row)
            np.testing.assert_array_equal(
                order_original, order_ranked,
                err_msg=f"rank order mismatch at row {i}"
            )


class TestZscoreAccuracy:
    """Test zscore against scipy.stats.zscore."""

    def test_zscore_matches_scipy(self, cross_sectional_df: pl.DataFrame) -> None:
        """zscore should match scipy.stats.zscore with ddof=0."""
        result = zscore(cross_sectional_df)

        value_cols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
        values = cross_sectional_df.select(value_cols).to_numpy()

        for i in range(len(cross_sectional_df)):
            row = values[i]
            # scipy zscore with ddof=0 (population std)
            expected = stats.zscore(row, ddof=0)
            actual = np.array([result[col][i] for col in value_cols])

            np.testing.assert_allclose(
                actual, expected,
                rtol=1e-10,
                err_msg=f"zscore mismatch at row {i}"
            )

    def test_zscore_properties(self, cross_sectional_df: pl.DataFrame) -> None:
        """zscore should have mean=0 and std=1 for each row."""
        result = zscore(cross_sectional_df)

        value_cols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]

        for i in range(len(result)):
            row = np.array([result[col][i] for col in value_cols])

            # Mean should be ~0
            np.testing.assert_allclose(
                np.mean(row), 0.0,
                atol=1e-10,
                err_msg=f"zscore mean != 0 at row {i}"
            )

            # Std should be ~1
            np.testing.assert_allclose(
                np.std(row, ddof=0), 1.0,
                rtol=1e-10,
                err_msg=f"zscore std != 1 at row {i}"
            )


class TestQuantileAccuracy:
    """Test quantile against scipy.stats.norm.ppf(rank)."""

    def test_quantile_gaussian_matches_scipy(self, cross_sectional_df: pl.DataFrame) -> None:
        """quantile with gaussian driver should match norm.ppf of ranks.

        Note: Uses slightly relaxed tolerance (1e-8) due to different numerical
        implementations of the inverse normal CDF between scipy.ndtri and our
        Abramowitz-Stegun approximation. Both are accurate to ~1e-9.
        """
        result = quantile(cross_sectional_df, driver="gaussian", sigma=1.0)

        value_cols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
        values = cross_sectional_df.select(value_cols).to_numpy()

        for i in range(len(cross_sectional_df)):
            row = values[i]
            n = len(row)

            # Compute ranks and transform
            order = np.argsort(row)
            ranks = np.empty(n)
            ranks[order] = np.arange(n)
            ranks = ranks / (n - 1)  # [0, 1]

            # Shift to avoid boundaries: [1/N, 1-1/N]
            shifted = 1 / n + ranks * (1 - 2 / n)

            # Apply inverse CDF
            expected = stats.norm.ppf(shifted)

            actual = np.array([result[col][i] for col in value_cols])

            # Relaxed tolerance due to different inverse normal implementations
            np.testing.assert_allclose(
                actual, expected,
                rtol=1e-8,
                err_msg=f"quantile gaussian mismatch at row {i}"
            )

    def test_quantile_uniform_range(self, cross_sectional_df: pl.DataFrame) -> None:
        """quantile with uniform driver should produce values in [-sigma, sigma]."""
        sigma = 2.0
        result = quantile(cross_sectional_df, driver="uniform", sigma=sigma)

        value_cols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]

        for col in value_cols:
            values = result[col].to_numpy()
            valid = values[~np.isnan(values)]

            assert np.all(valid >= -sigma - 1e-10), f"quantile uniform below -sigma for {col}"
            assert np.all(valid <= sigma + 1e-10), f"quantile uniform above sigma for {col}"


class TestWinsorizeAccuracy:
    """Test winsorize using std-deviation based clipping.

    Note: AlphaLab's winsorize clips at mean ± std*SD (std-deviation based),
    which differs from scipy.stats.mstats.winsorize (percentile based).
    The reference implementation uses np.clip with computed bounds.
    """

    def test_winsorize_clips_extremes(self, cross_sectional_df: pl.DataFrame) -> None:
        """winsorize should clip values outside mean +/- std*SD."""
        std_limit = 2.0
        result = winsorize(cross_sectional_df, std=std_limit)

        value_cols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
        values = cross_sectional_df.select(value_cols).to_numpy()

        for i in range(len(cross_sectional_df)):
            row = values[i]
            row_mean = np.mean(row)
            row_std = np.std(row, ddof=0)

            lower = row_mean - std_limit * row_std
            upper = row_mean + std_limit * row_std

            # Reference: clip at mean ± std*SD
            # Note: scipy_winsorize uses percentile-based limits which differs
            # from std-deviation based clipping used by AlphaLab's winsorize
            expected = np.clip(row, lower, upper)
            actual = np.array([result[col][i] for col in value_cols])

            np.testing.assert_allclose(
                actual, expected,
                rtol=1e-10,
                err_msg=f"winsorize mismatch at row {i}"
            )

    def test_winsorize_preserves_normal_values(self, cross_sectional_df: pl.DataFrame) -> None:
        """winsorize should not change values within limits."""
        std_limit = 4.0  # Wide enough that most values won't be clipped
        result = winsorize(cross_sectional_df, std=std_limit)

        value_cols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
        values = cross_sectional_df.select(value_cols).to_numpy()

        for i in range(len(cross_sectional_df)):
            row = values[i]
            row_mean = np.mean(row)
            row_std = np.std(row, ddof=0)

            lower = row_mean - std_limit * row_std
            upper = row_mean + std_limit * row_std

            for j, col in enumerate(value_cols):
                if lower <= row[j] <= upper:
                    # Value within limits should be unchanged
                    np.testing.assert_allclose(
                        result[col][i], row[j],
                        rtol=1e-10,
                        err_msg=f"winsorize changed value within limits at row {i}, col {col}"
                    )


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestEdgeCases:
    """Edge case tests for numerical stability and NaN handling."""

    def test_ts_mean_with_nans(self, accuracy_df_with_nans: pl.DataFrame) -> None:
        """ts_mean should handle NaN values like pandas."""
        window = 5
        result = ts_mean(accuracy_df_with_nans, window)

        for col in ["AAPL", "MSFT", "GOOGL"]:
            pd_series = pd.Series(accuracy_df_with_nans[col].to_numpy())
            expected = pd_series.rolling(window=window, min_periods=1).mean()

            actual = result[col].to_numpy()
            mask = ~np.isnan(expected.values) & ~np.isnan(actual)

            np.testing.assert_allclose(
                actual[mask], expected.values[mask],
                rtol=1e-10,
                err_msg=f"ts_mean with NaN mismatch for {col}"
            )

    def test_zscore_with_identical_values(self) -> None:
        """zscore should return NaN when all values are identical (std=0)."""
        dates = pl.date_range(date(2024, 1, 1), date(2024, 1, 5), eager=True)
        df = pl.DataFrame({
            "Date": dates,
            "A": [100.0, 100.0, 100.0, 100.0, 100.0],
            "B": [100.0, 100.0, 100.0, 100.0, 100.0],
            "C": [100.0, 100.0, 100.0, 100.0, 100.0],
        })

        result = zscore(df)

        # All zscores should be NaN when std=0
        for col in ["A", "B", "C"]:
            assert all(np.isnan(result[col].to_numpy())), f"zscore should be NaN for {col}"

    def test_rank_with_ties(self) -> None:
        """rank should handle tied values."""
        dates = pl.date_range(date(2024, 1, 1), date(2024, 1, 3), eager=True)
        df = pl.DataFrame({
            "Date": dates,
            "A": [1.0, 2.0, 1.0],
            "B": [2.0, 2.0, 2.0],
            "C": [3.0, 2.0, 3.0],
        })

        result = rank(df, rate=0)

        # Verify rank is in [0, 1]
        for col in ["A", "B", "C"]:
            values = result[col].to_numpy()
            assert np.all(values >= 0.0), f"rank below 0 for {col}"
            assert np.all(values <= 1.0), f"rank above 1 for {col}"

    def test_ts_std_single_value(self) -> None:
        """ts_std should return null for windows with single value."""
        dates = pl.date_range(date(2024, 1, 1), date(2024, 1, 5), eager=True)
        df = pl.DataFrame({
            "Date": dates,
            "AAPL": [100.0, 102.0, 101.0, 103.0, 105.0],
        })

        result = ts_std(df, 3)

        # First row should be null (only 1 value in window, needs min_samples=2)
        # Polars returns None (null), not NaN
        assert result["AAPL"][0] is None, "ts_std should be null for single value window"

        # Second row has 2 values, should have valid std
        assert result["AAPL"][1] is not None, "ts_std should be valid for 2+ values"

    def test_large_values_numerical_stability(self) -> None:
        """Operators should maintain accuracy with large values."""
        np.random.seed(42)
        dates = pl.date_range(date(2024, 1, 1), date(2024, 1, 10), eager=True)

        # Large values that could cause precision issues
        df = pl.DataFrame({
            "Date": dates,
            "A": np.random.randn(10) * 1e10 + 1e12,
            "B": np.random.randn(10) * 1e10 + 1e12,
            "C": np.random.randn(10) * 1e10 + 1e12,
        })

        # ts_mean should still be accurate
        result = ts_mean(df, 5)
        for col in ["A", "B", "C"]:
            pd_series = pd.Series(df[col].to_numpy())
            expected = pd_series.rolling(window=5, min_periods=1).mean()

            np.testing.assert_allclose(
                result[col].to_numpy(), expected.values,
                rtol=1e-9,
                err_msg=f"ts_mean large values mismatch for {col}"
            )

    def test_small_values_numerical_stability(self) -> None:
        """Operators should maintain accuracy with small values."""
        np.random.seed(42)
        dates = pl.date_range(date(2024, 1, 1), date(2024, 1, 10), eager=True)

        # Small values
        df = pl.DataFrame({
            "Date": dates,
            "A": np.random.randn(10) * 1e-10,
            "B": np.random.randn(10) * 1e-10,
            "C": np.random.randn(10) * 1e-10,
        })

        # ts_mean should still be accurate
        result = ts_mean(df, 5)
        for col in ["A", "B", "C"]:
            pd_series = pd.Series(df[col].to_numpy())
            expected = pd_series.rolling(window=5, min_periods=1).mean()

            np.testing.assert_allclose(
                result[col].to_numpy(), expected.values,
                rtol=1e-9,
                err_msg=f"ts_mean small values mismatch for {col}"
            )
