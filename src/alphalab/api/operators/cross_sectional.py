"""Cross-sectional operators for wide tables.

All operators work row-wise across symbols at each date:
- First column (date) is unchanged
- Operations applied across symbol columns within each row
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import polars as pl
from scipy import stats


def _get_value_cols(df: pl.DataFrame) -> list[str]:
    """Get value columns (all except first which is date)."""
    return df.columns[1:]


# Module-level RNG for reproducibility
_bucket_rank_rng = np.random.default_rng(seed=42)

# Thread pool for row-parallel operations
_RANK_MAX_WORKERS = 8


def _rank_row(row: np.ndarray) -> np.ndarray:
    """Rank a single row using argsort (faster than scipy.stats.rankdata).

    Args:
        row: 1D array of values

    Returns:
        Array of rank values in [0, 1], NaN for invalid inputs
    """
    mask = ~np.isnan(row)
    n_valid = mask.sum()
    result = np.full_like(row, np.nan)

    if n_valid == 0:
        return result
    if n_valid == 1:
        result[mask] = 0.0
        return result

    valid_vals = row[mask]
    order = np.argsort(valid_vals)
    ranks = np.empty(n_valid)
    ranks[order] = np.arange(n_valid)
    result[mask] = ranks / (n_valid - 1)
    return result


def _bucket_rank(values: np.ndarray, rate: int) -> np.ndarray:
    """Compute approximate rank using bucket-based method.

    Args:
        values: Array of values to rank
        rate: Controls number of buckets (n_buckets ≈ n_valid / 2^rate)

    Returns:
        Array of rank values in [0, 1], NaN for invalid inputs
    """
    valid_mask = ~np.isnan(values)
    n_valid = valid_mask.sum()

    if n_valid <= 1:
        result = np.zeros(len(values), dtype=np.float64)
        result[~valid_mask] = np.nan
        return result

    # Number of buckets based on rate
    n_buckets = max(2, int(n_valid / (2 ** rate)))

    valid_vals = values[valid_mask]

    # Random sample for pivot selection (unbiased quantile estimation)
    sample_size = min(n_buckets, n_valid)
    sample_idx = _bucket_rank_rng.choice(n_valid, size=sample_size, replace=False)
    sorted_sample = np.sort(valid_vals[sample_idx])
    thresholds = sorted_sample

    # Assign each value to a bucket via searchsorted
    bucket_indices = np.searchsorted(thresholds, valid_vals, side="right")

    # Normalize to [0, 1]
    result = np.zeros(len(values), dtype=np.float64)
    result[valid_mask] = bucket_indices / n_buckets
    result[~valid_mask] = np.nan

    return result


def _quantile_transform(values: np.ndarray, driver: str, sigma: float) -> np.ndarray:
    """Apply quantile transformation to values.

    Args:
        values: Array of values to transform
        driver: Distribution type: "gaussian", "uniform", "cauchy"
        sigma: Scale parameter for the output

    Returns:
        Array of transformed values, NaN for invalid inputs
    """
    valid_mask = ~np.isnan(values)
    n_valid = valid_mask.sum()

    if n_valid <= 1:
        result = np.zeros(len(values), dtype=np.float64)
        result[~valid_mask] = np.nan
        return result

    valid_vals = values[valid_mask]

    # Step 1: Rank to [0, 1]
    ranks = stats.rankdata(valid_vals, method="ordinal")
    ranks = (ranks - 1) / (n_valid - 1)  # [0, 1]

    # Step 2: Shift to [1/N, 1-1/N]
    shifted = 1 / n_valid + ranks * (1 - 2 / n_valid)

    # Step 3: Apply inverse CDF
    if driver == "gaussian":
        transformed = stats.norm.ppf(shifted) * sigma
    elif driver == "uniform":
        transformed = (shifted - 0.5) * 2 * sigma  # [-sigma, sigma]
    elif driver == "cauchy":
        transformed = stats.cauchy.ppf(shifted) * sigma
    else:
        raise ValueError(f"Unknown driver: {driver}")

    result = np.zeros(len(values), dtype=np.float64)
    result[valid_mask] = transformed
    result[~valid_mask] = np.nan

    return result


def rank(x: pl.DataFrame, rate: int = 2) -> pl.DataFrame:
    """Cross-sectional rank within each row (date).

    Ranks values across symbols and returns floats in [0.0, 1.0].
    When rate=0, uses precise sorting. Higher rate values use bucket-based
    approximate ranking for better performance on large datasets.

    This operator may help reduce outliers and drawdown while improving Sharpe.

    Args:
        x: Wide DataFrame with date + symbol columns
        rate: Controls ranking precision (default: 2).
            rate=0: Precise sorting O(N log N)
            rate>0: Bucket-based approx ranking O(N log B) where B ≈ N/2^rate

    Returns:
        Wide DataFrame with rank values in [0.0, 1.0]

    Examples:
        >>> rank(close)  # Default approximate ranking
        >>> rank(close, rate=0)  # Precise ranking
        >>> # X = (4,3,6,10,2) => rank(x) = (0.5, 0.25, 0.75, 1.0, 0.0)
    """
    date_col = x.columns[0]
    value_cols = _get_value_cols(x)
    n_symbols = len(value_cols)

    # Extract values as numpy array (rows × symbols)
    values = x.select(value_cols).to_numpy().astype(np.float64)

    # Choose ranking method
    use_precise = rate == 0 or n_symbols < 32

    if use_precise:
        # Parallel precise ranking using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=_RANK_MAX_WORKERS) as executor:
            results = list(executor.map(_rank_row, values))
        result = np.array(results)
    else:
        # Parallel bucket ranking
        result = _bucket_rank_2d(values, rate)

    # Rebuild DataFrame efficiently
    return pl.DataFrame({
        date_col: x[date_col],
        **{col: result[:, j] for j, col in enumerate(value_cols)}
    })


def _bucket_rank_row(row: np.ndarray, rate: int) -> np.ndarray:
    """Bucket rank a single row.

    Args:
        row: 1D array of values
        rate: Controls number of buckets (n_buckets ≈ n_valid / 2^rate)

    Returns:
        Array of rank values in [0, 1], NaN for invalid inputs
    """
    mask = ~np.isnan(row)
    n_valid = mask.sum()
    result = np.full_like(row, np.nan)

    if n_valid == 0:
        return result
    if n_valid <= 1:
        result[mask] = 0.0
        return result

    # Number of buckets based on rate
    n_buckets = max(2, int(n_valid / (2 ** rate)))

    valid_vals = row[mask]

    # Random sample for pivot selection (use thread-local RNG)
    rng = np.random.default_rng()
    sample_size = min(n_buckets, n_valid)
    sample_idx = rng.choice(n_valid, size=sample_size, replace=False)
    thresholds = np.sort(valid_vals[sample_idx])

    # Assign each value to a bucket via searchsorted
    bucket_indices = np.searchsorted(thresholds, valid_vals, side="right")
    result[mask] = bucket_indices / n_buckets

    return result


def _bucket_rank_2d(values: np.ndarray, rate: int) -> np.ndarray:
    """Parallel bucket ranking for 2D array (rows x cols).

    Processes rows in parallel using ThreadPoolExecutor.
    """
    def rank_with_rate(row: np.ndarray) -> np.ndarray:
        return _bucket_rank_row(row, rate)

    with ThreadPoolExecutor(max_workers=_RANK_MAX_WORKERS) as executor:
        results = list(executor.map(rank_with_rate, values))

    return np.array(results)


def zscore(x: pl.DataFrame) -> pl.DataFrame:
    """Cross-sectional z-score within each row (date).

    Computes (x - mean) / std across symbols for each date.
    Uses numpy for fast row-wise operations.

    Args:
        x: Wide DataFrame with date + symbol columns

    Returns:
        Wide DataFrame with z-scored values
    """
    date_col = x.columns[0]
    value_cols = _get_value_cols(x)

    # Use numpy for fast row-wise operations
    values = x.select(value_cols).to_numpy()

    row_mean = np.nanmean(values, axis=1, keepdims=True)
    row_std = np.nanstd(values, axis=1, keepdims=True)
    # Avoid division by zero
    row_std = np.where(row_std == 0, np.nan, row_std)

    result = (values - row_mean) / row_std

    return pl.DataFrame({
        date_col: x[date_col],
        **{col: result[:, j] for j, col in enumerate(value_cols)}
    })


def scale(
    x: pl.DataFrame,
    scale_factor: float = 1.0,
    longscale: float = 0.0,
    shortscale: float = 0.0,
) -> pl.DataFrame:
    """Scale values so that sum of absolute values equals target book size.

    Scales the input to the book size. The default scales so that sum(abs(x))
    equals 1. Use `scale_factor` parameter to set a different book size.

    For separate long/short scaling, use `longscale` and `shortscale` parameters
    to scale positive and negative positions independently.

    This operator may help reduce outliers.
    Uses numpy for fast row-wise operations.

    Args:
        x: Wide DataFrame with date + symbol columns
        scale_factor: Target sum of absolute values (default: 1.0). When longscale or
            shortscale are specified, this is ignored.
        longscale: Target sum of positive values (default: 0.0, meaning no scaling).
            When > 0, positive values are scaled so their sum equals this value.
        shortscale: Target sum of absolute negative values (default: 0.0, meaning
            no scaling). When > 0, negative values are scaled so sum(abs(neg)) equals
            this value.

    Returns:
        Wide DataFrame with scaled values

    Examples:
        >>> scale(returns, scale_factor=4)  # Scale to book size 4
        >>> scale(returns, scale_factor=1) + scale(close, scale_factor=20)
        >>> scale(returns, longscale=4, shortscale=3)  # Asymmetric long/short scaling
    """
    date_col = x.columns[0]
    value_cols = _get_value_cols(x)

    # Use numpy for fast row-wise operations
    values = x.select(value_cols).to_numpy().astype(np.float64)

    # Check if using long/short scaling
    use_asymmetric = longscale > 0 or shortscale > 0

    if use_asymmetric:
        # Sum of positive values per row
        pos_mask = values > 0
        long_sum = np.nansum(np.where(pos_mask, values, 0), axis=1, keepdims=True)

        # Sum of absolute negative values per row
        neg_mask = values < 0
        short_sum = np.nansum(np.where(neg_mask, -values, 0), axis=1, keepdims=True)

        # Scale factors (avoid division by zero)
        long_factor = np.where(long_sum > 0, longscale / long_sum, 0.0)
        short_factor = np.where(short_sum > 0, shortscale / short_sum, 0.0)

        # Apply scaling
        result = np.where(pos_mask, values * long_factor,
                         np.where(neg_mask, values * short_factor, 0.0))
    else:
        # Standard scaling: sum of absolute values equals scale_factor
        abs_sum = np.nansum(np.abs(values), axis=1, keepdims=True)
        # Avoid division by zero
        abs_sum = np.where(abs_sum == 0, np.nan, abs_sum)
        result = values * scale_factor / abs_sum

    return pl.DataFrame({
        date_col: x[date_col],
        **{col: result[:, j] for j, col in enumerate(value_cols)}
    })


def normalize(
    x: pl.DataFrame,
    useStd: bool = False,
    limit: float = 0.0,
) -> pl.DataFrame:
    """Cross-sectional normalization within each row (date).

    Subtracts row mean from each value. Optionally divides by std and clips.

    Args:
        x: Wide DataFrame with date + symbol columns
        useStd: If True, divide by std after subtracting mean
        limit: If > 0, clip values to [-limit, +limit]

    Returns:
        Wide DataFrame with normalized values

    Examples:
        >>> # x = [3,5,6,2], mean=4, std=1.82
        >>> normalize(x)  # [-1,1,2,-2]
        >>> normalize(x, useStd=True)  # [-0.55,0.55,1.1,-1.1]
    """
    date_col = x.columns[0]
    value_cols = _get_value_cols(x)

    # Use numpy for fast row-wise operations
    values = x.select(value_cols).to_numpy()

    # Row mean (ignoring NaN)
    row_mean = np.nanmean(values, axis=1, keepdims=True)
    result = values - row_mean

    if useStd:
        row_std = np.nanstd(values, axis=1, keepdims=True)
        # Avoid division by zero
        row_std = np.where(row_std == 0, np.nan, row_std)
        result = result / row_std

    if limit > 0:
        result = np.clip(result, -limit, limit)

    # Rebuild DataFrame
    return pl.DataFrame({
        date_col: x[date_col],
        **{col: result[:, j] for j, col in enumerate(value_cols)}
    })


def quantile(
    x: pl.DataFrame,
    driver: str = "gaussian",
    sigma: float = 1.0,
) -> pl.DataFrame:
    """Cross-sectional quantile transformation.

    Ranks input, shifts to avoid boundary issues, then applies distribution.
    This operator may help reduce outliers.

    Steps:
        1. Rank values to [0, 1]
        2. Shift: alpha = 1/N + alpha * (1 - 2/N) -> [1/N, 1-1/N]
        3. Apply inverse CDF of specified distribution

    Args:
        x: Wide DataFrame with date + symbol columns
        driver: Distribution type: "gaussian", "uniform", "cauchy"
        sigma: Scale parameter for the output

    Returns:
        Wide DataFrame with quantile-transformed values
    """
    date_col = x.columns[0]
    value_cols = _get_value_cols(x)

    # Extract values as numpy array (rows × symbols)
    values = x.select(value_cols).to_numpy().astype(np.float64)

    # Apply transform row-by-row in parallel
    def transform_row(row: np.ndarray) -> np.ndarray:
        return _quantile_transform(row, driver, sigma)

    with ThreadPoolExecutor(max_workers=_RANK_MAX_WORKERS) as executor:
        results = list(executor.map(transform_row, values))
    result = np.array(results)

    # Rebuild DataFrame
    return pl.DataFrame({
        date_col: x[date_col],
        **{col: result[:, j] for j, col in enumerate(value_cols)}
    })


def winsorize(x: pl.DataFrame, std: float = 4.0) -> pl.DataFrame:
    """Cross-sectional winsorization within each row (date).

    Clips values to [mean - std*SD, mean + std*SD].
    Uses numpy for fast row-wise operations.

    Args:
        x: Wide DataFrame with date + symbol columns
        std: Number of standard deviations for limits (default: 4)

    Returns:
        Wide DataFrame with winsorized values

    Examples:
        >>> # x = (2,4,5,6,3,8,10), mean=5.42, SD=2.61
        >>> winsorize(x, std=1)  # (2.81,4,5,6,3,8,8.03)
    """
    date_col = x.columns[0]
    value_cols = _get_value_cols(x)

    # Use numpy for fast row-wise operations
    values = x.select(value_cols).to_numpy()

    row_mean = np.nanmean(values, axis=1, keepdims=True)
    row_std = np.nanstd(values, axis=1, keepdims=True)

    lower = row_mean - std * row_std
    upper = row_mean + std * row_std

    result = np.clip(values, lower, upper)

    return pl.DataFrame({
        date_col: x[date_col],
        **{col: result[:, j] for j, col in enumerate(value_cols)}
    })


def bucket(x: pl.DataFrame, range_spec: str) -> pl.DataFrame:
    """Assign values to discrete buckets based on range specification.

    Buckets values into discrete bins. Each value is assigned the lower bound
    of the bucket it falls into. Values outside the range are clipped to the
    nearest bucket.

    Args:
        x: Wide DataFrame with date + symbol columns
        range_spec: Comma-separated "start,end,step" (e.g., "0,1,0.25")
            Creates buckets: [0, 0.25), [0.25, 0.5), [0.5, 0.75), [0.75, 1.0]

    Returns:
        Wide DataFrame with bucket lower bounds as values

    Examples:
        >>> # Bucket ranked values into quartiles
        >>> bucket(rank(prices), range_spec="0,1,0.25")
        >>> # Values: 0.1 -> 0.0, 0.3 -> 0.25, 0.6 -> 0.5, 0.9 -> 0.75
    """
    # Parse range_spec
    parts = range_spec.split(",")
    if len(parts) != 3:
        raise ValueError(f"range_spec must be 'start,end,step', got: {range_spec}")

    start, end, step = float(parts[0]), float(parts[1]), float(parts[2])

    if step <= 0:
        raise ValueError(f"step must be positive, got: {step}")

    date_col = x.columns[0]
    value_cols = _get_value_cols(x)

    # Use numpy for fast vectorized bucketing
    values = x.select(value_cols).to_numpy().astype(np.float64)

    # floor((value - start) / step) * step + start, clipped to [start, end - step]
    result = np.floor((values - start) / step) * step + start
    result = np.clip(result, start, end - step)

    return pl.DataFrame({
        date_col: x[date_col],
        **{col: result[:, j] for j, col in enumerate(value_cols)}
    })
