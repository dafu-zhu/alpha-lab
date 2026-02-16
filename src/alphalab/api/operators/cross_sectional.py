"""Cross-sectional operators for wide tables.

All operators work row-wise across symbols at each date:
- First column (date) is unchanged
- Operations applied across symbol columns within each row
"""

import numpy as np
import polars as pl
from scipy import stats


def _get_value_cols(df: pl.DataFrame) -> list[str]:
    """Get value columns (all except first which is date)."""
    return df.columns[1:]


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

    # Convert to long format
    long = x.unpivot(
        index=date_col,
        on=value_cols,
        variable_name="symbol",
        value_name="value",
    )

    # Use precise ranking when rate=0 or when dataset is small (< 32 items)
    # Bucket-based ranking is only beneficial for large universes
    n_symbols = len(value_cols)
    if rate == 0 or n_symbols < 32:
        # Precise ranking using ordinal method
        ranked = long.with_columns(
            ((pl.col("value").rank(method="ordinal").over(date_col) - 1)
             / (pl.col("value").count().over(date_col) - 1))
            .alias("value")
        )
    else:
        # Bucket-based approximate ranking
        # Logic:
        # 1. Sample M random pivots from N values (M ≈ N/2^rate)
        # 2. Sort only the sampled pivots to get bucket thresholds
        # 3. Assign each value to bucket via binary search O(log M)
        # 4. Normalize bucket index to [0, 1]
        # Total: O(M log M + N log M) vs O(N log N) for full sort
        # Random sampling gives unbiased quantile estimates
        rng = np.random.default_rng(seed=42)  # Fixed seed for reproducibility

        def bucket_rank(values: np.ndarray, rate: int) -> np.ndarray:
            """Compute approximate rank using bucket-based method."""
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
            sample_idx = rng.choice(n_valid, size=sample_size, replace=False)
            sorted_sample = np.sort(valid_vals[sample_idx])
            thresholds = sorted_sample

            # Assign each value to a bucket via searchsorted
            bucket_indices = np.searchsorted(thresholds, valid_vals, side="right")

            # Normalize to [0, 1]
            result = np.zeros(len(values), dtype=np.float64)
            result[valid_mask] = bucket_indices / n_buckets
            result[~valid_mask] = np.nan

            return result

        # Apply bucket ranking per date group
        ranked = long.with_columns(
            pl.col("value")
            .map_batches(lambda s: pl.Series(bucket_rank(s.to_numpy(), rate)), return_dtype=pl.Float64)
            .over(date_col)
            .alias("value")
        )

    # Pivot back to wide
    wide = ranked.pivot(values="value", index=date_col, on="symbol")

    # Ensure column order matches input
    return wide.select([date_col, *value_cols])


def zscore(x: pl.DataFrame) -> pl.DataFrame:
    """Cross-sectional z-score within each row (date).

    Computes (x - mean) / std across symbols for each date.

    Args:
        x: Wide DataFrame with date + symbol columns

    Returns:
        Wide DataFrame with z-scored values
    """
    value_cols = _get_value_cols(x)

    # Compute row-wise mean and std
    row_mean = pl.mean_horizontal(*[pl.col(c) for c in value_cols])
    row_std = pl.concat_list([pl.col(c) for c in value_cols]).list.eval(
        pl.element().std()
    ).list.first()

    return x.with_columns([
        ((pl.col(c) - row_mean) / row_std).alias(c)
        for c in value_cols
    ])


def scale(
    x: pl.DataFrame,
    scale: float = 1.0,
    longscale: float = 0.0,
    shortscale: float = 0.0,
) -> pl.DataFrame:
    """Scale values so that sum of absolute values equals target book size.

    Scales the input to the book size. The default scales so that sum(abs(x))
    equals 1. Use `scale` parameter to set a different book size.

    For separate long/short scaling, use `longscale` and `shortscale` parameters
    to scale positive and negative positions independently.

    This operator may help reduce outliers.

    Args:
        x: Wide DataFrame with date + symbol columns
        scale: Target sum of absolute values (default: 1.0). When longscale or
            shortscale are specified, this is ignored.
        longscale: Target sum of positive values (default: 0.0, meaning no scaling).
            When > 0, positive values are scaled so their sum equals this value.
        shortscale: Target sum of absolute negative values (default: 0.0, meaning
            no scaling). When > 0, negative values are scaled so sum(abs(neg)) equals
            this value.

    Returns:
        Wide DataFrame with scaled values

    Examples:
        >>> scale(returns, scale=4)  # Scale to book size 4
        >>> scale(returns, scale=1) + scale(close, scale=20)  # Combine scaled alphas
        >>> scale(returns, longscale=4, shortscale=3)  # Asymmetric long/short scaling
    """
    value_cols = _get_value_cols(x)

    # Check if using long/short scaling
    use_asymmetric = longscale > 0 or shortscale > 0

    if use_asymmetric:
        # Scale long and short positions separately
        # Sum of positive values across row
        long_sum = pl.sum_horizontal(
            *[pl.when(pl.col(c) > 0).then(pl.col(c)).otherwise(0.0) for c in value_cols]
        )
        # Sum of absolute negative values across row
        short_sum = pl.sum_horizontal(
            *[pl.when(pl.col(c) < 0).then(-pl.col(c)).otherwise(0.0) for c in value_cols]
        )

        # Scale factors (avoid division by zero)
        long_factor = pl.when(long_sum > 0).then(longscale / long_sum).otherwise(0.0)
        short_factor = pl.when(short_sum > 0).then(shortscale / short_sum).otherwise(0.0)

        return x.with_columns([
            pl.when(pl.col(c) > 0)
            .then(pl.col(c) * long_factor)
            .when(pl.col(c) < 0)
            .then(pl.col(c) * short_factor)
            .otherwise(0.0)
            .alias(c)
            for c in value_cols
        ])
    else:
        # Standard scaling: sum of absolute values equals scale
        abs_sum = pl.sum_horizontal(*[pl.col(c).abs() for c in value_cols])

        return x.with_columns([
            (pl.col(c) * scale / abs_sum).alias(c)
            for c in value_cols
        ])


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
    value_cols = _get_value_cols(x)

    row_mean = pl.mean_horizontal(*[pl.col(c) for c in value_cols])

    if useStd:
        row_std = pl.concat_list([pl.col(c) for c in value_cols]).list.eval(
            pl.element().std()
        ).list.first()
        result = x.with_columns([
            ((pl.col(c) - row_mean) / row_std).alias(c)
            for c in value_cols
        ])
    else:
        result = x.with_columns([
            (pl.col(c) - row_mean).alias(c)
            for c in value_cols
        ])

    if limit > 0:
        result = result.with_columns([
            pl.col(c).clip(-limit, limit).alias(c)
            for c in value_cols
        ])

    return result


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

    # Convert to long format
    long = x.unpivot(
        index=date_col,
        on=value_cols,
        variable_name="symbol",
        value_name="value",
    )

    def quantile_transform(values: np.ndarray) -> np.ndarray:
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

    # Apply transform per date group
    transformed = long.with_columns(
        pl.col("value")
        .map_batches(lambda s: pl.Series(quantile_transform(s.to_numpy())), return_dtype=pl.Float64)
        .over(date_col)
        .alias("value")
    )

    # Pivot back to wide
    wide = transformed.pivot(values="value", index=date_col, on="symbol")

    return wide.select([date_col, *value_cols])


def winsorize(x: pl.DataFrame, std: float = 4.0) -> pl.DataFrame:
    """Cross-sectional winsorization within each row (date).

    Clips values to [mean - std*SD, mean + std*SD].

    Args:
        x: Wide DataFrame with date + symbol columns
        std: Number of standard deviations for limits (default: 4)

    Returns:
        Wide DataFrame with winsorized values

    Examples:
        >>> # x = (2,4,5,6,3,8,10), mean=5.42, SD=2.61
        >>> winsorize(x, std=1)  # (2.81,4,5,6,3,8,8.03)
    """
    value_cols = _get_value_cols(x)

    row_mean = pl.mean_horizontal(*[pl.col(c) for c in value_cols])
    row_std = pl.concat_list([pl.col(c) for c in value_cols]).list.eval(
        pl.element().std()
    ).list.first()

    lower = row_mean - std * row_std
    upper = row_mean + std * row_std

    return x.with_columns([
        pl.col(c).clip(lower, upper).alias(c)
        for c in value_cols
    ])


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

    value_cols = _get_value_cols(x)

    # Create bucket boundaries
    # Values are assigned to floor((value - start) / step) * step + start
    # Clipped to [start, end - step]
    return x.with_columns([
        (
            (((pl.col(c) - start) / step).floor() * step + start)
            .clip(start, end - step)
        ).alias(c)
        for c in value_cols
    ])
