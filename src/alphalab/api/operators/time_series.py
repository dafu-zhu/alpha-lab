"""Time-series operators for wide tables.

All operators preserve the wide table structure:
- First column (date) is unchanged
- Operations applied column-wise to symbol columns
"""

import math

import polars as pl


def _get_value_cols(df: pl.DataFrame) -> list[str]:
    """Get value columns (all except first which is date)."""
    return df.columns[1:]


# =============================================================================
# Module-level helper functions (extracted for testability)
# =============================================================================


def _arg_max_fn(s: pl.Series, d: int) -> float | None:
    """Find days since max in rolling window.

    Args:
        s: Series of values
        d: Window size

    Returns:
        Days since max (0 = today, d-1 = oldest), None if window incomplete
    """
    if len(s) < d:
        return None
    idx = s.arg_max()
    if idx is None:
        return None
    # Convert to "days since": 0 = today (newest), d-1 = oldest
    return float((d - 1) - idx)


def _arg_min_fn(s: pl.Series, d: int) -> float | None:
    """Find days since min in rolling window.

    Args:
        s: Series of values
        d: Window size

    Returns:
        Days since min (0 = today, d-1 = oldest), None if window incomplete
    """
    if len(s) < d:
        return None
    idx = s.arg_min()
    if idx is None:
        return None
    # Convert to "days since": 0 = today (newest), d-1 = oldest
    return float((d - 1) - idx)


def _find_last_diff(s: pl.Series) -> float | None:
    """Find last value different from current in series.

    Args:
        s: Series of values

    Returns:
        Last different value, None if not found or series too short
    """
    if len(s) < 2:
        return None
    current = s[-1]
    for i in range(len(s) - 2, -1, -1):
        if s[i] != current and s[i] is not None:
            return float(s[i])
    return None


def _inv_norm(p: float) -> float:
    """Approximate inverse normal CDF (Abramowitz and Stegun approximation).

    Args:
        p: Probability value in (0, 1)

    Returns:
        Inverse normal CDF value
    """
    if p <= 0:
        return float("-inf")
    if p >= 1:
        return float("inf")
    # Abramowitz and Stegun approximation
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d_coef = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]
    p_low = 0.02425
    p_high = 1 - p_low
    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (
            ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        ) / ((((d_coef[0] * q + d_coef[1]) * q + d_coef[2]) * q + d_coef[3]) * q + 1)
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        return (
            ((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]
        ) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    else:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(
            ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        ) / ((((d_coef[0] * q + d_coef[1]) * q + d_coef[2]) * q + d_coef[3]) * q + 1)


def _ts_quantile_transform(s: pl.Series, driver: str) -> float | None:
    """Transform series values using quantile transform.

    Args:
        s: Series of values
        driver: "gaussian" or "uniform"

    Returns:
        Transformed value, None if current is None, 0.0 if single value
    """
    vals = s.to_list()
    current = vals[-1]
    if current is None:
        return None
    sorted_vals = sorted([v for v in vals if v is not None])
    if len(sorted_vals) <= 1:
        return 0.0
    idx = sorted_vals.index(current)
    rank_pct = (idx + 0.5) / len(sorted_vals)
    if driver == "gaussian":
        return _inv_norm(rank_pct)
    else:  # uniform
        return rank_pct * 2 - 1


def ts_mean(x: pl.DataFrame, d: int) -> pl.DataFrame:
    """Rolling mean over d periods (partial windows allowed).

    Args:
        x: Wide DataFrame with date + symbol columns
        d: Window size in periods

    Returns:
        Wide DataFrame with rolling mean values
    """
    date_col = x.columns[0]
    value_cols = _get_value_cols(x)
    return x.select(
        pl.col(date_col),
        *[pl.col(c).rolling_mean(window_size=d, min_samples=1).alias(c) for c in value_cols],
    )


def ts_sum(x: pl.DataFrame, d: int) -> pl.DataFrame:
    """Rolling sum over d periods (partial windows allowed).

    Args:
        x: Wide DataFrame with date + symbol columns
        d: Window size in periods

    Returns:
        Wide DataFrame with rolling sum values
    """
    date_col = x.columns[0]
    value_cols = _get_value_cols(x)
    return x.select(
        pl.col(date_col),
        *[pl.col(c).rolling_sum(window_size=d, min_samples=1).alias(c) for c in value_cols],
    )


def ts_std(x: pl.DataFrame, d: int) -> pl.DataFrame:
    """Rolling standard deviation over d periods (partial windows allowed, min 2 for std).

    Args:
        x: Wide DataFrame with date + symbol columns
        d: Window size in periods

    Returns:
        Wide DataFrame with rolling std values
    """
    date_col = x.columns[0]
    value_cols = _get_value_cols(x)
    return x.select(
        pl.col(date_col),
        *[pl.col(c).rolling_std(window_size=d, min_samples=2).alias(c) for c in value_cols],
    )


def ts_min(x: pl.DataFrame, d: int) -> pl.DataFrame:
    """Rolling minimum over d periods (partial windows allowed).

    Args:
        x: Wide DataFrame with date + symbol columns
        d: Window size in periods

    Returns:
        Wide DataFrame with rolling min values
    """
    date_col = x.columns[0]
    value_cols = _get_value_cols(x)
    return x.select(
        pl.col(date_col),
        *[pl.col(c).rolling_min(window_size=d, min_samples=1).alias(c) for c in value_cols],
    )


def ts_max(x: pl.DataFrame, d: int) -> pl.DataFrame:
    """Rolling maximum over d periods (partial windows allowed).

    Args:
        x: Wide DataFrame with date + symbol columns
        d: Window size in periods

    Returns:
        Wide DataFrame with rolling max values
    """
    date_col = x.columns[0]
    value_cols = _get_value_cols(x)
    return x.select(
        pl.col(date_col),
        *[pl.col(c).rolling_max(window_size=d, min_samples=1).alias(c) for c in value_cols],
    )


def ts_delta(
    x: pl.DataFrame, d: int = 1, lookback: pl.DataFrame | None = None
) -> pl.DataFrame:
    """Difference from d periods ago: x - ts_delay(x, d).

    Args:
        x: Wide DataFrame with date + symbol columns
        d: Lag periods (default: 1)
        lookback: Optional DataFrame with prior rows to avoid nulls at start.
                  If provided, uses lookback data for computing initial deltas,
                  then returns only rows from x.

    Returns:
        Wide DataFrame with differenced values
    """
    date_col = x.columns[0]
    value_cols = _get_value_cols(x)

    if lookback is not None:
        # Concatenate lookback + x, compute, then trim to x's rows
        combined = pl.concat([lookback, x])
        result = combined.select(
            pl.col(date_col),
            *[pl.col(c).diff(d).alias(c) for c in value_cols],
        )
        # Return only the rows corresponding to x (last len(x) rows)
        return result.tail(len(x))

    return x.select(
        pl.col(date_col),
        *[pl.col(c).diff(d).alias(c) for c in value_cols],
    )


def ts_delay(
    x: pl.DataFrame, d: int, lookback: pl.DataFrame | None = None
) -> pl.DataFrame:
    """Lag values by d periods.

    Args:
        x: Wide DataFrame with date + symbol columns
        d: Number of periods to lag
        lookback: Optional DataFrame with prior rows to avoid nulls at start.

    Returns:
        Wide DataFrame with lagged values
    """
    date_col = x.columns[0]
    value_cols = _get_value_cols(x)

    if lookback is not None:
        combined = pl.concat([lookback, x])
        result = combined.select(
            pl.col(date_col),
            *[pl.col(c).shift(d).alias(c) for c in value_cols],
        )
        return result.tail(len(x))

    return x.select(
        pl.col(date_col),
        *[pl.col(c).shift(d).alias(c) for c in value_cols],
    )


# Phase 1: Simple Rolling Ops


def ts_product(x: pl.DataFrame, d: int) -> pl.DataFrame:
    """Rolling product over d periods (partial windows allowed).

    Uses O(n) online algorithm with log transform for numerical stability.
    ThreadPoolExecutor parallelizes across columns.

    Args:
        x: Wide DataFrame with date + symbol columns
        d: Window size in periods

    Returns:
        Wide DataFrame with rolling product values
    """
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor

    from alphalab.api.operators._numba_kernels import rolling_product_online

    date_col = x.columns[0]
    value_cols = _get_value_cols(x)

    def process_col(c: str) -> tuple[str, np.ndarray]:
        x_arr = x[c].to_numpy().astype(np.float64)
        return (c, rolling_product_online(x_arr, d))

    with ThreadPoolExecutor(max_workers=min(8, len(value_cols))) as executor:
        col_results = dict(executor.map(process_col, value_cols))

    return pl.DataFrame({date_col: x[date_col], **col_results})


def ts_count_nans(x: pl.DataFrame, d: int) -> pl.DataFrame:
    """Count nulls in rolling window of d periods (partial windows allowed)."""
    date_col = x.columns[0]
    value_cols = _get_value_cols(x)
    return x.select(
        pl.col(date_col),
        *[
            pl.col(c).is_null().cast(pl.Int64).rolling_sum(window_size=d, min_samples=1).alias(c)
            for c in value_cols
        ],
    )


def ts_zscore(x: pl.DataFrame, d: int) -> pl.DataFrame:
    """Rolling z-score: (x - rolling_mean) / rolling_std (partial windows, min 2 for std)."""
    date_col = x.columns[0]
    value_cols = _get_value_cols(x)
    return x.select(
        pl.col(date_col),
        *[
            (
                (pl.col(c) - pl.col(c).rolling_mean(window_size=d, min_samples=1))
                / pl.col(c).rolling_std(window_size=d, min_samples=2)
            ).alias(c)
            for c in value_cols
        ],
    )


def ts_scale(x: pl.DataFrame, d: int, constant: float = 0) -> pl.DataFrame:
    """Scale to [constant, 1+constant] based on rolling min/max (partial windows, min 2)."""
    date_col = x.columns[0]
    value_cols = _get_value_cols(x)
    return x.select(
        pl.col(date_col),
        *[
            (
                (pl.col(c) - pl.col(c).rolling_min(window_size=d, min_samples=2))
                / (pl.col(c).rolling_max(window_size=d, min_samples=2) - pl.col(c).rolling_min(window_size=d, min_samples=2))
                + constant
            ).alias(c)
            for c in value_cols
        ],
    )


def ts_av_diff(x: pl.DataFrame, d: int) -> pl.DataFrame:
    """Difference from rolling mean: x - rolling_mean(x, d)."""
    date_col = x.columns[0]
    value_cols = _get_value_cols(x)
    return x.select(
        pl.col(date_col),
        *[(pl.col(c) - pl.col(c).rolling_mean(d)).alias(c) for c in value_cols],
    )


def ts_step(x: pl.DataFrame) -> pl.DataFrame:
    """Row counter: 1, 2, 3, ..."""
    date_col = x.columns[0]
    value_cols = _get_value_cols(x)
    n = len(x)
    step_values = list(range(1, n + 1))
    return x.select(
        pl.col(date_col),
        *[pl.lit(step_values).alias(c).explode() for c in value_cols],
    )


# Phase 2: Arg Ops


def ts_arg_max(x: pl.DataFrame, d: int) -> pl.DataFrame:
    """Days since max in rolling window (0 = today is max, d-1 = oldest day was max)."""
    date_col = x.columns[0]
    value_cols = _get_value_cols(x)

    return x.select(
        pl.col(date_col),
        *[pl.col(c).rolling_map(lambda s: _arg_max_fn(s, d), window_size=d).alias(c) for c in value_cols],
    )


def ts_arg_min(x: pl.DataFrame, d: int) -> pl.DataFrame:
    """Days since min in rolling window (0 = today is min, d-1 = oldest day was min)."""
    date_col = x.columns[0]
    value_cols = _get_value_cols(x)

    return x.select(
        pl.col(date_col),
        *[pl.col(c).rolling_map(lambda s: _arg_min_fn(s, d), window_size=d).alias(c) for c in value_cols],
    )


# Phase 3: Lookback/Backfill Ops


def ts_backfill(x: pl.DataFrame, d: int) -> pl.DataFrame:
    """Fill NaN with last valid value within d periods."""
    date_col = x.columns[0]
    value_cols = _get_value_cols(x)
    return x.select(
        pl.col(date_col),
        *[pl.col(c).forward_fill(limit=d).alias(c) for c in value_cols],
    )


def kth_element(x: pl.DataFrame, d: int, k: int) -> pl.DataFrame:  # noqa: ARG001
    """Get k-th element in lookback window (k=0 is current, k=1 is prev, etc)."""
    date_col = x.columns[0]
    value_cols = _get_value_cols(x)
    return x.select(
        pl.col(date_col),
        *[pl.col(c).shift(k).alias(c) for c in value_cols],
    )


def last_diff_value(x: pl.DataFrame, d: int) -> pl.DataFrame:
    """Last value different from current within d periods."""
    date_col = x.columns[0]
    value_cols = _get_value_cols(x)

    return x.select(
        pl.col(date_col),
        *[pl.col(c).rolling_map(_find_last_diff, window_size=d).alias(c) for c in value_cols],
    )


def _compute_days_from_last_change(col_data: list) -> list[int]:
    """Compute days since last change for a single column."""
    days: list[int] = []
    last_change_idx = 0
    for i, val in enumerate(col_data):
        if i == 0:
            days.append(0)
        elif val != col_data[i - 1]:
            last_change_idx = i
            days.append(0)
        else:
            days.append(i - last_change_idx)
    return days


def days_from_last_change(x: pl.DataFrame) -> pl.DataFrame:
    """Days since value changed (column-parallel)."""
    from concurrent.futures import ThreadPoolExecutor

    date_col = x.columns[0]
    value_cols = _get_value_cols(x)

    def process_col(c: str) -> tuple[str, list[int]]:
        return (c, _compute_days_from_last_change(x[c].to_list()))

    # Parallelize across columns
    with ThreadPoolExecutor(max_workers=min(8, len(value_cols))) as executor:
        col_results = dict(executor.map(process_col, value_cols))

    return pl.DataFrame({date_col: x[date_col], **col_results})


# Phase 4: Stateful Ops


def _compute_hump_col(col_data: list, row_limits: list[float], col_idx: int) -> list[float | None]:  # noqa: ARG001
    """Compute hump-limited values for a single column.

    Args:
        col_data: Values for this column
        row_limits: Pre-computed limit for each row (hump_factor * row_sum)
        col_idx: Not used but kept for consistent interface
    """
    n = len(col_data)
    out: list[float | None] = []

    for i in range(n):
        if i == 0:
            out.append(col_data[0])
        else:
            prev = out[i - 1]
            curr = col_data[i]
            limit = row_limits[i]

            if prev is None or curr is None:
                out.append(curr)
            else:
                change = curr - prev
                if abs(change) > limit:
                    out.append(prev + (1 if change > 0 else -1) * limit)
                else:
                    out.append(curr)

    return out


def hump(x: pl.DataFrame, hump: float = 0.01) -> pl.DataFrame:
    """Limit change magnitude (column-parallel where possible)."""
    from concurrent.futures import ThreadPoolExecutor

    date_col = x.columns[0]
    value_cols = _get_value_cols(x)

    # Pre-extract all column data for row_sum calculation
    all_col_data = {c: x[c].to_list() for c in value_cols}

    # Pre-compute row limits (this must be done sequentially since it depends on all columns)
    n_rows = x.height
    row_limits = []
    for i in range(n_rows):
        row_sum = sum(abs(all_col_data[c][i] or 0) for c in all_col_data)
        row_limits.append(hump * row_sum)

    def process_col(c: str) -> tuple[str, list[float | None]]:
        return (c, _compute_hump_col(all_col_data[c], row_limits, 0))

    with ThreadPoolExecutor(max_workers=min(8, len(value_cols))) as executor:
        col_results = dict(executor.map(process_col, value_cols))

    return pl.DataFrame({date_col: x[date_col], **col_results})


def ts_decay_linear(x: pl.DataFrame, d: int, dense: bool = False) -> pl.DataFrame:
    """Weighted average with linear decay weights [1, 2, ..., d].

    Uses O(n) online algorithm with numba JIT for dense=False case.
    Falls back to rolling_map for dense=True (which skips NaN values).
    """
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor
    from alphalab.api.operators._numba_kernels import rolling_decay_linear_online

    date_col = x.columns[0]
    value_cols = _get_value_cols(x)

    if dense:
        # dense=True: skip NaN values and reweight (use original implementation)
        weights = list(range(1, d + 1))

        def weighted_avg(s: pl.Series) -> float | None:
            if len(s) < d:
                return None
            vals = s.to_list()
            # Only use non-null values
            valid: list[tuple[int, float]] = [
                (w, v) for w, v in zip(weights, vals, strict=True) if v is not None
            ]
            if not valid:
                return None
            w_sum = sum(w for w, _ in valid)
            return float(sum(w * v for w, v in valid) / w_sum)

        return x.select(
            pl.col(date_col),
            *[pl.col(c).rolling_map(weighted_avg, window_size=d).alias(c) for c in value_cols],
        )
    else:
        # dense=False: use fast numba kernel (NaN propagation)
        def process_col(c: str) -> tuple[str, np.ndarray]:
            x_arr = x[c].to_numpy().astype(np.float64)
            return (c, rolling_decay_linear_online(x_arr, d))

        with ThreadPoolExecutor(max_workers=min(8, len(value_cols))) as executor:
            col_results = dict(executor.map(process_col, value_cols))

        return pl.DataFrame({date_col: x[date_col], **col_results})


def ts_rank(x: pl.DataFrame, d: int, constant: float = 0) -> pl.DataFrame:
    """Rank of current value in rolling window, scaled to [constant, 1+constant] (partial windows allowed).

    Uses O(n*d) numba-optimized kernel with ThreadPoolExecutor for column parallelism.

    Args:
        x: Wide DataFrame with date + symbol columns
        d: Window size in periods
        constant: Offset for rank scaling (default 0, gives range [0, 1])

    Returns:
        Wide DataFrame with rolling rank values in [constant, 1+constant]
    """
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor

    from alphalab.api.operators._numba_kernels import rolling_rank

    date_col = x.columns[0]
    value_cols = _get_value_cols(x)

    def process_col(c: str) -> tuple[str, np.ndarray]:
        x_arr = x[c].to_numpy().astype(np.float64)
        return (c, rolling_rank(x_arr, d, constant))

    with ThreadPoolExecutor(max_workers=min(8, len(value_cols))) as executor:
        col_results = dict(executor.map(process_col, value_cols))

    return pl.DataFrame({date_col: x[date_col], **col_results})


# Phase 5: Two-Variable Ops


def _is_invalid(v) -> bool:
    """Check if value is None or NaN."""
    return v is None or (isinstance(v, float) and math.isnan(v))


def _compute_rolling_corr(x_vals: list, y_vals: list, d: int) -> list[float | None]:
    """Compute rolling correlation for a single column pair."""
    corrs: list[float | None] = []
    for i in range(len(x_vals)):
        if i < d - 1:
            corrs.append(None)
        else:
            x_win = x_vals[i - d + 1 : i + 1]
            y_win = y_vals[i - d + 1 : i + 1]
            if any(_is_invalid(v) for v in x_win) or any(_is_invalid(v) for v in y_win):
                corrs.append(None)
            else:
                x_mean = sum(x_win) / d
                y_mean = sum(y_win) / d
                cov = sum((xv - x_mean) * (yv - y_mean) for xv, yv in zip(x_win, y_win, strict=True)) / d
                x_std = (sum((xv - x_mean) ** 2 for xv in x_win) / d) ** 0.5
                y_std = (sum((yv - y_mean) ** 2 for yv in y_win) / d) ** 0.5
                if x_std == 0 or y_std == 0:
                    corrs.append(None)
                else:
                    corrs.append(cov / (x_std * y_std))
    return corrs


def ts_corr(x: pl.DataFrame, y: pl.DataFrame, d: int) -> pl.DataFrame:
    """Rolling Pearson correlation (online algorithm + numba JIT).

    Uses O(n) online algorithm instead of O(n*d) naive approach.
    ~50-100x faster than the previous implementation.
    """
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor
    from alphalab.api.operators._numba_kernels import rolling_corr_online

    date_col = x.columns[0]
    value_cols = _get_value_cols(x)

    def process_col(c: str) -> tuple[str, np.ndarray]:
        x_arr = x[c].to_numpy().astype(np.float64)
        y_arr = y[c].to_numpy().astype(np.float64)
        return (c, rolling_corr_online(x_arr, y_arr, d))

    with ThreadPoolExecutor(max_workers=min(8, len(value_cols))) as executor:
        col_results = dict(executor.map(process_col, value_cols))

    return pl.DataFrame({date_col: x[date_col], **col_results})


def _compute_rolling_cov(x_vals: list, y_vals: list, d: int) -> list[float | None]:
    """Compute rolling covariance for a single column pair."""
    covs: list[float | None] = []
    for i in range(len(x_vals)):
        if i < d - 1:
            covs.append(None)
        else:
            x_win = x_vals[i - d + 1 : i + 1]
            y_win = y_vals[i - d + 1 : i + 1]
            if any(_is_invalid(v) for v in x_win) or any(_is_invalid(v) for v in y_win):
                covs.append(None)
            else:
                x_mean = sum(x_win) / d
                y_mean = sum(y_win) / d
                cov = sum((xv - x_mean) * (yv - y_mean) for xv, yv in zip(x_win, y_win, strict=True)) / d
                covs.append(cov)
    return covs


def ts_covariance(x: pl.DataFrame, y: pl.DataFrame, d: int) -> pl.DataFrame:
    """Rolling covariance (online algorithm + numba JIT).

    Uses O(n) online algorithm instead of O(n*d) naive approach.
    ~50-100x faster than the previous implementation.
    """
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor
    from alphalab.api.operators._numba_kernels import rolling_cov_online

    date_col = x.columns[0]
    value_cols = _get_value_cols(x)

    def process_col(c: str) -> tuple[str, np.ndarray]:
        x_arr = x[c].to_numpy().astype(np.float64)
        y_arr = y[c].to_numpy().astype(np.float64)
        return (c, rolling_cov_online(x_arr, y_arr, d))

    with ThreadPoolExecutor(max_workers=min(8, len(value_cols))) as executor:
        col_results = dict(executor.map(process_col, value_cols))

    return pl.DataFrame({date_col: x[date_col], **col_results})


def ts_quantile(x: pl.DataFrame, d: int, driver: str = "gaussian") -> pl.DataFrame:
    """Rolling quantile transform: ts_rank + inverse CDF (partial windows allowed).

    Uses O(n*d) numba-optimized kernel with ThreadPoolExecutor for column parallelism.

    Args:
        x: Wide DataFrame with date + symbol columns
        d: Window size in periods
        driver: "gaussian" (inverse normal CDF) or "uniform" (scaled to [-1, 1])

    Returns:
        Wide DataFrame with quantile-transformed values
    """
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor

    from alphalab.api.operators._numba_kernels import (
        rolling_quantile_gaussian,
        rolling_quantile_uniform,
    )

    date_col = x.columns[0]
    value_cols = _get_value_cols(x)

    kernel = rolling_quantile_gaussian if driver == "gaussian" else rolling_quantile_uniform

    def process_col(c: str) -> tuple[str, np.ndarray]:
        x_arr = x[c].to_numpy().astype(np.float64)
        return (c, kernel(x_arr, d))

    with ThreadPoolExecutor(max_workers=min(8, len(value_cols))) as executor:
        col_results = dict(executor.map(process_col, value_cols))

    return pl.DataFrame({date_col: x[date_col], **col_results})


# Phase 6: Regression


def _is_nan(v) -> bool:
    """Check if value is NaN (but not None)."""
    return isinstance(v, float) and math.isnan(v)


def _compute_regression_col(
    y_vals: list, x_vals: list, d: int, lag: int, rettype: int
) -> list[float | None]:
    """Compute rolling regression for a single column."""
    if lag > 0:
        x_vals = [None] * lag + x_vals[:-lag] if lag < len(x_vals) else [None] * len(x_vals)

    results: list[float | None] = []
    for i in range(len(y_vals)):
        start_idx = max(0, i - d + 1)
        y_win_raw = y_vals[start_idx : i + 1]
        x_win_raw = x_vals[start_idx : i + 1]

        # If any value in window is NaN, return None (consistent with ts_corr/ts_covariance)
        if any(_is_nan(v) for v in y_win_raw) or any(_is_nan(v) for v in x_win_raw):
            results.append(None)
            continue

        # Filter out None values (backward compatible with original behavior)
        pairs = [(yv, xv) for yv, xv in zip(y_win_raw, x_win_raw, strict=True)
                 if yv is not None and xv is not None]

        if len(pairs) < 2:
            results.append(None)
            continue

        y_win = [p[0] for p in pairs]
        x_win = [p[1] for p in pairs]
        n = len(pairs)

        x_mean = sum(x_win) / n
        y_mean = sum(y_win) / n

        ss_xx = sum((xv - x_mean) ** 2 for xv in x_win)
        ss_yy = sum((yv - y_mean) ** 2 for yv in y_win)
        ss_xy = sum((xv - x_mean) * (yv - y_mean) for xv, yv in zip(x_win, y_win, strict=True))

        if ss_xx == 0:
            results.append(None)
            continue

        beta = ss_xy / ss_xx
        alpha = y_mean - beta * x_mean
        y_pred = [alpha + beta * xv for xv in x_win]
        residuals = [yv - yp for yv, yp in zip(y_win, y_pred, strict=True)]
        ss_res = sum(r**2 for r in residuals)

        # Return based on rettype
        if rettype == 0:  # residual
            if _is_invalid(y_vals[i]) or _is_invalid(x_vals[i]):
                results.append(None)
            else:
                results.append(y_vals[i] - (alpha + beta * x_vals[i]))
        elif rettype == 1:  # beta
            results.append(beta)
        elif rettype == 2:  # alpha
            results.append(alpha)
        elif rettype == 3:  # predicted
            if _is_invalid(x_vals[i]):
                results.append(None)
            else:
                results.append(alpha + beta * x_vals[i])
        elif rettype == 4:  # correlation
            if ss_yy == 0:
                results.append(None)
            else:
                results.append(ss_xy / math.sqrt(ss_xx * ss_yy))
        elif rettype == 5:  # r-squared
            if ss_yy == 0:
                results.append(None)
            else:
                results.append(1 - ss_res / ss_yy)
        elif rettype == 6:  # t-stat beta
            if n <= 2 or ss_res == 0:
                results.append(None)
            else:
                mse = ss_res / (n - 2)
                se_beta = math.sqrt(mse / ss_xx)
                results.append(beta / se_beta if se_beta != 0 else None)
        elif rettype == 7:  # t-stat alpha
            if n <= 2 or ss_res == 0:
                results.append(None)
            else:
                mse = ss_res / (n - 2)
                se_alpha = math.sqrt(mse * (1 / n + x_mean**2 / ss_xx))
                results.append(alpha / se_alpha if se_alpha != 0 else None)
        elif rettype == 8:  # stderr beta
            if n <= 2:
                results.append(None)
            else:
                mse = ss_res / (n - 2)
                results.append(math.sqrt(mse / ss_xx))
        elif rettype == 9:  # stderr alpha
            if n <= 2:
                results.append(None)
            else:
                mse = ss_res / (n - 2)
                results.append(math.sqrt(mse * (1 / n + x_mean**2 / ss_xx)))
        else:
            results.append(None)

    return results


def ts_regression(
    y: pl.DataFrame,
    x: pl.DataFrame,
    d: int,
    lag: int = 0,
    rettype: int | str = 0,
) -> pl.DataFrame:
    """Rolling OLS regression of y on x (online algorithm + numba JIT).

    Uses O(n) online algorithm for rettype 0-5.
    Falls back to original implementation for rettype 6-9 (t-stats, std errors).

    rettype (int or str):
        0 or "resid": residual (y - predicted)
        1 or "beta": beta (slope)
        2 or "alpha": alpha (intercept)
        3 or "predicted": predicted (alpha + beta * x)
        4 or "corr": correlation
        5 or "r_squared": r-squared
        6 or "tstat_beta": t-stat for beta
        7 or "tstat_alpha": t-stat for alpha
        8 or "stderr_beta": std error of beta
        9 or "stderr_alpha": std error of alpha
    """
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor

    # Map string rettype to int
    rettype_map = {
        "resid": 0, "residual": 0,
        "beta": 1, "slope": 1,
        "alpha": 2, "intercept": 2,
        "predicted": 3, "pred": 3,
        "corr": 4, "correlation": 4,
        "r_squared": 5, "rsquared": 5, "r2": 5,
        "tstat_beta": 6,
        "tstat_alpha": 7,
        "stderr_beta": 8,
        "stderr_alpha": 9,
    }
    if isinstance(rettype, str):
        rettype = rettype_map.get(rettype.lower(), 0)

    date_col = y.columns[0]
    value_cols = _get_value_cols(y)

    # Use numba-optimized online algorithm for rettype 0-5 with lag=0
    if rettype <= 5 and lag == 0:
        from alphalab.api.operators._numba_kernels import rolling_regression_online

        def process_col(c: str) -> tuple[str, np.ndarray]:
            y_arr = y[c].to_numpy().astype(np.float64)
            x_arr = x[c].to_numpy().astype(np.float64)
            return (c, rolling_regression_online(y_arr, x_arr, d, rettype))

        with ThreadPoolExecutor(max_workers=min(8, len(value_cols))) as executor:
            col_results = dict(executor.map(process_col, value_cols))

        return pl.DataFrame({date_col: y[date_col], **col_results})

    # Fallback to original implementation for t-stats, std errors, or lag != 0
    def process_col_fallback(c: str) -> tuple[str, list[float | None]]:
        y_vals = y[c].to_list()
        x_vals = x[c].to_list()
        return (c, _compute_regression_col(y_vals, x_vals, d, lag, rettype))

    with ThreadPoolExecutor(max_workers=min(8, len(value_cols))) as executor:
        col_results = dict(executor.map(process_col_fallback, value_cols))

    return pl.DataFrame({date_col: y[date_col], **col_results})
