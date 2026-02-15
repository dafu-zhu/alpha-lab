"""Time-series operators for wide tables.

All operators preserve the wide table structure:
- First column (date) is unchanged
- Operations applied column-wise to symbol columns
"""

import polars as pl


def _get_value_cols(df: pl.DataFrame) -> list[str]:
    """Get value columns (all except first which is date)."""
    return df.columns[1:]


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
    """Rolling product over d periods (partial windows allowed)."""
    date_col = x.columns[0]
    value_cols = _get_value_cols(x)
    return x.select(
        pl.col(date_col),
        *[
            pl.col(c).rolling_map(lambda s: s.product(), window_size=d, min_samples=1).alias(c)
            for c in value_cols
        ],
    )


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

    def arg_max_fn(s: pl.Series) -> float | None:
        if len(s) < d:
            return None
        idx = s.arg_max()
        if idx is None:
            return None
        # Convert to "days since": 0 = today (newest), d-1 = oldest
        return float((d - 1) - idx)

    return x.select(
        pl.col(date_col),
        *[pl.col(c).rolling_map(arg_max_fn, window_size=d).alias(c) for c in value_cols],
    )


def ts_arg_min(x: pl.DataFrame, d: int) -> pl.DataFrame:
    """Days since min in rolling window (0 = today is min, d-1 = oldest day was min)."""
    date_col = x.columns[0]
    value_cols = _get_value_cols(x)

    def arg_min_fn(s: pl.Series) -> float | None:
        if len(s) < d:
            return None
        idx = s.arg_min()
        if idx is None:
            return None
        # Convert to "days since": 0 = today (newest), d-1 = oldest
        return float((d - 1) - idx)

    return x.select(
        pl.col(date_col),
        *[pl.col(c).rolling_map(arg_min_fn, window_size=d).alias(c) for c in value_cols],
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

    def find_last_diff(s: pl.Series) -> float | None:
        if len(s) < 2:
            return None
        current = s[-1]
        for i in range(len(s) - 2, -1, -1):
            if s[i] != current and s[i] is not None:
                return float(s[i])
        return None

    return x.select(
        pl.col(date_col),
        *[pl.col(c).rolling_map(find_last_diff, window_size=d).alias(c) for c in value_cols],
    )


def days_from_last_change(x: pl.DataFrame) -> pl.DataFrame:
    """Days since value changed."""
    date_col = x.columns[0]
    value_cols = _get_value_cols(x)
    result_data: dict[str, pl.Series | list[int]] = {date_col: x[date_col]}

    for c in value_cols:
        col_data = x[c].to_list()
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
        result_data[c] = days

    return pl.DataFrame(result_data)


# Phase 4: Stateful Ops


def hump(x: pl.DataFrame, hump: float = 0.01) -> pl.DataFrame:
    """Limit change magnitude per row based on hump * sum(|all values|)."""
    date_col = x.columns[0]
    value_cols = _get_value_cols(x)
    result_data: dict[str, pl.Series | list[float | None]] = {date_col: x[date_col]}

    # Get all values as lists
    col_lists = {c: x[c].to_list() for c in value_cols}
    n = len(x)

    # Init output with first row
    out_lists: dict[str, list[float | None]] = {c: [] for c in value_cols}

    for i in range(n):
        if i == 0:
            for c in value_cols:
                out_lists[c].append(col_lists[c][0])
        else:
            # Compute limit = hump * sum(|all current values|)
            row_sum = sum(abs(col_lists[c][i] or 0) for c in value_cols)
            limit = hump * row_sum

            for c in value_cols:
                prev = out_lists[c][i - 1]
                curr = col_lists[c][i]
                if prev is None or curr is None:
                    out_lists[c].append(curr)
                else:
                    change = curr - prev
                    if abs(change) > limit:
                        out_lists[c].append(prev + (1 if change > 0 else -1) * limit)
                    else:
                        out_lists[c].append(prev)

    for c in value_cols:
        result_data[c] = out_lists[c]

    return pl.DataFrame(result_data)


def ts_decay_linear(x: pl.DataFrame, d: int, dense: bool = False) -> pl.DataFrame:
    """Weighted average with linear decay weights [1, 2, ..., d]."""
    date_col = x.columns[0]
    value_cols = _get_value_cols(x)
    weights = list(range(1, d + 1))
    weight_sum = sum(weights)

    def weighted_avg(s: pl.Series) -> float | None:
        if len(s) < d:
            return None
        vals = s.to_list()
        if dense:
            # Only use non-null values
            valid: list[tuple[int, float]] = [
                (w, v) for w, v in zip(weights, vals, strict=True) if v is not None
            ]
            if not valid:
                return None
            w_sum = sum(w for w, _ in valid)
            return float(sum(w * v for w, v in valid) / w_sum)
        else:
            if any(v is None for v in vals):
                return None
            return float(sum(w * v for w, v in zip(weights, vals, strict=True)) / weight_sum)

    return x.select(
        pl.col(date_col),
        *[pl.col(c).rolling_map(weighted_avg, window_size=d).alias(c) for c in value_cols],
    )


def ts_rank(x: pl.DataFrame, d: int, constant: float = 0) -> pl.DataFrame:
    """Rank of current value in rolling window, scaled to [constant, 1+constant] (partial windows allowed)."""
    date_col = x.columns[0]
    value_cols = _get_value_cols(x)

    def rank_in_window(s: pl.Series) -> float | None:
        vals = s.to_list()
        current = vals[-1]
        if current is None:
            return None
        sorted_vals = sorted([v for v in vals if v is not None])
        if len(sorted_vals) <= 1:
            return constant + 0.5
        idx = sorted_vals.index(current)
        return constant + idx / (len(sorted_vals) - 1)

    return x.select(
        pl.col(date_col),
        *[pl.col(c).rolling_map(rank_in_window, window_size=d, min_samples=1).alias(c) for c in value_cols],
    )


# Phase 5: Two-Variable Ops


def ts_corr(x: pl.DataFrame, y: pl.DataFrame, d: int) -> pl.DataFrame:
    """Rolling Pearson correlation between matching columns of x and y."""
    date_col = x.columns[0]
    value_cols = _get_value_cols(x)
    result_data: dict[str, pl.Series | list[float | None]] = {date_col: x[date_col]}

    for c in value_cols:
        x_vals = x[c].to_list()
        y_vals = y[c].to_list()
        corrs: list[float | None] = []
        for i in range(len(x_vals)):
            if i < d - 1:
                corrs.append(None)
            else:
                x_win = x_vals[i - d + 1 : i + 1]
                y_win = y_vals[i - d + 1 : i + 1]
                if any(v is None for v in x_win) or any(v is None for v in y_win):
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
        result_data[c] = corrs

    return pl.DataFrame(result_data)


def ts_covariance(x: pl.DataFrame, y: pl.DataFrame, d: int) -> pl.DataFrame:
    """Rolling covariance between matching columns of x and y."""
    date_col = x.columns[0]
    value_cols = _get_value_cols(x)
    result_data: dict[str, pl.Series | list[float | None]] = {date_col: x[date_col]}

    for c in value_cols:
        x_vals = x[c].to_list()
        y_vals = y[c].to_list()
        covs: list[float | None] = []
        for i in range(len(x_vals)):
            if i < d - 1:
                covs.append(None)
            else:
                x_win = x_vals[i - d + 1 : i + 1]
                y_win = y_vals[i - d + 1 : i + 1]
                if any(v is None for v in x_win) or any(v is None for v in y_win):
                    covs.append(None)
                else:
                    x_mean = sum(x_win) / d
                    y_mean = sum(y_win) / d
                    cov = sum((xv - x_mean) * (yv - y_mean) for xv, yv in zip(x_win, y_win, strict=True)) / d
                    covs.append(cov)
        result_data[c] = covs

    return pl.DataFrame(result_data)


def ts_quantile(x: pl.DataFrame, d: int, driver: str = "gaussian") -> pl.DataFrame:
    """Rolling quantile transform: ts_rank + inverse CDF (partial windows allowed)."""
    import math

    date_col = x.columns[0]
    value_cols = _get_value_cols(x)

    def inv_norm(p: float) -> float:
        """Approximate inverse normal CDF."""
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

    def quantile_transform(s: pl.Series) -> float | None:
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
            return inv_norm(rank_pct)
        else:  # uniform
            return rank_pct * 2 - 1

    return x.select(
        pl.col(date_col),
        *[pl.col(c).rolling_map(quantile_transform, window_size=d, min_samples=1).alias(c) for c in value_cols],
    )


# Phase 6: Regression


def ts_regression(
    y: pl.DataFrame,
    x: pl.DataFrame,
    d: int,
    lag: int = 0,
    rettype: int | str = 0,
) -> pl.DataFrame:
    """Rolling OLS regression of y on x (partial windows allowed, min 2 samples).

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
    import math

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
    result_data: dict[str, pl.Series | list[float | None]] = {date_col: y[date_col]}

    for c in value_cols:
        y_vals = y[c].to_list()
        x_vals = x[c].shift(lag).to_list() if lag > 0 else x[c].to_list()
        results: list[float | None] = []

        for i in range(len(y_vals)):
            # Use available window up to d (partial windows allowed)
            start_idx = max(0, i - d + 1)
            y_win_raw = y_vals[start_idx : i + 1]
            x_win_raw = x_vals[start_idx : i + 1]

            # Filter out pairs where either is null
            pairs = [(yv, xv) for yv, xv in zip(y_win_raw, x_win_raw, strict=True) if yv is not None and xv is not None]

            # Need at least 2 points for regression
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

            if rettype == 0:  # residual
                if y_vals[i] is None or x_vals[i] is None:
                    results.append(None)
                else:
                    results.append(y_vals[i] - (alpha + beta * x_vals[i]))
            elif rettype == 1:  # beta
                results.append(beta)
            elif rettype == 2:  # alpha
                results.append(alpha)
            elif rettype == 3:  # predicted
                if x_vals[i] is None:
                    results.append(None)
                else:
                    results.append(alpha + beta * x_vals[i])
            elif rettype == 4:  # correlation
                if ss_xx == 0 or ss_yy == 0:
                    results.append(None)
                else:
                    results.append(ss_xy / math.sqrt(ss_xx * ss_yy))
            elif rettype == 5:  # r-squared
                if ss_yy == 0:
                    results.append(None)
                else:
                    results.append(1 - ss_res / ss_yy)
            elif rettype == 6:  # t-stat for beta
                if n <= 2 or ss_res == 0:
                    results.append(None)
                else:
                    mse = ss_res / (n - 2)
                    se_beta = math.sqrt(mse / ss_xx)
                    results.append(beta / se_beta if se_beta != 0 else None)
            elif rettype == 7:  # t-stat for alpha
                if n <= 2 or ss_res == 0:
                    results.append(None)
                else:
                    mse = ss_res / (n - 2)
                    se_alpha = math.sqrt(mse * (1 / n + x_mean**2 / ss_xx))
                    results.append(alpha / se_alpha if se_alpha != 0 else None)
            elif rettype == 8:  # std error of beta
                if n <= 2:
                    results.append(None)
                else:
                    mse = ss_res / (n - 2)
                    results.append(math.sqrt(mse / ss_xx))
            elif rettype == 9:  # std error of alpha
                if n <= 2:
                    results.append(None)
                else:
                    mse = ss_res / (n - 2)
                    results.append(math.sqrt(mse * (1 / n + x_mean**2 / ss_xx)))
            else:
                results.append(None)

        result_data[c] = results

    return pl.DataFrame(result_data)
