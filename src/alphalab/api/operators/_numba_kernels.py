"""Numba-optimized kernels for operators.

Contains:
- Time-series kernels: Online/incremental algorithms for O(n) complexity
- Group kernels: Row-wise operations across symbols within each date
"""

import numpy as np
from numba import njit, prange


@njit(cache=True)
def rolling_corr_online(x: np.ndarray, y: np.ndarray, d: int) -> np.ndarray:
    """O(n) rolling correlation using online algorithm.

    Uses the formula: corr = cov(x,y) / (std(x) * std(y))
    where cov and std are computed incrementally.
    Tracks NaN count in window to handle NaN propagation correctly.
    """
    n = len(x)
    result = np.empty(n, dtype=np.float64)
    result[:d-1] = np.nan

    if n < d:
        result[:] = np.nan
        return result

    # Initialize running sums for first window
    sx = 0.0   # sum of x
    sy = 0.0   # sum of y
    sxy = 0.0  # sum of x*y
    sx2 = 0.0  # sum of x^2
    sy2 = 0.0  # sum of y^2
    nan_count = 0  # count of NaN values in current window

    # Initialize first window
    for j in range(d):
        if np.isnan(x[j]) or np.isnan(y[j]):
            nan_count += 1
        else:
            sx += x[j]
            sy += y[j]
            sxy += x[j] * y[j]
            sx2 += x[j] * x[j]
            sy2 += y[j] * y[j]

    if nan_count > 0:
        result[d-1] = np.nan
    else:
        # Compute correlation for first window
        cov = (sxy - sx * sy / d) / d
        vx = (sx2 - sx * sx / d) / d
        vy = (sy2 - sy * sy / d) / d
        if vx > 0 and vy > 0:
            result[d-1] = cov / np.sqrt(vx * vy)
        else:
            result[d-1] = np.nan

    # Slide window: O(1) per step
    for i in range(d, n):
        old_x, old_y = x[i-d], y[i-d]
        new_x, new_y = x[i], y[i]
        old_is_nan = np.isnan(old_x) or np.isnan(old_y)
        new_is_nan = np.isnan(new_x) or np.isnan(new_y)

        # Update NaN count
        if old_is_nan:
            nan_count -= 1
        if new_is_nan:
            nan_count += 1

        # Update running sums (only for non-NaN values)
        if not old_is_nan:
            sx -= old_x
            sy -= old_y
            sxy -= old_x * old_y
            sx2 -= old_x * old_x
            sy2 -= old_y * old_y
        if not new_is_nan:
            sx += new_x
            sy += new_y
            sxy += new_x * new_y
            sx2 += new_x * new_x
            sy2 += new_y * new_y

        # Compute result only if no NaN in window
        if nan_count > 0:
            result[i] = np.nan
        else:
            cov = (sxy - sx * sy / d) / d
            vx = (sx2 - sx * sx / d) / d
            vy = (sy2 - sy * sy / d) / d
            if vx > 0 and vy > 0:
                result[i] = cov / np.sqrt(vx * vy)
            else:
                result[i] = np.nan

    return result


@njit(cache=True)
def rolling_cov_online(x: np.ndarray, y: np.ndarray, d: int) -> np.ndarray:
    """O(n) rolling covariance using online algorithm.

    Tracks NaN count in window to handle NaN propagation correctly.
    """
    n = len(x)
    result = np.empty(n, dtype=np.float64)
    result[:d-1] = np.nan

    if n < d:
        result[:] = np.nan
        return result

    # Initialize running sums
    sx = 0.0
    sy = 0.0
    sxy = 0.0
    nan_count = 0

    # Initialize first window
    for j in range(d):
        if np.isnan(x[j]) or np.isnan(y[j]):
            nan_count += 1
        else:
            sx += x[j]
            sy += y[j]
            sxy += x[j] * y[j]

    if nan_count > 0:
        result[d-1] = np.nan
    else:
        result[d-1] = (sxy - sx * sy / d) / d

    # Slide window
    for i in range(d, n):
        old_x, old_y = x[i-d], y[i-d]
        new_x, new_y = x[i], y[i]
        old_is_nan = np.isnan(old_x) or np.isnan(old_y)
        new_is_nan = np.isnan(new_x) or np.isnan(new_y)

        # Update NaN count
        if old_is_nan:
            nan_count -= 1
        if new_is_nan:
            nan_count += 1

        # Update running sums (only for non-NaN values)
        if not old_is_nan:
            sx -= old_x
            sy -= old_y
            sxy -= old_x * old_y
        if not new_is_nan:
            sx += new_x
            sy += new_y
            sxy += new_x * new_y

        # Compute result only if no NaN in window
        if nan_count > 0:
            result[i] = np.nan
        else:
            result[i] = (sxy - sx * sy / d) / d

    return result


@njit(cache=True)
def rolling_regression_online(
    y: np.ndarray, x: np.ndarray, d: int, rettype: int
) -> np.ndarray:
    """O(n) rolling OLS regression using online algorithm.

    rettype: 0=residual, 1=beta, 2=alpha, 3=predicted, 4=corr, 5=r2
    Tracks NaN count in window to handle NaN propagation correctly.
    """
    n = len(y)
    result = np.empty(n, dtype=np.float64)
    result[:d-1] = np.nan

    if n < d:
        result[:] = np.nan
        return result

    # Running sums
    sx = 0.0
    sy = 0.0
    sxy = 0.0
    sx2 = 0.0
    sy2 = 0.0
    nan_count = 0

    # Initialize first window
    for j in range(d):
        if np.isnan(x[j]) or np.isnan(y[j]):
            nan_count += 1
        else:
            sx += x[j]
            sy += y[j]
            sxy += x[j] * y[j]
            sx2 += x[j] * x[j]
            sy2 += y[j] * y[j]

    if nan_count > 0:
        result[d-1] = np.nan
    else:
        result[d-1] = _compute_regression_result(
            sx, sy, sxy, sx2, sy2, d, x[d-1], y[d-1], rettype
        )

    # Slide window
    for i in range(d, n):
        old_x, old_y = x[i-d], y[i-d]
        new_x, new_y = x[i], y[i]
        old_is_nan = np.isnan(old_x) or np.isnan(old_y)
        new_is_nan = np.isnan(new_x) or np.isnan(new_y)

        # Update NaN count
        if old_is_nan:
            nan_count -= 1
        if new_is_nan:
            nan_count += 1

        # Update running sums (only for non-NaN values)
        if not old_is_nan:
            sx -= old_x
            sy -= old_y
            sxy -= old_x * old_y
            sx2 -= old_x * old_x
            sy2 -= old_y * old_y
        if not new_is_nan:
            sx += new_x
            sy += new_y
            sxy += new_x * new_y
            sx2 += new_x * new_x
            sy2 += new_y * new_y

        # Compute result only if no NaN in window
        if nan_count > 0:
            result[i] = np.nan
        else:
            result[i] = _compute_regression_result(
                sx, sy, sxy, sx2, sy2, d, new_x, new_y, rettype
            )

    return result


@njit(cache=True)
def _compute_regression_result(
    sx: float, sy: float, sxy: float, sx2: float, sy2: float,
    d: int, curr_x: float, curr_y: float, rettype: int
) -> float:
    """Compute regression statistic from running sums.

    rettype: 0=residual, 1=beta, 2=alpha, 3=predicted, 4=corr, 5=r2,
             6=tstat_beta, 7=tstat_alpha, 8=stderr_beta, 9=stderr_alpha
    """
    x_mean = sx / d
    y_mean = sy / d

    ss_xx = sx2 - sx * sx / d
    ss_yy = sy2 - sy * sy / d
    ss_xy = sxy - sx * sy / d

    if ss_xx == 0:
        return np.nan

    beta = ss_xy / ss_xx
    alpha = y_mean - beta * x_mean

    if rettype == 0:  # residual
        return curr_y - (alpha + beta * curr_x)
    elif rettype == 1:  # beta
        return beta
    elif rettype == 2:  # alpha
        return alpha
    elif rettype == 3:  # predicted
        return alpha + beta * curr_x
    elif rettype == 4:  # correlation
        if ss_yy == 0:
            return np.nan
        return ss_xy / np.sqrt(ss_xx * ss_yy)
    elif rettype == 5:  # r-squared
        if ss_yy == 0:
            return np.nan
        # r2 = 1 - SS_res / SS_tot = corr^2
        corr = ss_xy / np.sqrt(ss_xx * ss_yy)
        return corr * corr
    elif rettype >= 6 and rettype <= 9:  # t-stats and std errors
        # SS_res = SS_yy - SS_xy^2 / SS_xx (OLS identity)
        ss_res = ss_yy - ss_xy * ss_xy / ss_xx
        if d <= 2 or ss_res <= 0:
            return np.nan
        mse = ss_res / (d - 2)

        if rettype == 6:  # t-stat beta
            se_beta = np.sqrt(mse / ss_xx)
            if se_beta == 0:
                return np.nan
            return beta / se_beta
        elif rettype == 7:  # t-stat alpha
            se_alpha = np.sqrt(mse * (1.0 / d + x_mean * x_mean / ss_xx))
            if se_alpha == 0:
                return np.nan
            return alpha / se_alpha
        elif rettype == 8:  # stderr beta
            return np.sqrt(mse / ss_xx)
        else:  # rettype == 9, stderr alpha
            return np.sqrt(mse * (1.0 / d + x_mean * x_mean / ss_xx))
    else:
        return np.nan


@njit(cache=True)
def rolling_regression_online_with_lag(
    y: np.ndarray, x: np.ndarray, d: int, lag: int, rettype: int
) -> np.ndarray:
    """O(n) rolling OLS regression with lag support.

    rettype: 0=residual, 1=beta, 2=alpha, 3=predicted, 4=corr, 5=r2,
             6=tstat_beta, 7=tstat_alpha, 8=stderr_beta, 9=stderr_alpha
    lag: shift x backward by lag positions (x[i] uses x[i-lag])
    """
    n = len(y)
    result = np.empty(n, dtype=np.float64)

    # Need at least d + lag observations
    min_idx = d - 1 + lag
    result[:min_idx] = np.nan

    if n <= min_idx:
        result[:] = np.nan
        return result

    # Running sums
    sx = 0.0
    sy = 0.0
    sxy = 0.0
    sx2 = 0.0
    sy2 = 0.0
    nan_count = 0

    # Initialize first window (indices lag to lag+d-1 for x, 0 to d-1 for y alignment)
    # But we compute at index d-1+lag, so window covers y[lag:lag+d] and x[0:d]
    for j in range(d):
        y_idx = lag + j
        x_idx = j
        if y_idx >= n or np.isnan(x[x_idx]) or np.isnan(y[y_idx]):
            nan_count += 1
        else:
            sx += x[x_idx]
            sy += y[y_idx]
            sxy += x[x_idx] * y[y_idx]
            sx2 += x[x_idx] * x[x_idx]
            sy2 += y[y_idx] * y[y_idx]

    if nan_count > 0:
        result[min_idx] = np.nan
    else:
        # curr_x and curr_y for residual/predicted
        curr_x = x[d - 1]
        curr_y = y[min_idx]
        result[min_idx] = _compute_regression_result(
            sx, sy, sxy, sx2, sy2, d, curr_x, curr_y, rettype
        )

    # Slide window
    for i in range(min_idx + 1, n):
        # Window for y: [i - d + 1, i], window for x: [i - d + 1 - lag, i - lag]
        old_y_idx = i - d
        new_y_idx = i
        old_x_idx = i - d - lag
        new_x_idx = i - lag

        old_y = y[old_y_idx] if old_y_idx >= 0 else np.nan
        new_y = y[new_y_idx]
        old_x = x[old_x_idx] if old_x_idx >= 0 else np.nan
        new_x = x[new_x_idx] if new_x_idx >= 0 else np.nan

        old_is_nan = np.isnan(old_x) or np.isnan(old_y)
        new_is_nan = np.isnan(new_x) or np.isnan(new_y)

        # Update NaN count
        if old_is_nan:
            nan_count -= 1
        if new_is_nan:
            nan_count += 1

        # Update running sums
        if not old_is_nan:
            sx -= old_x
            sy -= old_y
            sxy -= old_x * old_y
            sx2 -= old_x * old_x
            sy2 -= old_y * old_y
        if not new_is_nan:
            sx += new_x
            sy += new_y
            sxy += new_x * new_y
            sx2 += new_x * new_x
            sy2 += new_y * new_y

        if nan_count > 0:
            result[i] = np.nan
        else:
            result[i] = _compute_regression_result(
                sx, sy, sxy, sx2, sy2, d, new_x, new_y, rettype
            )

    return result


@njit(cache=True)
def rolling_mean_online(x: np.ndarray, d: int) -> np.ndarray:
    """O(n) rolling mean."""
    n = len(x)
    result = np.empty(n, dtype=np.float64)
    result[:d-1] = np.nan

    if n < d:
        result[:] = np.nan
        return result

    window_sum = 0.0
    for j in range(d):
        if np.isnan(x[j]):
            window_sum = np.nan
            break
        window_sum += x[j]

    result[d-1] = window_sum / d if not np.isnan(window_sum) else np.nan

    for i in range(d, n):
        if np.isnan(x[i]) or np.isnan(x[i-d]):
            result[i] = np.nan
            window_sum = np.nan
        elif np.isnan(window_sum):
            result[i] = np.nan
        else:
            window_sum += x[i] - x[i-d]
            result[i] = window_sum / d

    return result


@njit(cache=True)
def rolling_std_online(x: np.ndarray, d: int) -> np.ndarray:
    """O(n) rolling standard deviation using Welford's algorithm."""
    n = len(x)
    result = np.empty(n, dtype=np.float64)
    result[:d-1] = np.nan

    if n < d:
        result[:] = np.nan
        return result

    # Initialize
    sx = 0.0
    sx2 = 0.0

    has_nan = False
    for j in range(d):
        if np.isnan(x[j]):
            has_nan = True
            break
        sx += x[j]
        sx2 += x[j] * x[j]

    if has_nan:
        result[d-1] = np.nan
    else:
        var = (sx2 - sx * sx / d) / d
        result[d-1] = np.sqrt(var) if var > 0 else 0.0

    for i in range(d, n):
        old_x = x[i-d]
        new_x = x[i]

        if np.isnan(new_x) or np.isnan(old_x):
            result[i] = np.nan
            continue

        sx += new_x - old_x
        sx2 += new_x * new_x - old_x * old_x

        var = (sx2 - sx * sx / d) / d
        result[i] = np.sqrt(var) if var > 0 else 0.0

    return result


@njit(cache=True)
def rolling_decay_linear_online(x: np.ndarray, d: int) -> np.ndarray:
    """O(n) rolling weighted average with linear decay weights [1, 2, ..., d].

    Online update: wsum_new = wsum_old - sx_old + d * new_x
    Tracks NaN count in window to handle NaN propagation correctly.
    """
    n = len(x)
    result = np.empty(n, dtype=np.float64)
    result[:d-1] = np.nan

    if n < d:
        result[:] = np.nan
        return result

    weight_sum = d * (d + 1) / 2

    # Initialize running sums for first window
    wsum = 0.0  # weighted sum: sum(w[j] * x[j])
    sx = 0.0    # simple sum: sum(x[j])
    nan_count = 0

    # Initialize first window (weight j+1 for position j)
    for j in range(d):
        if np.isnan(x[j]):
            nan_count += 1
        else:
            wsum += (j + 1) * x[j]
            sx += x[j]

    if nan_count > 0:
        result[d-1] = np.nan
    else:
        result[d-1] = wsum / weight_sum

    # Slide window: O(1) per step
    for i in range(d, n):
        old_x = x[i - d]
        new_x = x[i]
        old_is_nan = np.isnan(old_x)
        new_is_nan = np.isnan(new_x)

        # Update NaN count
        if old_is_nan:
            nan_count -= 1
        if new_is_nan:
            nan_count += 1

        # Update wsum (must use sx BEFORE updating it)
        if not new_is_nan:
            wsum = wsum - sx + d * new_x
        else:
            wsum = wsum - sx

        # Update sx
        if not old_is_nan:
            sx -= old_x
        if not new_is_nan:
            sx += new_x

        # Compute result only if no NaN in window
        if nan_count > 0:
            result[i] = np.nan
        else:
            result[i] = wsum / weight_sum

    return result


@njit(cache=True)
def rolling_product_online(x: np.ndarray, d: int) -> np.ndarray:
    """O(n) rolling product using log transform.

    Uses product = exp(sum(log(|x|))) with sign tracking.
    Handles zeros (product=0), NaNs (propagated), and negative values.
    Partial windows allowed (min_samples=1).
    """
    n = len(x)
    result = np.empty(n, dtype=np.float64)

    if n == 0:
        return result

    log_sum = 0.0
    neg_count = 0
    zero_count = 0
    nan_count = 0
    valid_count = 0

    # Build up partial windows
    for i in range(min(d, n)):
        val = x[i]

        if np.isnan(val):
            nan_count += 1
        elif val == 0.0:
            zero_count += 1
            valid_count += 1
        else:
            if val < 0:
                neg_count += 1
            log_sum += np.log(np.abs(val))
            valid_count += 1

        if nan_count > 0:
            result[i] = np.nan
        elif zero_count > 0:
            result[i] = 0.0
        elif valid_count == 0:
            result[i] = np.nan
        else:
            sign = -1.0 if (neg_count % 2 == 1) else 1.0
            result[i] = sign * np.exp(log_sum)

    # Slide window
    for i in range(d, n):
        old_val = x[i - d]
        if np.isnan(old_val):
            nan_count -= 1
        elif old_val == 0.0:
            zero_count -= 1
            valid_count -= 1
        else:
            if old_val < 0:
                neg_count -= 1
            log_sum -= np.log(np.abs(old_val))
            valid_count -= 1

        new_val = x[i]
        if np.isnan(new_val):
            nan_count += 1
        elif new_val == 0.0:
            zero_count += 1
            valid_count += 1
        else:
            if new_val < 0:
                neg_count += 1
            log_sum += np.log(np.abs(new_val))
            valid_count += 1

        if nan_count > 0:
            result[i] = np.nan
        elif zero_count > 0:
            result[i] = 0.0
        elif valid_count == 0:
            result[i] = np.nan
        else:
            sign = -1.0 if (neg_count % 2 == 1) else 1.0
            result[i] = sign * np.exp(log_sum)

    return result


# Batch processing for multiple columns
def process_columns_corr(
    x_arrays: list[np.ndarray], y_arrays: list[np.ndarray], d: int
) -> list[np.ndarray]:
    """Process multiple column pairs for correlation."""
    return [rolling_corr_online(x, y, d) for x, y in zip(x_arrays, y_arrays)]


def process_columns_cov(
    x_arrays: list[np.ndarray], y_arrays: list[np.ndarray], d: int
) -> list[np.ndarray]:
    """Process multiple column pairs for covariance."""
    return [rolling_cov_online(x, y, d) for x, y in zip(x_arrays, y_arrays)]


def process_columns_regression(
    y_arrays: list[np.ndarray], x_arrays: list[np.ndarray], d: int, rettype: int
) -> list[np.ndarray]:
    """Process multiple column pairs for regression."""
    return [
        rolling_regression_online(y, x, d, rettype)
        for y, x in zip(y_arrays, x_arrays)
    ]


@njit(cache=True)
def norm_inv_cdf(p: float) -> float:
    """Beasley-Springer-Moro approximation for inverse normal CDF.

    Returns NaN for p <= 0 or p >= 1.
    """
    if p <= 0.0 or p >= 1.0:
        return np.nan

    # Rational approximation coefficients (central region)
    a0, a1, a2, a3, a4, a5 = (
        -3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
        1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00,
    )
    b0, b1, b2, b3, b4 = (
        -5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
        6.680131188771972e01, -1.328068155288572e01,
    )
    # Tail region coefficients
    c0, c1, c2, c3, c4, c5 = (
        -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
        -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00,
    )
    d0, d1, d2, d3 = (
        7.784695709041462e-03, 3.224671290700398e-01,
        2.445134137142996e00, 3.754408661907416e00,
    )

    p_low = 0.02425
    p_high = 1.0 - p_low

    if p < p_low:
        # Lower tail
        q = np.sqrt(-2.0 * np.log(p))
        numer = ((((c0 * q + c1) * q + c2) * q + c3) * q + c4) * q + c5
        denom = (((d0 * q + d1) * q + d2) * q + d3) * q + 1.0
        return numer / denom
    elif p <= p_high:
        # Central region
        q = p - 0.5
        r = q * q
        numer = (((((a0 * r + a1) * r + a2) * r + a3) * r + a4) * r + a5) * q
        denom = ((((b0 * r + b1) * r + b2) * r + b3) * r + b4) * r + 1.0
        return numer / denom
    else:
        # Upper tail (symmetric)
        q = np.sqrt(-2.0 * np.log(1.0 - p))
        numer = ((((c0 * q + c1) * q + c2) * q + c3) * q + c4) * q + c5
        denom = (((d0 * q + d1) * q + d2) * q + d3) * q + 1.0
        return -numer / denom


@njit(cache=True)
def rolling_quantile_gaussian(x: np.ndarray, d: int) -> np.ndarray:
    """Rolling quantile transform with Gaussian inverse CDF.

    Computes rolling rank then applies inverse normal CDF.
    Partial windows allowed. NaN current value propagates NaN.
    """
    n = len(x)
    result = np.empty(n, dtype=np.float64)

    if n == 0:
        return result

    for i in range(n):
        current = x[i]

        # If current value is NaN, result is NaN
        if np.isnan(current):
            result[i] = np.nan
            continue

        # Define window bounds: [max(0, i-d+1), i]
        start = max(0, i - d + 1)

        # Count valid values and values less than current
        valid_count = 0
        count_less = 0

        for j in range(start, i + 1):
            val = x[j]
            if not np.isnan(val):
                valid_count += 1
                if val < current:
                    count_less += 1

        # Compute quantile transform
        if valid_count == 0:
            result[i] = np.nan
        elif valid_count == 1:
            # Single value -> rank_pct = 0.5 -> inv_norm(0.5) = 0.0
            result[i] = 0.0
        else:
            # rank_pct = (count_less + 0.5) / valid_count
            # This matches the original: sorted_vals.index(current) + 0.5 / len(sorted_vals)
            rank_pct = (count_less + 0.5) / valid_count
            result[i] = norm_inv_cdf(rank_pct)

    return result


@njit(cache=True)
def rolling_quantile_uniform(x: np.ndarray, d: int) -> np.ndarray:
    """Rolling quantile transform with uniform distribution (scaled to [-1, 1]).

    Computes rolling rank then scales to [-1, 1] range.
    Partial windows allowed. NaN current value propagates NaN.
    """
    n = len(x)
    result = np.empty(n, dtype=np.float64)

    if n == 0:
        return result

    for i in range(n):
        current = x[i]

        # If current value is NaN, result is NaN
        if np.isnan(current):
            result[i] = np.nan
            continue

        # Define window bounds: [max(0, i-d+1), i]
        start = max(0, i - d + 1)

        # Count valid values and values less than current
        valid_count = 0
        count_less = 0

        for j in range(start, i + 1):
            val = x[j]
            if not np.isnan(val):
                valid_count += 1
                if val < current:
                    count_less += 1

        # Compute quantile transform
        if valid_count == 0:
            result[i] = np.nan
        elif valid_count == 1:
            # Single value -> rank_pct = 0.5 -> uniform: 0.5 * 2 - 1 = 0.0
            result[i] = 0.0
        else:
            # rank_pct = (count_less + 0.5) / valid_count
            rank_pct = (count_less + 0.5) / valid_count
            # Scale to [-1, 1]: rank_pct * 2 - 1
            result[i] = rank_pct * 2.0 - 1.0

    return result


@njit(cache=True)
def rolling_arg_max(x: np.ndarray, d: int) -> np.ndarray:
    """O(n*d) rolling argmax - days since max in window.

    For each position i, scan window [i-d+1, i] and find index of max value.
    Returns i - max_idx (0 = current is max, d-1 = oldest was max).
    If current is NaN, returns NaN. NaN values in window are skipped.
    First d-1 values are NaN (window not complete).
    """
    n = len(x)
    result = np.empty(n, dtype=np.float64)
    result[:d-1] = np.nan

    if n < d:
        result[:] = np.nan
        return result

    for i in range(d - 1, n):
        current = x[i]

        # If current value is NaN, result is NaN
        if np.isnan(current):
            result[i] = np.nan
            continue

        # Define window bounds: [i-d+1, i]
        start = i - d + 1

        # Find max value and its index (closest to current if ties)
        max_val = -np.inf
        max_idx = -1

        for j in range(start, i + 1):
            val = x[j]
            if not np.isnan(val):
                # Use >= to prefer more recent (higher j) in case of ties
                if val >= max_val:
                    max_val = val
                    max_idx = j

        # If no valid values found, return NaN
        if max_idx == -1:
            result[i] = np.nan
        else:
            # Days since max: i - max_idx
            # 0 = current is max, d-1 = oldest was max
            result[i] = float(i - max_idx)

    return result


@njit(cache=True)
def rolling_arg_min(x: np.ndarray, d: int) -> np.ndarray:
    """O(n*d) rolling argmin - days since min in window.

    For each position i, scan window [i-d+1, i] and find index of min value.
    Returns i - min_idx (0 = current is min, d-1 = oldest was min).
    If current is NaN, returns NaN. NaN values in window are skipped.
    First d-1 values are NaN (window not complete).
    """
    n = len(x)
    result = np.empty(n, dtype=np.float64)
    result[:d-1] = np.nan

    if n < d:
        result[:] = np.nan
        return result

    for i in range(d - 1, n):
        current = x[i]

        # If current value is NaN, result is NaN
        if np.isnan(current):
            result[i] = np.nan
            continue

        # Define window bounds: [i-d+1, i]
        start = i - d + 1

        # Find min value and its index (closest to current if ties)
        min_val = np.inf
        min_idx = -1

        for j in range(start, i + 1):
            val = x[j]
            if not np.isnan(val):
                # Use <= to prefer more recent (higher j) in case of ties
                if val <= min_val:
                    min_val = val
                    min_idx = j

        # If no valid values found, return NaN
        if min_idx == -1:
            result[i] = np.nan
        else:
            # Days since min: i - min_idx
            # 0 = current is min, d-1 = oldest was min
            result[i] = float(i - min_idx)

    return result


@njit(cache=True)
def rolling_last_diff(x: np.ndarray, d: int) -> np.ndarray:
    """O(n*d) find last different value in rolling window.

    For each position i, scan backwards from i-1 to find first value != x[i].
    Window bounds: [max(0, i-d+1), i-1] (excludes current position).

    Args:
        x: Input array of float64 values
        d: Window size (looks back up to d-1 positions)

    Returns:
        Array where each element is the last different value within the window,
        or NaN if current is NaN or no different value found.
    """
    n = len(x)
    result = np.empty(n, dtype=np.float64)

    if n == 0:
        return result

    for i in range(n):
        current = x[i]

        # If current value is NaN, result is NaN
        if np.isnan(current):
            result[i] = np.nan
            continue

        # Define window bounds: [max(0, i-d+1), i-1]
        # We look back at most d-1 positions (since current is at position i)
        start = max(0, i - d + 1)

        # Scan backwards from i-1 to find first different value
        found = False
        for j in range(i - 1, start - 1, -1):
            val = x[j]
            if not np.isnan(val) and val != current:
                result[i] = val
                found = True
                break

        if not found:
            result[i] = np.nan

    return result


@njit(cache=True)
def rolling_rank(x: np.ndarray, d: int, constant: float) -> np.ndarray:
    """O(n*d) rolling rank scaled to [constant, 1+constant].

    Counts values less than current within the window.
    Partial windows allowed (min_samples=1). NaN current value propagates NaN.
    """
    n = len(x)
    result = np.empty(n, dtype=np.float64)

    if n == 0:
        return result

    for i in range(n):
        current = x[i]

        # If current value is NaN, result is NaN
        if np.isnan(current):
            result[i] = np.nan
            continue

        # Define window bounds: [max(0, i-d+1), i]
        start = max(0, i - d + 1)

        # Count valid values and values less than current
        valid_count = 0
        count_less = 0

        for j in range(start, i + 1):
            val = x[j]
            if not np.isnan(val):
                valid_count += 1
                if val < current:
                    count_less += 1

        # Compute rank
        if valid_count == 0:
            result[i] = np.nan
        elif valid_count == 1:
            result[i] = constant + 0.5
        else:
            result[i] = constant + count_less / (valid_count - 1)

    return result


@njit(cache=True)
def hump_column(col_data: np.ndarray, row_limits: np.ndarray) -> np.ndarray:
    """Apply hump limiting to a single column.

    For each row, limits the change from previous output to row_limits[i].
    Handles NaN values by passing through without limiting.

    Args:
        col_data: Column values
        row_limits: Pre-computed limit for each row (hump_factor * row_abs_sum)

    Returns:
        Hump-limited column values
    """
    n = len(col_data)
    result = np.empty(n, dtype=np.float64)

    for i in range(n):
        if i == 0:
            result[i] = col_data[i]
        else:
            prev = result[i - 1]
            curr = col_data[i]
            limit = row_limits[i]

            if np.isnan(prev) or np.isnan(curr):
                result[i] = curr
            else:
                change = curr - prev
                if abs(change) > limit:
                    if change > 0:
                        result[i] = prev + limit
                    else:
                        result[i] = prev - limit
                else:
                    result[i] = curr

    return result


@njit(cache=True)
def _values_are_same(
    curr_val: float, prev_val: float, curr_is_null: bool, prev_is_null: bool
) -> bool:
    """Determine if two values are considered "same" for change detection.

    Rules:
    - null == null: same (Polars semantics)
    - null vs non-null: different
    - NaN vs anything: different (NaN != NaN)
    - Otherwise: compare values
    """
    # Handle null cases first
    if curr_is_null and prev_is_null:
        return True
    if curr_is_null or prev_is_null:
        return False

    # Handle NaN cases (NaN that isn't Polars null)
    if np.isnan(curr_val) or np.isnan(prev_val):
        return False

    # Both are valid numeric values
    return curr_val == prev_val


@njit(cache=True)
def days_since_change_with_null(x: np.ndarray, is_null: np.ndarray) -> np.ndarray:
    """Compute days since value last changed.

    Handles Polars null vs NaN distinction:
    - Polars null (is_null=True): null == null is "same value"
    - Polars NaN (is_null=False but isnan=True): NaN != NaN is "different"

    Args:
        x: Input array of values (NaN for both null and NaN)
        is_null: Boolean mask where True means Polars null (not NaN)

    Returns:
        Array of integers representing days since last change
    """
    n = len(x)
    result = np.zeros(n, dtype=np.int64)

    last_change_idx = 0
    for i in range(1, n):
        same = _values_are_same(x[i], x[i - 1], is_null[i], is_null[i - 1])
        if same:
            result[i] = i - last_change_idx
        else:
            last_change_idx = i

    return result


# =============================================================================
# Group Operations - Row-wise with group assignments
# =============================================================================


@njit(cache=True)
def group_neutralize_row(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Subtract group mean from each value in a single row.

    Args:
        values: Row values (1D array)
        groups: Group assignments (1D array of integers)

    Returns:
        Neutralized values
    """
    n = len(values)
    result = np.empty(n, dtype=np.float64)

    # Find unique groups and compute means
    max_group = -1
    for i in range(n):
        if not np.isnan(groups[i]) and groups[i] > max_group:
            max_group = int(groups[i])

    if max_group < 0:
        result[:] = np.nan
        return result

    # Allocate arrays for group sums and counts
    group_sums = np.zeros(max_group + 1, dtype=np.float64)
    group_counts = np.zeros(max_group + 1, dtype=np.int64)

    # First pass: compute group sums and counts
    for i in range(n):
        if not np.isnan(values[i]) and not np.isnan(groups[i]):
            g = int(groups[i])
            group_sums[g] += values[i]
            group_counts[g] += 1

    # Second pass: subtract group mean
    for i in range(n):
        if np.isnan(values[i]) or np.isnan(groups[i]):
            result[i] = np.nan
        else:
            g = int(groups[i])
            if group_counts[g] > 0:
                result[i] = values[i] - group_sums[g] / group_counts[g]
            else:
                result[i] = np.nan

    return result


@njit(cache=True)
def group_zscore_row(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Z-score within groups for a single row.

    Args:
        values: Row values (1D array)
        groups: Group assignments (1D array of integers)

    Returns:
        Z-scored values
    """
    n = len(values)
    result = np.empty(n, dtype=np.float64)

    max_group = -1
    for i in range(n):
        if not np.isnan(groups[i]) and groups[i] > max_group:
            max_group = int(groups[i])

    if max_group < 0:
        result[:] = np.nan
        return result

    # First pass: compute group sums and counts
    group_sums = np.zeros(max_group + 1, dtype=np.float64)
    group_sq_sums = np.zeros(max_group + 1, dtype=np.float64)
    group_counts = np.zeros(max_group + 1, dtype=np.int64)

    for i in range(n):
        if not np.isnan(values[i]) and not np.isnan(groups[i]):
            g = int(groups[i])
            group_sums[g] += values[i]
            group_sq_sums[g] += values[i] * values[i]
            group_counts[g] += 1

    # Compute group means and stds
    group_means = np.zeros(max_group + 1, dtype=np.float64)
    group_stds = np.zeros(max_group + 1, dtype=np.float64)

    for g in range(max_group + 1):
        if group_counts[g] > 0:
            group_means[g] = group_sums[g] / group_counts[g]
            variance = group_sq_sums[g] / group_counts[g] - group_means[g] ** 2
            if variance > 0:
                group_stds[g] = np.sqrt(variance)

    # Second pass: compute z-scores
    for i in range(n):
        if np.isnan(values[i]) or np.isnan(groups[i]):
            result[i] = np.nan
        else:
            g = int(groups[i])
            if group_counts[g] > 0 and group_stds[g] > 0:
                result[i] = (values[i] - group_means[g]) / group_stds[g]
            else:
                result[i] = np.nan

    return result


@njit(cache=True)
def group_scale_row(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Min-max scale within groups for a single row.

    Args:
        values: Row values (1D array)
        groups: Group assignments (1D array of integers)

    Returns:
        Scaled values in [0, 1]
    """
    n = len(values)
    result = np.empty(n, dtype=np.float64)

    max_group = -1
    for i in range(n):
        if not np.isnan(groups[i]) and groups[i] > max_group:
            max_group = int(groups[i])

    if max_group < 0:
        result[:] = np.nan
        return result

    # First pass: find group min/max
    group_mins = np.full(max_group + 1, np.inf, dtype=np.float64)
    group_maxs = np.full(max_group + 1, -np.inf, dtype=np.float64)

    for i in range(n):
        if not np.isnan(values[i]) and not np.isnan(groups[i]):
            g = int(groups[i])
            if values[i] < group_mins[g]:
                group_mins[g] = values[i]
            if values[i] > group_maxs[g]:
                group_maxs[g] = values[i]

    # Second pass: scale values
    for i in range(n):
        if np.isnan(values[i]) or np.isnan(groups[i]):
            result[i] = np.nan
        else:
            g = int(groups[i])
            range_val = group_maxs[g] - group_mins[g]
            if range_val > 0:
                result[i] = (values[i] - group_mins[g]) / range_val
            else:
                result[i] = np.nan  # All same value in group (0/0)

    return result


@njit(cache=True)
def group_rank_row(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Rank within groups for a single row, normalized to [0, 1].

    Args:
        values: Row values (1D array)
        groups: Group assignments (1D array of integers)

    Returns:
        Rank values in [0, 1], 0.5 for single-member groups
    """
    n = len(values)
    result = np.empty(n, dtype=np.float64)

    max_group = -1
    for i in range(n):
        if not np.isnan(groups[i]) and groups[i] > max_group:
            max_group = int(groups[i])

    if max_group < 0:
        result[:] = np.nan
        return result

    # Count valid values per group
    group_counts = np.zeros(max_group + 1, dtype=np.int64)
    for i in range(n):
        if not np.isnan(values[i]) and not np.isnan(groups[i]):
            group_counts[int(groups[i])] += 1

    # Compute rank for each element
    for i in range(n):
        if np.isnan(values[i]) or np.isnan(groups[i]):
            result[i] = np.nan
        else:
            g = int(groups[i])
            count = group_counts[g]
            if count <= 1:
                result[i] = 0.5  # Single member group
            else:
                # Count elements in same group that are less than current
                rank = 0
                for j in range(n):
                    if j != i and not np.isnan(values[j]) and not np.isnan(groups[j]):
                        if int(groups[j]) == g and values[j] < values[i]:
                            rank += 1
                result[i] = rank / (count - 1)

    return result


@njit(cache=True)
def trade_when_column(
    trade_arr: np.ndarray,
    alpha_arr: np.ndarray,
    exit_arr: np.ndarray,
) -> np.ndarray:
    """Apply trade_when logic to a single column.

    Args:
        trade_arr: Trade entry signals (>0 = enter)
        alpha_arr: Alpha values to use on entry
        exit_arr: Exit signals (>0 = exit)

    Returns:
        Conditional alpha values with carry-forward
    """
    n = len(trade_arr)
    result = np.full(n, np.nan, dtype=np.float64)
    prev = np.nan

    for i in range(n):
        if exit_arr[i] > 0:
            prev = np.nan
            result[i] = np.nan
        elif trade_arr[i] > 0:
            prev = alpha_arr[i]
            result[i] = prev
        else:
            result[i] = prev

    return result


# =============================================================================
# Group operator kernels (row-wise across symbols)
# =============================================================================


@njit(cache=True)
def group_neutralize_rows(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Subtract group mean from each value, row by row."""
    n_rows, n_cols = values.shape
    result = np.empty_like(values)

    for i in range(n_rows):
        max_gid = -1
        for j in range(n_cols):
            if groups[i, j] > max_gid:
                max_gid = groups[i, j]

        if max_gid < 0:
            for j in range(n_cols):
                result[i, j] = np.nan
            continue

        n_groups = max_gid + 1
        group_sum = np.zeros(n_groups, dtype=np.float64)
        group_count = np.zeros(n_groups, dtype=np.int32)

        for j in range(n_cols):
            gid = groups[i, j]
            val = values[i, j]
            if gid >= 0 and not np.isnan(val):
                group_sum[gid] += val
                group_count[gid] += 1

        group_mean = np.empty(n_groups, dtype=np.float64)
        for g in range(n_groups):
            if group_count[g] > 0:
                group_mean[g] = group_sum[g] / group_count[g]
            else:
                group_mean[g] = np.nan

        for j in range(n_cols):
            gid = groups[i, j]
            val = values[i, j]
            if gid >= 0 and not np.isnan(val):
                result[i, j] = val - group_mean[gid]
            else:
                result[i, j] = np.nan

    return result


@njit(cache=True)
def group_zscore_rows(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Z-score within groups, row by row."""
    n_rows, n_cols = values.shape
    result = np.empty_like(values)

    for i in range(n_rows):
        max_gid = -1
        for j in range(n_cols):
            if groups[i, j] > max_gid:
                max_gid = groups[i, j]

        if max_gid < 0:
            for j in range(n_cols):
                result[i, j] = np.nan
            continue

        n_groups = max_gid + 1
        group_sum = np.zeros(n_groups, dtype=np.float64)
        group_sum_sq = np.zeros(n_groups, dtype=np.float64)
        group_count = np.zeros(n_groups, dtype=np.int32)

        for j in range(n_cols):
            gid = groups[i, j]
            val = values[i, j]
            if gid >= 0 and not np.isnan(val):
                group_sum[gid] += val
                group_sum_sq[gid] += val * val
                group_count[gid] += 1

        group_mean = np.empty(n_groups, dtype=np.float64)
        group_std = np.empty(n_groups, dtype=np.float64)
        for g in range(n_groups):
            if group_count[g] > 0:
                mean = group_sum[g] / group_count[g]
                group_mean[g] = mean
                variance = group_sum_sq[g] / group_count[g] - mean * mean
                group_std[g] = np.sqrt(max(0.0, variance))
            else:
                group_mean[g] = np.nan
                group_std[g] = np.nan

        for j in range(n_cols):
            gid = groups[i, j]
            val = values[i, j]
            if gid >= 0 and not np.isnan(val):
                std = group_std[gid]
                if std > 0:
                    result[i, j] = (val - group_mean[gid]) / std
                else:
                    result[i, j] = np.nan
            else:
                result[i, j] = np.nan

    return result


@njit(cache=True)
def group_scale_rows(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Min-max scale within groups to [0, 1], row by row."""
    n_rows, n_cols = values.shape
    result = np.empty_like(values)

    for i in range(n_rows):
        max_gid = -1
        for j in range(n_cols):
            if groups[i, j] > max_gid:
                max_gid = groups[i, j]

        if max_gid < 0:
            for j in range(n_cols):
                result[i, j] = np.nan
            continue

        n_groups = max_gid + 1
        group_min = np.full(n_groups, np.inf, dtype=np.float64)
        group_max = np.full(n_groups, -np.inf, dtype=np.float64)

        for j in range(n_cols):
            gid = groups[i, j]
            val = values[i, j]
            if gid >= 0 and not np.isnan(val):
                if val < group_min[gid]:
                    group_min[gid] = val
                if val > group_max[gid]:
                    group_max[gid] = val

        for j in range(n_cols):
            gid = groups[i, j]
            val = values[i, j]
            if gid >= 0 and not np.isnan(val):
                rng = group_max[gid] - group_min[gid]
                if rng > 0:
                    result[i, j] = (val - group_min[gid]) / rng
                else:
                    result[i, j] = np.nan
            else:
                result[i, j] = np.nan

    return result


@njit(cache=True)
def group_rank_rows(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Rank within groups normalized to [0, 1], row by row."""
    n_rows, n_cols = values.shape
    result = np.empty_like(values)

    for i in range(n_rows):
        max_gid = -1
        for j in range(n_cols):
            if groups[i, j] > max_gid:
                max_gid = groups[i, j]

        if max_gid < 0:
            for j in range(n_cols):
                result[i, j] = np.nan
            continue

        n_groups = max_gid + 1
        group_count = np.zeros(n_groups, dtype=np.int32)
        for j in range(n_cols):
            gid = groups[i, j]
            val = values[i, j]
            if gid >= 0 and not np.isnan(val):
                group_count[gid] += 1

        for j in range(n_cols):
            gid = groups[i, j]
            val = values[i, j]
            if gid < 0 or np.isnan(val):
                result[i, j] = np.nan
                continue

            count = group_count[gid]
            if count == 1:
                result[i, j] = 0.5
                continue

            rank = 0
            for k in range(n_cols):
                if groups[i, k] == gid and not np.isnan(values[i, k]):
                    if values[i, k] < val:
                        rank += 1

            result[i, j] = rank / (count - 1)

    return result


@njit(cache=True)
def group_mean_rows(
    values: np.ndarray,
    weights: np.ndarray,
    groups: np.ndarray,
) -> np.ndarray:
    """Weighted mean within groups, broadcast to all members, row by row."""
    n_rows, n_cols = values.shape
    result = np.empty_like(values)

    for i in range(n_rows):
        max_gid = -1
        for j in range(n_cols):
            if groups[i, j] > max_gid:
                max_gid = groups[i, j]

        if max_gid < 0:
            for j in range(n_cols):
                result[i, j] = np.nan
            continue

        n_groups = max_gid + 1
        weighted_sum = np.zeros(n_groups, dtype=np.float64)
        weight_sum = np.zeros(n_groups, dtype=np.float64)

        for j in range(n_cols):
            gid = groups[i, j]
            val = values[i, j]
            w = weights[i, j]
            if gid >= 0 and not np.isnan(val) and not np.isnan(w):
                weighted_sum[gid] += val * w
                weight_sum[gid] += w

        group_mean = np.empty(n_groups, dtype=np.float64)
        for g in range(n_groups):
            if weight_sum[g] > 0:
                group_mean[g] = weighted_sum[g] / weight_sum[g]
            else:
                group_mean[g] = np.nan

        for j in range(n_cols):
            gid = groups[i, j]
            if gid >= 0:
                result[i, j] = group_mean[gid]
            else:
                result[i, j] = np.nan

    return result


@njit(cache=True)
def group_backfill_kernel(
    values: np.ndarray,
    groups: np.ndarray,
    d: int,
    std: float,
) -> np.ndarray:
    """Fill NaN with winsorized group mean over d rows lookback."""
    n_rows, n_cols = values.shape
    result = values.copy()

    for i in range(n_rows):
        for j in range(n_cols):
            if not np.isnan(values[i, j]):
                continue

            gid = groups[i, j]
            if gid < 0:
                continue

            start_row = max(0, i - d + 1)
            group_vals = []
            for row in range(start_row, i + 1):
                for col in range(n_cols):
                    if groups[row, col] == gid and not np.isnan(values[row, col]):
                        group_vals.append(values[row, col])

            if len(group_vals) == 0:
                continue

            vals_arr = np.array(group_vals)
            mean = np.mean(vals_arr)
            std_val = np.std(vals_arr)
            if std_val > 0:
                lower = mean - std * std_val
                upper = mean + std * std_val
                clipped_sum = 0.0
                for v in vals_arr:
                    if v < lower:
                        clipped_sum += lower
                    elif v > upper:
                        clipped_sum += upper
                    else:
                        clipped_sum += v
                result[i, j] = clipped_sum / len(vals_arr)
            else:
                result[i, j] = mean

    return result


# ============================================================================
# Cross-sectional kernels
# ============================================================================


@njit(cache=True)
def _inv_norm_approx(p: float) -> float:
    """Approximate inverse normal CDF (Abramowitz and Stegun).

    Accurate to ~1e-9 for p in (0, 1).
    """
    if p <= 0.0:
        return -np.inf
    if p >= 1.0:
        return np.inf

    # Coefficients for rational approximation
    a = np.array([
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ])
    b = np.array([
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ])
    c = np.array([
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ])
    d = np.array([
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ])

    p_low = 0.02425
    p_high = 1.0 - p_low

    if p < p_low:
        # Lower tail
        q = np.sqrt(-2.0 * np.log(p))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
    elif p <= p_high:
        # Central region
        q = p - 0.5
        r = q * q
        return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
               (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0)
    else:
        # Upper tail
        q = np.sqrt(-2.0 * np.log(1.0 - p))
        return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)


@njit(cache=True, parallel=True)
def quantile_transform_2d(
    values: np.ndarray,
    driver: int,
    sigma: float,
) -> np.ndarray:
    """Quantile transformation for 2D array (rows × cols).

    Args:
        values: 2D array of values
        driver: 0=gaussian, 1=uniform, 2=cauchy
        sigma: Scale parameter

    Returns:
        Transformed values
    """
    n_rows, n_cols = values.shape
    result = np.empty((n_rows, n_cols), dtype=np.float64)

    for i in prange(n_rows):
        row = values[i]
        out = result[i]

        # Count valid values and get indices
        valid_count = 0
        for j in range(n_cols):
            if not np.isnan(row[j]):
                valid_count += 1
            else:
                out[j] = np.nan

        if valid_count <= 1:
            for j in range(n_cols):
                if not np.isnan(row[j]):
                    out[j] = 0.0
            continue

        # Extract valid values and their indices
        valid_vals = np.empty(valid_count, dtype=np.float64)
        valid_idx = np.empty(valid_count, dtype=np.int64)
        k = 0
        for j in range(n_cols):
            if not np.isnan(row[j]):
                valid_vals[k] = row[j]
                valid_idx[k] = j
                k += 1

        # Rank using argsort
        order = np.argsort(valid_vals)
        ranks = np.empty(valid_count, dtype=np.float64)
        for k in range(valid_count):
            ranks[order[k]] = k

        # Normalize to [0, 1] and shift to [1/N, 1-1/N]
        n = float(valid_count)
        for k in range(valid_count):
            r = ranks[k] / (n - 1.0)  # [0, 1]
            shifted = 1.0 / n + r * (1.0 - 2.0 / n)  # [1/N, 1-1/N]

            # Apply inverse CDF
            if driver == 0:  # gaussian
                transformed = _inv_norm_approx(shifted) * sigma
            elif driver == 1:  # uniform
                transformed = (shifted - 0.5) * 2.0 * sigma
            else:  # cauchy
                transformed = np.tan(np.pi * (shifted - 0.5)) * sigma

            out[valid_idx[k]] = transformed

    return result
