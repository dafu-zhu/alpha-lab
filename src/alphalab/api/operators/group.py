"""Group operators for wide tables.

All operators work on groups defined by a separate group DataFrame:
- First column (date) is unchanged
- Operations applied within groups across symbols at each date

Uses numpy row-wise operations for performance.
"""

import numpy as np
import polars as pl
from numba import njit


def _get_value_cols(df: pl.DataFrame) -> list[str]:
    """Get value columns (all except first which is date)."""
    return df.columns[1:]


def _align_and_extract(
    x: pl.DataFrame,
    group: pl.DataFrame,
) -> tuple[np.ndarray, np.ndarray, str, list[str]]:
    """Align x and group DataFrames, extract numpy arrays with integer group IDs."""
    date_col = x.columns[0]
    value_cols = _get_value_cols(x)
    group_cols = _get_value_cols(group)

    # Align columns - use x's columns, get corresponding group values
    values = x.select(value_cols).to_numpy().astype(np.float64)

    # Build group array aligned to x's columns
    group_df = group.select(group_cols)
    n_rows = group_df.height

    # Map columns from x to group (handle different column orders/subsets)
    x_col_set = set(value_cols)
    group_col_set = set(group_cols)
    common_cols = x_col_set & group_col_set

    # Build column index mapping: x column index -> group column index
    group_col_to_idx = {col: i for i, col in enumerate(group_cols)}

    # Extract group values, map strings to integers
    first_col_dtype = group_df[group_cols[0]].dtype
    is_string = first_col_dtype == pl.Utf8 or first_col_dtype == pl.Categorical

    if is_string:
        # Collect all unique group values
        all_values = set()
        for col in group_cols:
            all_values.update(v for v in group_df[col].to_list() if v is not None)
        group_map = {v: i for i, v in enumerate(sorted(all_values))}
        n_groups = len(group_map)

        # Build aligned group array
        groups = np.full((n_rows, len(value_cols)), -1, dtype=np.int32)
        for j, col in enumerate(value_cols):
            if col in common_cols:
                g_idx = group_col_to_idx[col]
                col_values = group_df[group_cols[g_idx]].to_list()
                for i, v in enumerate(col_values):
                    if v is not None:
                        groups[i, j] = group_map[v]
    else:
        # Numeric group IDs
        groups = np.full((n_rows, len(value_cols)), -1, dtype=np.int32)
        for j, col in enumerate(value_cols):
            if col in common_cols:
                g_idx = group_col_to_idx[col]
                col_values = group_df[group_cols[g_idx]].to_numpy()
                for i, v in enumerate(col_values):
                    if not np.isnan(v):
                        groups[i, j] = int(v)

    return values, groups, date_col, value_cols


@njit(cache=True)
def _group_neutralize_rows(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Subtract group mean from each value, row by row."""
    n_rows, n_cols = values.shape
    result = np.empty_like(values)

    for i in range(n_rows):
        # Find max group ID in this row
        max_gid = -1
        for j in range(n_cols):
            if groups[i, j] > max_gid:
                max_gid = groups[i, j]

        if max_gid < 0:
            # No valid groups
            for j in range(n_cols):
                result[i, j] = np.nan
            continue

        # Compute group sums and counts
        n_groups = max_gid + 1
        group_sum = np.zeros(n_groups, dtype=np.float64)
        group_count = np.zeros(n_groups, dtype=np.int32)

        for j in range(n_cols):
            gid = groups[i, j]
            val = values[i, j]
            if gid >= 0 and not np.isnan(val):
                group_sum[gid] += val
                group_count[gid] += 1

        # Compute means
        group_mean = np.empty(n_groups, dtype=np.float64)
        for g in range(n_groups):
            if group_count[g] > 0:
                group_mean[g] = group_sum[g] / group_count[g]
            else:
                group_mean[g] = np.nan

        # Subtract means
        for j in range(n_cols):
            gid = groups[i, j]
            val = values[i, j]
            if gid >= 0 and not np.isnan(val):
                result[i, j] = val - group_mean[gid]
            else:
                result[i, j] = np.nan

    return result


@njit(cache=True)
def _group_zscore_rows(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
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
def _group_scale_rows(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
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
def _group_rank_rows(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
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

        # Count valid values per group
        group_count = np.zeros(n_groups, dtype=np.int32)
        for j in range(n_cols):
            gid = groups[i, j]
            val = values[i, j]
            if gid >= 0 and not np.isnan(val):
                group_count[gid] += 1

        # For each column, count how many in same group have smaller value
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

            # Count values less than val in same group
            rank = 0
            for k in range(n_cols):
                if groups[i, k] == gid and not np.isnan(values[i, k]):
                    if values[i, k] < val:
                        rank += 1

            result[i, j] = rank / (count - 1)

    return result


def _rebuild_dataframe(
    result: np.ndarray,
    x: pl.DataFrame,
    date_col: str,
    value_cols: list[str],
) -> pl.DataFrame:
    """Rebuild DataFrame from numpy result array."""
    return pl.DataFrame({
        date_col: x[date_col],
        **{col: result[:, j] for j, col in enumerate(value_cols)}
    })


def _pivot_back(
    df: pl.DataFrame,
    date_col: str,
    value_cols: list[str],
) -> pl.DataFrame:
    """Pivot long format back to wide and restore column order."""
    wide = df.pivot(values="value", index=date_col, on="symbol")
    return wide.select([date_col, *value_cols])


def group_neutralize(x: pl.DataFrame, group: pl.DataFrame) -> pl.DataFrame:
    """Subtract group mean from each value.

    Args:
        x: Wide DataFrame with date + symbol columns
        group: Wide DataFrame with group assignments (same shape as x)

    Returns:
        Wide DataFrame with group-neutralized values
    """
    values, groups, date_col, value_cols = _align_and_extract(x, group)
    result = _group_neutralize_rows(values, groups)
    return _rebuild_dataframe(result, x, date_col, value_cols)


def group_zscore(x: pl.DataFrame, group: pl.DataFrame) -> pl.DataFrame:
    """Z-score within groups: (x - group_mean) / group_std.

    Args:
        x: Wide DataFrame with date + symbol columns
        group: Wide DataFrame with group assignments (same shape as x)

    Returns:
        Wide DataFrame with group z-scored values
    """
    values, groups, date_col, value_cols = _align_and_extract(x, group)
    result = _group_zscore_rows(values, groups)
    return _rebuild_dataframe(result, x, date_col, value_cols)


def group_scale(x: pl.DataFrame, group: pl.DataFrame) -> pl.DataFrame:
    """Min-max scale within groups to [0, 1].

    Args:
        x: Wide DataFrame with date + symbol columns
        group: Wide DataFrame with group assignments (same shape as x)

    Returns:
        Wide DataFrame with group-scaled values in [0, 1]
    """
    values, groups, date_col, value_cols = _align_and_extract(x, group)
    result = _group_scale_rows(values, groups)
    return _rebuild_dataframe(result, x, date_col, value_cols)


def group_rank(x: pl.DataFrame, group: pl.DataFrame) -> pl.DataFrame:
    """Rank within groups, normalized to [0, 1].

    Single-member groups return 0.5.

    Args:
        x: Wide DataFrame with date + symbol columns
        group: Wide DataFrame with group assignments (same shape as x)

    Returns:
        Wide DataFrame with group rank values in [0, 1]
    """
    values, groups, date_col, value_cols = _align_and_extract(x, group)
    result = _group_rank_rows(values, groups)
    return _rebuild_dataframe(result, x, date_col, value_cols)


def group_mean(
    x: pl.DataFrame,
    weight: pl.DataFrame,
    group: pl.DataFrame,
) -> pl.DataFrame:
    """Weighted mean within groups.

    Computes sum(x * weight) / sum(weight) for each group.

    Args:
        x: Wide DataFrame with date + symbol columns
        weight: Wide DataFrame with weights (same shape as x)
        group: Wide DataFrame with group assignments (same shape as x)

    Returns:
        Wide DataFrame with weighted group means (broadcast to all members)
    """
    date_col = x.columns[0]
    value_cols = _get_value_cols(x)

    x_long = x.unpivot(
        index=date_col,
        on=value_cols,
        variable_name="symbol",
        value_name="value",
    )

    w_long = weight.unpivot(
        index=date_col,
        on=_get_value_cols(weight),
        variable_name="symbol",
        value_name="weight",
    )

    g_long = group.unpivot(
        index=date_col,
        on=_get_value_cols(group),
        variable_name="symbol",
        value_name="group_id",
    )

    joined = x_long.join(w_long, on=[date_col, "symbol"]).join(
        g_long, on=[date_col, "symbol"]
    )

    result = joined.with_columns(
        ((pl.col("value") * pl.col("weight")).sum().over([date_col, "group_id"])
         / pl.col("weight").sum().over([date_col, "group_id"]))
        .alias("value")
    )

    return _pivot_back(result, date_col, value_cols)


def group_backfill(
    x: pl.DataFrame,
    group: pl.DataFrame,
    d: int,
    std: float = 4.0,
) -> pl.DataFrame:
    """Fill NaN with winsorized group mean over d days.

    For each NaN, looks back up to d days and computes the winsorized
    mean of non-NaN group values. If all values in the lookback window
    are NaN, keeps NaN.

    Args:
        x: Wide DataFrame with date + symbol columns
        group: Wide DataFrame with group assignments (same shape as x)
        d: Number of days to look back
        std: Number of standard deviations for winsorization (default: 4.0)

    Returns:
        Wide DataFrame with NaN values filled
    """
    date_col = x.columns[0]
    value_cols = _get_value_cols(x)

    x_long = x.unpivot(
        index=date_col,
        on=value_cols,
        variable_name="symbol",
        value_name="value",
    )

    g_long = group.unpivot(
        index=date_col,
        on=_get_value_cols(group),
        variable_name="symbol",
        value_name="group_id",
    )

    joined = x_long.join(g_long, on=[date_col, "symbol"]).sort([date_col, "symbol"])

    dates = joined.select(date_col).unique().sort(date_col).to_series().to_list()
    date_to_idx = {dt: i for i, dt in enumerate(dates)}

    result_rows = []

    for row in joined.iter_rows(named=True):
        date_val = row[date_col]
        symbol = row["symbol"]
        value = row["value"]
        group_id = row["group_id"]

        if value is not None:
            result_rows.append({
                date_col: date_val,
                "symbol": symbol,
                "value": value,
            })
            continue

        current_idx = date_to_idx[date_val]
        start_idx = max(0, current_idx - d + 1)
        lookback_dates = dates[start_idx:current_idx + 1]

        group_vals = (
            joined
            .filter(
                (pl.col(date_col).is_in(lookback_dates))
                & (pl.col("group_id") == group_id)
                & (pl.col("value").is_not_null())
            )
            .select("value")
            .to_series()
            .to_list()
        )

        if len(group_vals) == 0:
            fill_value = None
        else:
            vals = np.array(group_vals)
            mean = np.mean(vals)
            std_val = np.std(vals)
            if std_val > 0:
                lower = mean - std * std_val
                upper = mean + std * std_val
                clipped = np.clip(vals, lower, upper)
                fill_value = float(np.mean(clipped))
            else:
                fill_value = float(mean)

        result_rows.append({
            date_col: date_val,
            "symbol": symbol,
            "value": fill_value,
        })

    result = pl.DataFrame(result_rows)
    return _pivot_back(result, date_col, value_cols)
