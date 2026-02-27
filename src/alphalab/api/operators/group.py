"""Group operators for wide tables.

All operators work on groups defined by a separate group DataFrame:
- First column (date) is unchanged
- Operations applied within groups across symbols at each date

Uses numba-optimized row-wise operations for performance.
"""

import numpy as np
import polars as pl

from alphalab.api.operators._numba_kernels import (
    group_backfill_kernel,
    group_mean_rows,
    group_neutralize_rows,
    group_rank_rows,
    group_scale_rows,
    group_zscore_rows,
)


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

    # Extract values as numpy
    values = x.select(value_cols).to_numpy().astype(np.float64)

    # Check first column dtype
    first_col_dtype = group[group_cols[0]].dtype
    is_string = first_col_dtype == pl.Utf8 or first_col_dtype == pl.Categorical

    if is_string:
        # Build global enum from all unique values for consistent encoding
        all_vals: set[str] = set()
        for c in group_cols:
            all_vals.update(group[c].drop_nulls().unique().to_list())
        enum_type = pl.Enum(sorted(all_vals))

        # Encode all columns with global enum
        encoded = group.select([
            group.columns[0],
            *[pl.col(c).cast(enum_type).to_physical().cast(pl.Int32).fill_null(-1)
              for c in group_cols]
        ])
    else:
        # Numeric - convert NaN to -1
        encoded = group.select([
            group.columns[0],
            *[pl.col(c).fill_null(-1).cast(pl.Int32) for c in group_cols]
        ])

    # Fast path: columns match exactly
    if value_cols == group_cols:
        groups = encoded.select(value_cols).to_numpy().astype(np.int32)
        return values, groups, date_col, value_cols

    # Slow path: align columns
    common_cols = set(value_cols) & set(group_cols)
    groups = np.full((group.height, len(value_cols)), -1, dtype=np.int32)
    encoded_arr = encoded.select(group_cols).to_numpy()
    group_col_to_idx = {col: i for i, col in enumerate(group_cols)}

    for j, col in enumerate(value_cols):
        if col in common_cols:
            groups[:, j] = encoded_arr[:, group_col_to_idx[col]]

    return values, groups, date_col, value_cols


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


def group_neutralize(x: pl.DataFrame, group: pl.DataFrame) -> pl.DataFrame:
    """Subtract group mean from each value.

    Args:
        x: Wide DataFrame with date + symbol columns
        group: Wide DataFrame with group assignments (same shape as x)

    Returns:
        Wide DataFrame with group-neutralized values
    """
    values, groups, date_col, value_cols = _align_and_extract(x, group)
    result = group_neutralize_rows(values, groups)
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
    result = group_zscore_rows(values, groups)
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
    result = group_scale_rows(values, groups)
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
    result = group_rank_rows(values, groups)
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

    # Extract aligned arrays
    values = x.select(value_cols).to_numpy().astype(np.float64)
    weights = weight.select(value_cols).to_numpy().astype(np.float64)

    # Get group IDs (reuse alignment logic)
    _, groups, _, _ = _align_and_extract(x, group)

    result = group_mean_rows(values, weights, groups)
    return _rebuild_dataframe(result, x, date_col, value_cols)


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
    values, groups, date_col, value_cols = _align_and_extract(x, group)
    result = group_backfill_kernel(values, groups, d, std)

    # Convert NaN back to null (preserves Polars null semantics)
    return pl.DataFrame({
        date_col: x[date_col],
        **{col: pl.Series(result[:, j]).fill_nan(None) for j, col in enumerate(value_cols)}
    })
