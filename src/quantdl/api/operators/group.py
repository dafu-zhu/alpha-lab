"""Group operators for wide tables.

All operators work on groups defined by a separate group DataFrame:
- First column (date) is unchanged
- Operations applied within groups across symbols at each date
"""

import polars as pl


def _get_value_cols(df: pl.DataFrame) -> list[str]:
    """Get value columns (all except first which is date)."""
    return df.columns[1:]


def _to_long_with_group(
    x: pl.DataFrame,
    group: pl.DataFrame,
) -> tuple[pl.DataFrame, str, list[str]]:
    """Convert x and group to long format and join.

    Returns joined long DataFrame, date column name, and original value columns.
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

    joined = x_long.join(g_long, on=[date_col, "symbol"])
    return joined, date_col, value_cols


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
    joined, date_col, value_cols = _to_long_with_group(x, group)

    result = joined.with_columns(
        (pl.col("value") - pl.col("value").mean().over([date_col, "group_id"]))
        .alias("value")
    )

    return _pivot_back(result, date_col, value_cols)


def group_zscore(x: pl.DataFrame, group: pl.DataFrame) -> pl.DataFrame:
    """Z-score within groups: (x - group_mean) / group_std.

    Args:
        x: Wide DataFrame with date + symbol columns
        group: Wide DataFrame with group assignments (same shape as x)

    Returns:
        Wide DataFrame with group z-scored values
    """
    joined, date_col, value_cols = _to_long_with_group(x, group)

    result = joined.with_columns(
        ((pl.col("value") - pl.col("value").mean().over([date_col, "group_id"]))
         / pl.col("value").std().over([date_col, "group_id"]))
        .alias("value")
    )

    return _pivot_back(result, date_col, value_cols)


def group_scale(x: pl.DataFrame, group: pl.DataFrame) -> pl.DataFrame:
    """Min-max scale within groups to [0, 1].

    Args:
        x: Wide DataFrame with date + symbol columns
        group: Wide DataFrame with group assignments (same shape as x)

    Returns:
        Wide DataFrame with group-scaled values in [0, 1]
    """
    joined, date_col, value_cols = _to_long_with_group(x, group)

    result = joined.with_columns(
        ((pl.col("value") - pl.col("value").min().over([date_col, "group_id"]))
         / (pl.col("value").max().over([date_col, "group_id"])
            - pl.col("value").min().over([date_col, "group_id"])))
        .alias("value")
    )

    return _pivot_back(result, date_col, value_cols)


def group_rank(x: pl.DataFrame, group: pl.DataFrame) -> pl.DataFrame:
    """Rank within groups, normalized to [0, 1].

    Single-member groups return 0.5.

    Args:
        x: Wide DataFrame with date + symbol columns
        group: Wide DataFrame with group assignments (same shape as x)

    Returns:
        Wide DataFrame with group rank values in [0, 1]
    """
    joined, date_col, value_cols = _to_long_with_group(x, group)

    result = joined.with_columns(
        pl.when(pl.col("value").count().over([date_col, "group_id"]) == 1)
        .then(0.5)
        .otherwise(
            (pl.col("value").rank(method="ordinal").over([date_col, "group_id"]) - 1)
            / (pl.col("value").count().over([date_col, "group_id"]) - 1)
        )
        .alias("value")
    )

    return _pivot_back(result, date_col, value_cols)


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
            import numpy as np
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
