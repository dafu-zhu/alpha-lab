"""Transformational operators for wide tables.

These operators transform data based on conditional logic (e.g., trade signals).
"""

import numpy as np
import polars as pl


def _get_value_cols(df: pl.DataFrame) -> list[str]:
    """Get value columns (all except first which is date)."""
    return df.columns[1:]


def trade_when(
    trigger_trade: pl.DataFrame,
    alpha: pl.DataFrame,
    trigger_exit: pl.DataFrame | int | float,
) -> pl.DataFrame:
    """Conditional alpha with entry/exit signals and carry-forward.

    Per row per symbol:
      - if trigger_exit > 0: result = NaN (exit position)
      - elif trigger_trade > 0: result = alpha (enter/update position)
      - else: result = previous result (carry forward)

    Args:
        trigger_trade: Wide DataFrame with trade entry signals (>0 = enter)
        alpha: Wide DataFrame with alpha values to use on entry
        trigger_exit: Wide DataFrame with exit signals (>0 = exit),
                      or a scalar (e.g., -1 means never exit)

    Returns:
        Wide DataFrame with conditional alpha values
    """
    date_col = trigger_trade.columns[0]
    value_cols = _get_value_cols(trigger_trade)

    # Scalar exit: constant for all rows
    scalar_exit = None
    if isinstance(trigger_exit, (int, float)):
        scalar_exit = trigger_exit

    result_cols = [trigger_trade[date_col]]

    for col in value_cols:
        trade_arr = trigger_trade[col].to_numpy().astype(np.float64)
        alpha_arr = alpha[col].to_numpy().astype(np.float64)

        if scalar_exit is not None:
            exit_arr = np.full(len(trade_arr), scalar_exit, dtype=np.float64)
        else:
            exit_arr = trigger_exit[col].to_numpy().astype(np.float64)

        out = np.full(len(trade_arr), np.nan)
        prev = np.nan
        for i in range(len(trade_arr)):
            if exit_arr[i] > 0:
                prev = np.nan
                out[i] = np.nan
            elif trade_arr[i] > 0:
                prev = alpha_arr[i]
                out[i] = prev
            else:
                out[i] = prev

        result_cols.append(pl.Series(col, out))

    return pl.DataFrame(result_cols)
