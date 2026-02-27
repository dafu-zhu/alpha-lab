"""Transformational operators for wide tables.

Transform data based on conditional logic (e.g., trade signals).
Uses numba-optimized kernels for performance.
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

    Uses numba-optimized column processing with parallel execution.

    Args:
        trigger_trade: Wide DataFrame with trade entry signals (>0 = enter)
        alpha: Wide DataFrame with alpha values to use on entry
        trigger_exit: Wide DataFrame with exit signals (>0 = exit),
                      or a scalar (e.g., -1 means never exit)

    Returns:
        Wide DataFrame with conditional alpha values
    """
    from concurrent.futures import ThreadPoolExecutor

    from alphalab.api.operators._numba_kernels import trade_when_column

    date_col = trigger_trade.columns[0]
    value_cols = _get_value_cols(trigger_trade)

    # Extract arrays
    trade_values = trigger_trade.select(value_cols).to_numpy().astype(np.float64)
    alpha_values = alpha.select(value_cols).to_numpy().astype(np.float64)

    # Handle scalar exit
    if isinstance(trigger_exit, (int, float)):
        exit_values = np.full_like(trade_values, trigger_exit)
    else:
        exit_values = trigger_exit.select(value_cols).to_numpy().astype(np.float64)

    def process_col(idx: int) -> tuple[str, np.ndarray]:
        return (
            value_cols[idx],
            trade_when_column(
                trade_values[:, idx],
                alpha_values[:, idx],
                exit_values[:, idx],
            ),
        )

    with ThreadPoolExecutor(max_workers=min(8, len(value_cols))) as executor:
        col_results = dict(executor.map(process_col, range(len(value_cols))))

    return pl.DataFrame({date_col: trigger_trade[date_col], **col_results})
