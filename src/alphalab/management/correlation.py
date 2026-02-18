"""Correlation check logic."""
from __future__ import annotations

import polars as pl

from alphalab.management.models import CheckResult, Correlation, Version

CORR_THRESHOLD = 0.7
IMPROVEMENT_THRESHOLD = 10.0  # percent


def check_correlation(
    new_pnl: pl.Series,
    new_sharpe: float,
    existing: list[Version],
) -> CheckResult:
    """
    Check new alpha against all submitted alphas.

    Pass criteria (per existing alpha):
        corr < 0.7  OR  (corr >= 0.7 AND improvement >= 10%)

    Overall pass: ALL individual checks pass

    Args:
        new_pnl: Daily returns of the new alpha
        new_sharpe: Sharpe ratio of the new alpha
        existing: List of submitted alpha versions with pnl_data

    Returns:
        CheckResult with passed flag, full details, and blocking alphas
    """
    if not existing:
        return CheckResult(details=[])

    results = []
    for version in existing:
        if not version.pnl_data or version.sharpe is None:
            continue

        exist_pnl = pl.Series([d["ret"] for d in version.pnl_data])

        # Align lengths (use shorter)
        min_len = min(len(new_pnl), len(exist_pnl))
        if min_len == 0:
            continue

        # Compute Pearson correlation via DataFrame
        df = pl.DataFrame({"new": new_pnl[:min_len], "exist": exist_pnl[:min_len]})
        corr = df.select(pl.corr("new", "exist")).item()

        if corr is None:
            continue

        if version.sharpe == 0:
            improv = float('inf') if new_sharpe > 0 else 0.0
        else:
            improv = (new_sharpe - version.sharpe) / version.sharpe * 100
        passed = corr < CORR_THRESHOLD or improv >= IMPROVEMENT_THRESHOLD

        results.append(
            Correlation(
                new_alpha_id="",  # filled by caller
                exist_alpha_id=version.id,
                corr=corr,
                new_sharpe=new_sharpe,
                exist_sharpe=version.sharpe,
                improvement=improv,
                passed=passed,
            )
        )

    return CheckResult(details=results)
