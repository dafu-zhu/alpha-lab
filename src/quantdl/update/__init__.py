"""
Daily Update Module for US Equity Data Lake
============================================

This module provides automated daily update functionality for:
- Daily ticks (OHLCV data)
- Minute ticks (intraday data)
- Fundamental data (financial statements)

The update process intelligently checks:
- Market calendar to determine if updates are needed
- EDGAR for new filings before updating fundamental data

Usage:
    from quantdl.update.app import DailyUpdateApp

    app = DailyUpdateApp()
    app.run_daily_update()  # Updates yesterday's data

Command Line:
    python -m quantdl.update.app
    python -m quantdl.update.app --date 2025-01-09
    python -m quantdl.update.app --no-ticks
    python -m quantdl.update.app --no-fundamentals
    python -m quantdl.update.app --lookback 14
"""

from quantdl.update.app import DailyUpdateApp

__all__ = ['DailyUpdateApp']
