## Current Fundamental Data Storage Strategy

Storage path: `data/raw/fundamental/{symbol}/fundamental.parquet`   

Each symbol's years of historical fundamental data is stored sparsely in a single parquet file. 

Concern: Consider rebranding, symbol is not a longevitable identifier for stocks (e.g. META in 2002 and META in 2023 are not identical). The current strategy has risk of losing data.

## Proposal

Use CIK as identifier for fundamental data. For example `data/raw/fundamental/320193/fundamental.parquet`

Evaluate this proposal, and raise improvement, if any

## Current updating logic

Load current existing symbols from Nasdaq Trader. Check each symbol to find if it has recent filing on EDGAR. Collect symbols with recent filings. Refetch and recalculate raw, TTM and derived fundamental data for these symbols. Overwrite their data on S3.

## Issue

1. method `get_symbols_with_recent_filings` in update/app.py is processing too slow
2. Refetching and recalculating is not efficient, while downloading from S3 can generate costs

## Task

Make a plan to solve the above issues