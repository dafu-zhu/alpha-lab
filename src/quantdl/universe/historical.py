"""
Historical universe functions.

Primary: get_hist_universe_local() — uses local security master parquet.
"""

import polars as pl
from datetime import date
from pathlib import Path
from typing import Optional, Any
from dotenv import load_dotenv

from quantdl.utils.validation import validate_year

load_dotenv()

# Default local path for security master
LOCAL_MASTER_PATH = Path("data/meta/master/security_master.parquet")


def get_hist_universe_local(
    year: int,
    security_master: Any = None
) -> pl.DataFrame:
    """
    Historical universe from local security master parquet.

    Filters securities active at any point during the year.
    No WRDS or exchcd filtering (uses all securities in the master).

    :param year: Year (e.g., 2024)
    :param security_master: Optional SecurityMaster instance (uses master_tb attribute)
    :return: DataFrame with columns: Ticker, Name
    """
    validated_year = validate_year(year)

    if security_master is not None:
        master_df = security_master.master_tb
    elif LOCAL_MASTER_PATH.exists():
        master_df = pl.read_parquet(str(LOCAL_MASTER_PATH))
    else:
        raise FileNotFoundError(
            f"No security master found at {LOCAL_MASTER_PATH}. "
            "Run: python scripts/build_security_master.py"
        )

    year_start = date(validated_year, 1, 1)
    year_end = date(validated_year, 12, 31)

    result = (
        master_df
        .filter(
            pl.col('start_date').le(year_end),
            pl.col('end_date').ge(year_start),
        )
        .select(
            pl.col('symbol').alias('Ticker'),
            pl.col('company').alias('Name'),
        )
        .unique(subset=['Ticker'], maintain_order=True)
    )

    return result
