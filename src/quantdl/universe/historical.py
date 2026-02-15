"""
Historical universe functions.

Primary: get_hist_universe_local() — uses local security master parquet.
Legacy: get_hist_universe_crsp() / get_hist_universe_nasdaq() — require wrds package.
"""

import polars as pl
from datetime import date
from pathlib import Path
from typing import Optional, Any
import os
from dotenv import load_dotenv

from quantdl.utils.validation import validate_year

load_dotenv()

try:
    import wrds
    HAS_WRDS = True
except ImportError:
    HAS_WRDS = False

# Default local path for security master
LOCAL_MASTER_PATH = Path("data/master/security_master.parquet")


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
            "Run: python scripts/download_security_master.py"
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


def get_hist_universe_crsp(year: int, month: int = 12, db: Any = None) -> pl.DataFrame:
    """
    Historical universe from CRSP database.

    Requires: wrds package and WRDS credentials.

    :param year: Year (e.g., 2024)
    :param month: Deprecated, kept for backward compatibility
    :param db: Optional WRDS connection
    :return: DataFrame with columns: Ticker, Name, PERMNO
    """
    if not HAS_WRDS:
        raise ImportError(
            "get_hist_universe_crsp() requires the wrds package. "
            "Use get_hist_universe_local() instead."
        )

    from quantdl.utils.wrds import raw_sql_with_retry
    from sqlalchemy.exc import OperationalError

    close_db = False
    if db is None:
        username = os.getenv('WRDS_USERNAME')
        password = os.getenv('WRDS_PASSWORD')
        if not username or not password:
            raise ValueError(
                "WRDS credentials not found. Set WRDS_USERNAME and WRDS_PASSWORD environment variables."
            )
        db = wrds.Connection(wrds_username=username, wrds_password=password)
        close_db = True

    try:
        validated_year = validate_year(year)
        year_start = f"{validated_year}-01-01"
        year_end = f"{validated_year}-12-31"

        sql = f"""
        SELECT DISTINCT
            ticker, tsymbol, permno, comnam, shrcd, exchcd
        FROM crsp_a_stock.dsenames
        WHERE namedt <= '{year_end}'
          AND nameendt >= '{year_start}'
          AND ticker IS NOT NULL
          AND shrcd IN (10, 11)
          AND exchcd IN (1, 2, 3)
        ORDER BY ticker;
        """

        try:
            df = raw_sql_with_retry(db, sql)
        except OperationalError as e:
            if close_db and ("closed the connection" in str(e) or "server closed" in str(e)):
                username = os.getenv('WRDS_USERNAME')
                password = os.getenv('WRDS_PASSWORD')
                try:
                    db.close()
                except:
                    pass
                db = wrds.Connection(wrds_username=username, wrds_password=password)
                df = raw_sql_with_retry(db, sql)
            else:
                raise

        if df.empty:
            return pl.DataFrame({
                'Ticker': [],
                'Name': [],
                'PERMNO': []
            })

        result = pl.DataFrame({
            'Ticker': df['tsymbol'].str.upper().tolist(),
            'Name': df['comnam'].tolist(),
            'PERMNO': df['permno'].tolist()
        }).unique(subset=['Ticker'], maintain_order=True)

        return result

    finally:
        if close_db:
            db.close()


def get_hist_universe_nasdaq(
    year: int,
    with_validation: bool = True,
    security_master: Any = None,
    db: Any = None
) -> pl.DataFrame:
    """
    Historical universe with symbols in Nasdaq format.

    Requires: wrds package.

    :param year: Year (e.g., 2024)
    :param with_validation: Use SecurityMaster for symbol validation
    :param security_master: Optional SecurityMaster instance
    :param db: Optional WRDS connection
    :return: DataFrame with columns: Ticker (Nasdaq format), Name, PERMNO
    """
    if not HAS_WRDS:
        raise ImportError(
            "get_hist_universe_nasdaq() requires the wrds package. "
            "Use get_hist_universe_local() instead."
        )

    from quantdl.master.security_master import SymbolNormalizer, SecurityMaster as SM

    crsp_df = get_hist_universe_crsp(year, month=12, db=db)
    crsp_symbols = crsp_df['Ticker'].to_list()

    sm_to_close = None
    if with_validation:
        if security_master is not None:
            sm = security_master
        else:
            sm = SM()
            sm_to_close = sm

        normalizer = SymbolNormalizer(security_master=sm)
    else:
        normalizer = SymbolNormalizer()

    reference_day = f"{year}-12-31" if with_validation else None
    nasdaq_symbols = normalizer.batch_normalize(crsp_symbols, day=reference_day)

    if sm_to_close is not None:
        sm_to_close.close()

    result = pl.DataFrame({
        'Ticker': nasdaq_symbols,
        'Name': crsp_df['Name'].to_list(),
        'PERMNO': crsp_df['PERMNO'].to_list()
    })

    return result
