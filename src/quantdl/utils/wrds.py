from typing import Any, Optional
import time

import pandas as pd
from sqlalchemy.exc import PendingRollbackError, OperationalError


def raw_sql_with_retry(db: Any, sql: str, max_retries: int = 2, **kwargs: Any) -> pd.DataFrame:
    """
    Execute WRDS raw_sql with retry logic for connection errors.

    Handles:
    - PendingRollbackError: rollback and retry
    - OperationalError (connection closed): retry with backoff

    Note: Connection must be recreated by caller if retry fails.

    :param db: WRDS connection object
    :param sql: SQL query string
    :param max_retries: Maximum retry attempts (default: 2)
    :param kwargs: Additional arguments to pass to raw_sql
    :return: Query results as DataFrame
    """
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            return db.raw_sql(sql, **kwargs)
        except PendingRollbackError as e:
            last_error = e
            connection = getattr(db, "connection", None)
            if connection is not None:
                try:
                    connection.rollback()
                except:
                    pass
            if attempt < max_retries:
                time.sleep(1)
            else:
                raise
        except OperationalError as e:
            last_error = e
            # Connection closed or network error
            if "server closed the connection" in str(e) or "closed the connection" in str(e):
                if attempt < max_retries:
                    time.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s
                else:
                    raise
            else:
                # Other operational errors should not be retried
                raise

    # Should not reach here, but just in case
    raise last_error or RuntimeError(f"Failed to execute query after {max_retries + 1} attempts")
