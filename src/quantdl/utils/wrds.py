from typing import Any, Optional

import pandas as pd
from sqlalchemy.exc import PendingRollbackError


def raw_sql_with_retry(db: Any, sql: str, **kwargs: Any) -> pd.DataFrame:
    """
    Execute WRDS raw_sql with a rollback on pending-rollback errors.

    This clears failed transaction state and retries once.
    """
    try:
        return db.raw_sql(sql, **kwargs)
    except PendingRollbackError:
        connection = getattr(db, "connection", None)
        if connection is not None:
            connection.rollback()
        return db.raw_sql(sql, **kwargs)
