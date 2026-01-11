import pandas as pd
from sqlalchemy.exc import PendingRollbackError

from quantdl.utils.wrds import raw_sql_with_retry


def test_raw_sql_with_retry_success():
    db = type("Db", (), {})()
    db.raw_sql = lambda sql, **kwargs: pd.DataFrame({"a": [1]})

    result = raw_sql_with_retry(db, "select 1")

    assert result["a"][0] == 1


def test_raw_sql_with_retry_rolls_back_and_retries():
    calls = {"count": 0}

    class Conn:
        def __init__(self) -> None:
            self.rolled_back = False

        def rollback(self) -> None:
            self.rolled_back = True

    class Db:
        def __init__(self) -> None:
            self.connection = Conn()

        def raw_sql(self, sql, **kwargs):
            calls["count"] += 1
            if calls["count"] == 1:
                raise PendingRollbackError("pending rollback")
            return pd.DataFrame({"a": [2]})

    db = Db()

    result = raw_sql_with_retry(db, "select 1")

    assert calls["count"] == 2
    assert db.connection.rolled_back is True
    assert result["a"][0] == 2


def test_raw_sql_with_retry_without_connection_attribute():
    calls = {"count": 0}

    class Db:
        def raw_sql(self, sql, **kwargs):
            calls["count"] += 1
            if calls["count"] == 1:
                raise PendingRollbackError("pending rollback")
            return pd.DataFrame({"a": [3]})

    db = Db()

    result = raw_sql_with_retry(db, "select 1")

    assert calls["count"] == 2
    assert result["a"][0] == 3
