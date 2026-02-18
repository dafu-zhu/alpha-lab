"""Tests for database schema and connection."""
import sqlite3
from pathlib import Path
import pytest


def test_init_db_creates_tables(tmp_path):
    """init_db creates alphas, versions, correlations tables."""
    from alphalab.management.db import init_db

    db_path = tmp_path / "alphas.db"
    conn = init_db(db_path)

    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )
    tables = [row[0] for row in cursor.fetchall()]

    assert "alphas" in tables
    assert "versions" in tables
    assert "correlations" in tables
    conn.close()


def test_init_db_idempotent(tmp_path):
    """init_db can be called multiple times safely."""
    from alphalab.management.db import init_db

    db_path = tmp_path / "alphas.db"
    conn1 = init_db(db_path)
    conn1.close()

    conn2 = init_db(db_path)
    cursor = conn2.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    assert len(tables) >= 3
    conn2.close()
