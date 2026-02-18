"""Database schema and connection management."""
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS alphas (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT,
    tags TEXT,
    status TEXT DEFAULT 'draft',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS versions (
    id TEXT PRIMARY KEY,
    alpha_id TEXT NOT NULL REFERENCES alphas(id),
    version_num INTEGER NOT NULL,
    expression TEXT NOT NULL,
    sharpe REAL,
    turnover REAL,
    fitness REAL,
    returns REAL,
    drawdown REAL,
    pnl_data TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(alpha_id, version_num)
);

CREATE TABLE IF NOT EXISTS correlations (
    new_alpha_id TEXT NOT NULL,
    exist_alpha_id TEXT NOT NULL,
    corr REAL NOT NULL,
    new_sharpe REAL NOT NULL,
    exist_sharpe REAL NOT NULL,
    improvement REAL NOT NULL,
    passed INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (new_alpha_id, exist_alpha_id)
);

CREATE INDEX IF NOT EXISTS idx_correlations_new ON correlations(new_alpha_id);
CREATE INDEX IF NOT EXISTS idx_correlations_exist ON correlations(exist_alpha_id);
CREATE INDEX IF NOT EXISTS idx_versions_alpha ON versions(alpha_id);
"""


def init_db(db_path: Path | str) -> sqlite3.Connection:
    """
    Initialize database with schema.

    Args:
        db_path: Path to SQLite database file

    Returns:
        Open connection to database
    """
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    conn.commit()
    return conn
