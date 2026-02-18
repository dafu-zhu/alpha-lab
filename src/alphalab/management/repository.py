"""Alpha repository for database operations."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from alphalab.management.db import init_db
from alphalab.management.models import Alpha, Correlation, Version


class AlphaRepository:
    """Repository for alpha CRUD operations."""

    def __init__(self, db_path: Path | str):
        """
        Initialize repository.

        Args:
            db_path: Path to SQLite database file
        """
        self._conn = init_db(db_path)
        self._conn.row_factory = sqlite3.Row

    def close(self):
        """Close database connection."""
        self._conn.close()

    def create_alpha(self, alpha: Alpha) -> None:
        """Insert new alpha."""
        self._conn.execute(
            """
            INSERT INTO alphas (id, name, category, tags, status)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                alpha.id,
                alpha.name,
                alpha.category,
                json.dumps(alpha.tags),
                alpha.status,
            ),
        )
        self._conn.commit()

    def get_alpha(self, alpha_id: str) -> Alpha | None:
        """Get alpha by id."""
        row = self._conn.execute(
            "SELECT * FROM alphas WHERE id = ?", (alpha_id,)
        ).fetchone()

        if row is None:
            return None

        return Alpha(
            id=row["id"],
            name=row["name"],
            category=row["category"],
            tags=json.loads(row["tags"]) if row["tags"] else [],
            status=row["status"],
        )

    def create_version(self, version: Version) -> None:
        """Insert new version."""
        self._conn.execute(
            """
            INSERT INTO versions (id, alpha_id, version_num, expression,
                                  sharpe, turnover, fitness, returns, drawdown, pnl_data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                version.id,
                version.alpha_id,
                version.version_num,
                version.expression,
                version.sharpe,
                version.turnover,
                version.fitness,
                version.returns,
                version.drawdown,
                json.dumps(version.pnl_data) if version.pnl_data else None,
            ),
        )
        self._conn.commit()

    def get_version(self, version_id: str) -> Version | None:
        """Get version by id."""
        row = self._conn.execute(
            "SELECT * FROM versions WHERE id = ?", (version_id,)
        ).fetchone()

        if row is None:
            return None

        return Version(
            id=row["id"],
            alpha_id=row["alpha_id"],
            version_num=row["version_num"],
            expression=row["expression"],
            sharpe=row["sharpe"],
            turnover=row["turnover"],
            fitness=row["fitness"],
            returns=row["returns"],
            drawdown=row["drawdown"],
            pnl_data=json.loads(row["pnl_data"]) if row["pnl_data"] else None,
        )

    def get_submitted(self) -> list[Version]:
        """Get all versions of submitted alphas."""
        rows = self._conn.execute(
            """
            SELECT v.* FROM versions v
            JOIN alphas a ON v.alpha_id = a.id
            WHERE a.status = 'submitted'
            ORDER BY v.created_at DESC
            """
        ).fetchall()

        return [
            Version(
                id=r["id"],
                alpha_id=r["alpha_id"],
                version_num=r["version_num"],
                expression=r["expression"],
                sharpe=r["sharpe"],
                pnl_data=json.loads(r["pnl_data"]) if r["pnl_data"] else None,
            )
            for r in rows
        ]

    def submit_alpha(self, alpha_id: str, correlations: list[Correlation]) -> None:
        """Mark alpha as submitted and insert correlations."""
        self._conn.execute(
            "UPDATE alphas SET status = 'submitted', updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (alpha_id,),
        )
        self.insert_correlations(correlations)
        self._conn.commit()

    def insert_correlations(self, correlations: list[Correlation]) -> None:
        """Insert correlation records."""
        for c in correlations:
            self._conn.execute(
                """
                INSERT INTO correlations
                    (new_alpha_id, exist_alpha_id, corr, new_sharpe, exist_sharpe, improvement, passed)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    c.new_alpha_id,
                    c.exist_alpha_id,
                    c.corr,
                    c.new_sharpe,
                    c.exist_sharpe,
                    c.improvement,
                    1 if c.passed else 0,
                ),
            )
        self._conn.commit()

    def get_correlations(self, version_id: str) -> list[Correlation]:
        """Get all correlations involving this version."""
        rows = self._conn.execute(
            """
            SELECT * FROM correlations
            WHERE new_alpha_id = ? OR exist_alpha_id = ?
            """,
            (version_id, version_id),
        ).fetchall()

        return [
            Correlation(
                new_alpha_id=r["new_alpha_id"],
                exist_alpha_id=r["exist_alpha_id"],
                corr=r["corr"],
                new_sharpe=r["new_sharpe"],
                exist_sharpe=r["exist_sharpe"],
                improvement=r["improvement"],
                passed=bool(r["passed"]),
            )
            for r in rows
        ]

    def delete_alpha(self, alpha_id: str) -> None:
        """Delete alpha and cascade to versions and correlations."""
        # Get version ids
        version_ids = [
            r[0]
            for r in self._conn.execute(
                "SELECT id FROM versions WHERE alpha_id = ?", (alpha_id,)
            ).fetchall()
        ]

        # Delete correlations referencing any version
        for vid in version_ids:
            self._conn.execute(
                "DELETE FROM correlations WHERE new_alpha_id = ? OR exist_alpha_id = ?",
                (vid, vid),
            )

        # Delete versions
        self._conn.execute("DELETE FROM versions WHERE alpha_id = ?", (alpha_id,))

        # Delete alpha
        self._conn.execute("DELETE FROM alphas WHERE id = ?", (alpha_id,))

        self._conn.commit()
