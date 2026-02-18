"""Alpha management service."""
from __future__ import annotations

from pathlib import Path

import polars as pl

from alphalab.management.correlation import check_correlation
from alphalab.management.models import Alpha, CheckResult, Correlation, Version
from alphalab.management.repository import AlphaRepository


class AlphaService:
    """Service layer for alpha management."""

    def __init__(self, db_path: Path | str):
        """
        Initialize service.

        Args:
            db_path: Path to SQLite database file
        """
        self.repo = AlphaRepository(db_path)
        self._pending: dict[str, tuple[Version, list[Correlation]]] = {}

    def submit(
        self,
        alpha_id: str,
        name: str,
        version_num: int,
        expression: str,
        sharpe: float,
        pnl: pl.Series,
    ) -> CheckResult:
        """
        Attempt to submit alpha with correlation check.

        Args:
            alpha_id: Alpha identifier
            name: Alpha name
            version_num: Version number
            expression: Alpha expression
            sharpe: Sharpe ratio
            pnl: Daily returns series

        Returns:
            CheckResult indicating pass/fail and blocking alphas
        """
        version_id = f"{alpha_id}-v{version_num}"
        pnl_data = [{"date": str(i), "ret": float(v)} for i, v in enumerate(pnl)]

        # Create alpha if not exists
        if self.repo.get_alpha(alpha_id) is None:
            self.repo.create_alpha(Alpha(id=alpha_id, name=name))

        # Create version
        version = Version(
            id=version_id,
            alpha_id=alpha_id,
            version_num=version_num,
            expression=expression,
            sharpe=sharpe,
            pnl_data=pnl_data,
        )
        self.repo.create_version(version)

        # Check correlation
        existing = self.repo.get_submitted()
        result = check_correlation(pnl, sharpe, existing)

        # Fill in new_alpha_id
        correlations = [
            Correlation(
                new_alpha_id=version_id,
                exist_alpha_id=c.exist_alpha_id,
                corr=c.corr,
                new_sharpe=c.new_sharpe,
                exist_sharpe=c.exist_sharpe,
                improvement=c.improvement,
                passed=c.passed,
            )
            for c in result.details
        ]

        if result.passed:
            self.repo.submit_alpha(alpha_id, correlations)
        else:
            # Store for potential force submit
            self._pending[alpha_id] = (version, correlations)

        return result

    def force_submit(self, alpha_id: str) -> None:
        """Force submit a blocked alpha."""
        if alpha_id not in self._pending:
            raise ValueError(f"No pending submission for {alpha_id}")

        _, correlations = self._pending.pop(alpha_id)
        self.repo.submit_alpha(alpha_id, correlations)

    def get_correlations(self, version_id: str) -> list[Correlation]:
        """Get correlations for a version."""
        return self.repo.get_correlations(version_id)

    def delete(self, alpha_id: str) -> None:
        """Delete alpha and cascade."""
        self.repo.delete_alpha(alpha_id)
