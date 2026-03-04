"""Data models for alpha management."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Alpha:
    """Alpha metadata."""

    id: str
    name: str
    status: str = "draft"
    category: str | None = None
    tags: list[str] = field(default_factory=list)
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass
class Version:
    """Alpha version with expression and metrics."""

    id: str
    alpha_id: str
    version_num: int
    expression: str
    sharpe: float | None = None
    turnover: float | None = None
    fitness: float | None = None
    returns: float | None = None
    drawdown: float | None = None
    pnl_data: list[dict] | None = None
    created_at: datetime | None = None


@dataclass
class Correlation:
    """Correlation check result between two alpha versions."""

    new_alpha_id: str
    exist_alpha_id: str
    corr: float
    new_sharpe: float
    exist_sharpe: float
    improvement: float
    passed: bool
    created_at: datetime | None = None


@dataclass
class CheckResult:
    """Aggregated correlation check result."""

    details: list[Correlation]

    @property
    def passed(self) -> bool:
        """Overall pass: all individual checks pass."""
        return all(c.passed for c in self.details)

    @property
    def blocking(self) -> list[Correlation]:
        """List of correlations that failed the check."""
        return [c for c in self.details if not c.passed]
