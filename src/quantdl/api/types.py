"""Type definitions for QuantDL API."""

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True, slots=True)
class SecurityInfo:
    """Point-in-time security information."""

    security_id: str
    symbol: str
    company: str
    cik: str | None
    cusip: str | None
    start_date: date
    end_date: date | None
    exchange: str | None = None
    sector: str | None = None
    industry: str | None = None
    subindustry: str | None = None

    def __repr__(self) -> str:
        return f"SecurityInfo({self.symbol}, id={self.security_id})"
