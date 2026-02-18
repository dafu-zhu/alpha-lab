# Alpha Correlation Check Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add alpha correlation check that runs before submission, showing which existing alphas block the new one.

**Architecture:** New `management` module with SQLite storage, correlation logic, and repository pattern. API layer uses FastAPI (foundation for local dashboard).

**Tech Stack:** SQLite, Polars (correlation), FastAPI, pytest

---

## Task 1: Database Schema Module

**Files:**
- Create: `src/alphalab/management/__init__.py`
- Create: `src/alphalab/management/db.py`
- Test: `tests/unit/management/__init__.py`
- Test: `tests/unit/management/test_db.py`

**Step 1: Write the failing test**

```python
# tests/unit/management/__init__.py
# (empty)
```

```python
# tests/unit/management/test_db.py
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
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/management/test_db.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'alphalab.management'`

**Step 3: Write minimal implementation**

```python
# src/alphalab/management/__init__.py
"""Alpha management module."""
```

```python
# src/alphalab/management/db.py
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
    conn.executescript(SCHEMA)
    conn.commit()
    return conn
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/unit/management/test_db.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/alphalab/management/ tests/unit/management/
git commit -m "feat(management): add database schema module"
```

---

## Task 2: Data Models

**Files:**
- Create: `src/alphalab/management/models.py`
- Test: `tests/unit/management/test_models.py`

**Step 1: Write the failing test**

```python
# tests/unit/management/test_models.py
"""Tests for data models."""
from datetime import datetime


def test_alpha_model_creation():
    """Alpha model stores required fields."""
    from alphalab.management.models import Alpha

    alpha = Alpha(id="momentum_01", name="Momentum Alpha", status="draft")

    assert alpha.id == "momentum_01"
    assert alpha.name == "Momentum Alpha"
    assert alpha.status == "draft"


def test_version_model_creation():
    """Version model stores expression and metrics."""
    from alphalab.management.models import Version

    version = Version(
        id="momentum_01-v1",
        alpha_id="momentum_01",
        version_num=1,
        expression="rank(-ts_delta(close, 5))",
        sharpe=1.5,
    )

    assert version.id == "momentum_01-v1"
    assert version.alpha_id == "momentum_01"
    assert version.sharpe == 1.5


def test_correlation_model_creation():
    """Correlation model stores check results."""
    from alphalab.management.models import Correlation

    corr = Correlation(
        new_alpha_id="momentum_01-v1",
        exist_alpha_id="mean_rev_02-v1",
        corr=0.82,
        new_sharpe=1.6,
        exist_sharpe=1.5,
        improvement=6.67,
        passed=False,
    )

    assert corr.corr == 0.82
    assert corr.passed is False


def test_check_result_model():
    """CheckResult aggregates correlation checks."""
    from alphalab.management.models import Correlation, CheckResult

    results = [
        Correlation("a-v1", "b-v1", 0.45, 1.6, 1.4, 14.3, True),
        Correlation("a-v1", "c-v1", 0.82, 1.6, 1.5, 6.7, False),
    ]

    check = CheckResult(results)

    assert check.passed is False
    assert len(check.blocking) == 1
    assert check.blocking[0].exist_alpha_id == "c-v1"
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/management/test_models.py -v
```

Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# src/alphalab/management/models.py
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
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/unit/management/test_models.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/alphalab/management/models.py tests/unit/management/test_models.py
git commit -m "feat(management): add data models"
```

---

## Task 3: Correlation Check Logic

**Files:**
- Create: `src/alphalab/management/correlation.py`
- Test: `tests/unit/management/test_correlation.py`

**Step 1: Write the failing test**

```python
# tests/unit/management/test_correlation.py
"""Tests for correlation check logic."""
import polars as pl
import pytest


def test_check_single_alpha_low_corr_passes():
    """corr < 0.7 passes regardless of sharpe improvement."""
    from alphalab.management.correlation import check_correlation
    from alphalab.management.models import Version

    new_pnl = pl.Series([0.01, -0.02, 0.03, 0.01, -0.01])
    new_sharpe = 1.0

    existing = [
        Version(
            id="exist-v1",
            alpha_id="exist",
            version_num=1,
            expression="x",
            sharpe=1.5,
            pnl_data=[
                {"date": "2024-01-01", "ret": 0.05},
                {"date": "2024-01-02", "ret": -0.03},
                {"date": "2024-01-03", "ret": 0.02},
                {"date": "2024-01-04", "ret": -0.01},
                {"date": "2024-01-05", "ret": 0.04},
            ],
        )
    ]

    result = check_correlation(new_pnl, new_sharpe, existing)

    assert result.passed is True
    assert len(result.blocking) == 0


def test_check_high_corr_low_improvement_fails():
    """corr >= 0.7 AND improvement < 10% fails."""
    from alphalab.management.correlation import check_correlation
    from alphalab.management.models import Version

    # Same data = corr = 1.0
    pnl_data = [
        {"date": "2024-01-01", "ret": 0.01},
        {"date": "2024-01-02", "ret": -0.02},
        {"date": "2024-01-03", "ret": 0.03},
    ]
    new_pnl = pl.Series([d["ret"] for d in pnl_data])
    new_sharpe = 1.05  # 5% improvement < 10%

    existing = [
        Version(
            id="exist-v1",
            alpha_id="exist",
            version_num=1,
            expression="x",
            sharpe=1.0,
            pnl_data=pnl_data,
        )
    ]

    result = check_correlation(new_pnl, new_sharpe, existing)

    assert result.passed is False
    assert len(result.blocking) == 1
    assert result.blocking[0].exist_alpha_id == "exist-v1"


def test_check_high_corr_high_improvement_passes():
    """corr >= 0.7 AND improvement >= 10% passes."""
    from alphalab.management.correlation import check_correlation
    from alphalab.management.models import Version

    pnl_data = [
        {"date": "2024-01-01", "ret": 0.01},
        {"date": "2024-01-02", "ret": -0.02},
        {"date": "2024-01-03", "ret": 0.03},
    ]
    new_pnl = pl.Series([d["ret"] for d in pnl_data])
    new_sharpe = 1.2  # 20% improvement >= 10%

    existing = [
        Version(
            id="exist-v1",
            alpha_id="exist",
            version_num=1,
            expression="x",
            sharpe=1.0,
            pnl_data=pnl_data,
        )
    ]

    result = check_correlation(new_pnl, new_sharpe, existing)

    assert result.passed is True


def test_check_no_existing_alphas_passes():
    """No submitted alphas = auto-pass."""
    from alphalab.management.correlation import check_correlation

    new_pnl = pl.Series([0.01, -0.02, 0.03])
    result = check_correlation(new_pnl, 1.5, [])

    assert result.passed is True
    assert len(result.details) == 0


def test_check_multiple_alphas_one_blocks():
    """Multiple existing alphas, one blocks."""
    from alphalab.management.correlation import check_correlation
    from alphalab.management.models import Version

    pnl_data = [
        {"date": "2024-01-01", "ret": 0.01},
        {"date": "2024-01-02", "ret": -0.02},
        {"date": "2024-01-03", "ret": 0.03},
    ]
    new_pnl = pl.Series([d["ret"] for d in pnl_data])
    new_sharpe = 1.05

    existing = [
        Version("a-v1", "a", 1, "x", sharpe=1.0, pnl_data=pnl_data),  # blocks
        Version(
            "b-v1",
            "b",
            1,
            "y",
            sharpe=2.0,
            pnl_data=[
                {"date": "2024-01-01", "ret": 0.05},
                {"date": "2024-01-02", "ret": 0.01},
                {"date": "2024-01-03", "ret": -0.04},
            ],
        ),  # different data, low corr
    ]

    result = check_correlation(new_pnl, new_sharpe, existing)

    assert result.passed is False
    assert len(result.blocking) == 1
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/management/test_correlation.py -v
```

Expected: FAIL

**Step 3: Write minimal implementation**

```python
# src/alphalab/management/correlation.py
"""Correlation check logic."""
from __future__ import annotations

import polars as pl

from alphalab.management.models import CheckResult, Correlation, Version

CORR_THRESHOLD = 0.7
IMPROVEMENT_THRESHOLD = 10.0  # percent


def check_correlation(
    new_pnl: pl.Series,
    new_sharpe: float,
    existing: list[Version],
) -> CheckResult:
    """
    Check new alpha against all submitted alphas.

    Pass criteria (per existing alpha):
        corr < 0.7  OR  (corr >= 0.7 AND improvement >= 10%)

    Overall pass: ALL individual checks pass

    Args:
        new_pnl: Daily returns of the new alpha
        new_sharpe: Sharpe ratio of the new alpha
        existing: List of submitted alpha versions with pnl_data

    Returns:
        CheckResult with passed flag, full details, and blocking alphas
    """
    if not existing:
        return CheckResult(details=[])

    results = []
    for version in existing:
        if not version.pnl_data or version.sharpe is None:
            continue

        exist_pnl = pl.Series([d["ret"] for d in version.pnl_data])

        # Align lengths (use shorter)
        min_len = min(len(new_pnl), len(exist_pnl))
        if min_len == 0:
            continue

        corr = new_pnl[:min_len].pearson_corr(exist_pnl[:min_len])

        if corr is None:
            continue

        improv = (new_sharpe - version.sharpe) / version.sharpe * 100
        passed = corr < CORR_THRESHOLD or improv >= IMPROVEMENT_THRESHOLD

        results.append(
            Correlation(
                new_alpha_id="",  # filled by caller
                exist_alpha_id=version.id,
                corr=corr,
                new_sharpe=new_sharpe,
                exist_sharpe=version.sharpe,
                improvement=improv,
                passed=passed,
            )
        )

    return CheckResult(details=results)
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/unit/management/test_correlation.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/alphalab/management/correlation.py tests/unit/management/test_correlation.py
git commit -m "feat(management): add correlation check logic"
```

---

## Task 4: Alpha Repository

**Files:**
- Create: `src/alphalab/management/repository.py`
- Test: `tests/unit/management/test_repository.py`

**Step 1: Write the failing test**

```python
# tests/unit/management/test_repository.py
"""Tests for alpha repository."""
import json
import pytest


@pytest.fixture
def repo(tmp_path):
    """Create repository with temp database."""
    from alphalab.management.repository import AlphaRepository

    db_path = tmp_path / "alphas.db"
    return AlphaRepository(db_path)


def test_create_alpha(repo):
    """create_alpha inserts new alpha."""
    from alphalab.management.models import Alpha

    alpha = Alpha(id="test_01", name="Test Alpha")
    repo.create_alpha(alpha)

    result = repo.get_alpha("test_01")
    assert result is not None
    assert result.name == "Test Alpha"
    assert result.status == "draft"


def test_create_version(repo):
    """create_version inserts new version."""
    from alphalab.management.models import Alpha, Version

    repo.create_alpha(Alpha(id="test_01", name="Test"))

    version = Version(
        id="test_01-v1",
        alpha_id="test_01",
        version_num=1,
        expression="rank(close)",
        sharpe=1.5,
        pnl_data=[{"date": "2024-01-01", "ret": 0.01}],
    )
    repo.create_version(version)

    result = repo.get_version("test_01-v1")
    assert result is not None
    assert result.sharpe == 1.5
    assert result.pnl_data == [{"date": "2024-01-01", "ret": 0.01}]


def test_get_submitted_versions(repo):
    """get_submitted returns only submitted alphas."""
    from alphalab.management.models import Alpha, Version

    repo.create_alpha(Alpha(id="a", name="A", status="submitted"))
    repo.create_alpha(Alpha(id="b", name="B", status="draft"))

    repo.create_version(Version("a-v1", "a", 1, "x", sharpe=1.0, pnl_data=[]))
    repo.create_version(Version("b-v1", "b", 1, "y", sharpe=1.0, pnl_data=[]))

    submitted = repo.get_submitted()

    assert len(submitted) == 1
    assert submitted[0].id == "a-v1"


def test_submit_alpha(repo):
    """submit_alpha changes status and inserts correlations."""
    from alphalab.management.models import Alpha, Version, Correlation

    repo.create_alpha(Alpha(id="a", name="A"))
    repo.create_version(Version("a-v1", "a", 1, "x", sharpe=1.0, pnl_data=[]))

    correlations = [
        Correlation("a-v1", "exist-v1", 0.5, 1.0, 0.9, 11.1, True),
    ]
    repo.submit_alpha("a", correlations)

    alpha = repo.get_alpha("a")
    assert alpha.status == "submitted"

    corrs = repo.get_correlations("a-v1")
    assert len(corrs) == 1


def test_delete_alpha_cascades(repo):
    """delete_alpha removes alpha, versions, and correlations."""
    from alphalab.management.models import Alpha, Version, Correlation

    repo.create_alpha(Alpha(id="a", name="A", status="submitted"))
    repo.create_version(Version("a-v1", "a", 1, "x", sharpe=1.0, pnl_data=[]))

    repo.create_alpha(Alpha(id="b", name="B", status="submitted"))
    repo.create_version(Version("b-v1", "b", 1, "y", sharpe=1.0, pnl_data=[]))

    # b was submitted after a
    correlations = [Correlation("b-v1", "a-v1", 0.5, 1.0, 0.9, 11.1, True)]
    repo.insert_correlations(correlations)

    # Delete a
    repo.delete_alpha("a")

    assert repo.get_alpha("a") is None
    assert repo.get_version("a-v1") is None
    # Correlation referencing a should be deleted
    assert len(repo.get_correlations("b-v1")) == 0
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/management/test_repository.py -v
```

Expected: FAIL

**Step 3: Write minimal implementation**

```python
# src/alphalab/management/repository.py
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
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/unit/management/test_repository.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/alphalab/management/repository.py tests/unit/management/test_repository.py
git commit -m "feat(management): add alpha repository"
```

---

## Task 5: Management Service

**Files:**
- Create: `src/alphalab/management/service.py`
- Test: `tests/unit/management/test_service.py`

**Step 1: Write the failing test**

```python
# tests/unit/management/test_service.py
"""Tests for management service."""
import pytest
import polars as pl


@pytest.fixture
def service(tmp_path):
    """Create service with temp database."""
    from alphalab.management.service import AlphaService

    db_path = tmp_path / "alphas.db"
    return AlphaService(db_path)


def test_submit_new_alpha_no_existing(service):
    """First alpha submission auto-passes."""
    pnl = pl.Series([0.01, -0.02, 0.03])
    result = service.submit(
        alpha_id="test_01",
        name="Test",
        version_num=1,
        expression="rank(close)",
        sharpe=1.5,
        pnl=pnl,
    )

    assert result.passed is True
    assert len(result.details) == 0


def test_submit_blocked_then_force(service):
    """Blocked submission can be force submitted."""
    pnl_data = [{"date": "2024-01-01", "ret": 0.01}]
    pnl = pl.Series([0.01])

    # First alpha
    service.submit("a", "Alpha A", 1, "x", 1.0, pnl)

    # Second alpha - same PnL, low improvement
    result = service.submit("b", "Alpha B", 1, "y", 1.05, pnl)

    assert result.passed is False

    # Force submit
    service.force_submit("b")

    alpha = service.repo.get_alpha("b")
    assert alpha.status == "submitted"
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/management/test_service.py -v
```

Expected: FAIL

**Step 3: Write minimal implementation**

```python
# src/alphalab/management/service.py
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
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/unit/management/test_service.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/alphalab/management/service.py tests/unit/management/test_service.py
git commit -m "feat(management): add alpha service"
```

---

## Task 6: Export Public API

**Files:**
- Modify: `src/alphalab/management/__init__.py`
- Test: `tests/unit/management/test_init.py`

**Step 1: Write the failing test**

```python
# tests/unit/management/test_init.py
"""Tests for management module exports."""


def test_public_exports():
    """Management module exports key classes."""
    from alphalab.management import (
        Alpha,
        Version,
        Correlation,
        CheckResult,
        AlphaRepository,
        AlphaService,
        check_correlation,
    )

    assert Alpha is not None
    assert AlphaService is not None
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/management/test_init.py -v
```

Expected: FAIL with `ImportError`

**Step 3: Write minimal implementation**

```python
# src/alphalab/management/__init__.py
"""Alpha management module.

Provides alpha storage, correlation checking, and submission workflow.
"""
from alphalab.management.correlation import check_correlation
from alphalab.management.models import Alpha, CheckResult, Correlation, Version
from alphalab.management.repository import AlphaRepository
from alphalab.management.service import AlphaService

__all__ = [
    "Alpha",
    "Version",
    "Correlation",
    "CheckResult",
    "AlphaRepository",
    "AlphaService",
    "check_correlation",
]
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/unit/management/test_init.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/alphalab/management/__init__.py tests/unit/management/test_init.py
git commit -m "feat(management): export public API"
```

---

## Task 7: Run Full Test Suite

**Step 1: Run all management tests**

```bash
uv run pytest tests/unit/management/ -v --tb=short
```

Expected: All tests PASS

**Step 2: Run full project tests**

```bash
uv run pytest --tb=short
```

Expected: No regressions

**Step 3: Commit any fixes if needed**

---

## Summary

| Task | Files | Purpose |
|------|-------|---------|
| 1 | db.py | SQLite schema |
| 2 | models.py | Data classes |
| 3 | correlation.py | Check logic |
| 4 | repository.py | DB operations |
| 5 | service.py | Business logic |
| 6 | __init__.py | Public exports |
| 7 | - | Integration test |

**Next phase:** API endpoints (FastAPI) and UI components will be added when building the local dashboard (Phase 4 of #45).
