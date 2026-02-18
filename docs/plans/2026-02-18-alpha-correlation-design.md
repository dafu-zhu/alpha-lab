# Alpha Correlation Check Design

**Date**: 2026-02-18
**Status**: Approved
**Issue**: #45 (extends local dashboard)

## Overview

Alpha correlation check before submission — inspired by WorldQuant BRAIN. Prevents submitting alphas too similar to existing ones unless they show meaningful improvement.

## Goals

- Check new alpha correlation against all submitted alphas
- Pass criteria: `corr < 0.7` OR `(corr >= 0.7 AND sharpe improvement >= 10%)`
- Show which alphas block submission (WQ BRAIN pain point solved)
- Store full correlation records for analysis

## Non-Goals

- Auto-reject (warning with override instead)
- Cross-user correlation (local-only, single user)

---

## 1. Storage Schema

### New table: `correlations`

```sql
CREATE TABLE correlations (
    new_alpha_id TEXT NOT NULL,      -- "momentum_01-v3"
    exist_alpha_id TEXT NOT NULL,    -- "mean_rev_02-v1"
    corr REAL NOT NULL,
    new_sharpe REAL NOT NULL,
    exist_sharpe REAL NOT NULL,
    improvement REAL NOT NULL,       -- percentage
    passed BOOLEAN NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (new_alpha_id, exist_alpha_id)
);

CREATE INDEX idx_correlations_new ON correlations(new_alpha_id);
CREATE INDEX idx_correlations_exist ON correlations(exist_alpha_id);
```

### Modified tables

```sql
-- alphas table: add status
ALTER TABLE alphas ADD COLUMN status TEXT DEFAULT 'draft';
-- status: 'draft' | 'submitted'

-- versions table: add PnL storage
ALTER TABLE versions ADD COLUMN pnl_data TEXT;
-- JSON array: [{"date": "2024-01-02", "ret": 0.012}, ...]
```

### Alpha identifier format

- **Alpha ID**: Same across parameter changes (structural identity)
- **Version**: Increments on parameter changes
- **Real identifier**: `{alpha_id}-v{version}` (e.g., `momentum_01-v3`)

---

## 2. Correlation Check Logic

```python
def check_correlation(new_pnl: Series, new_sharpe: float) -> CheckResult:
    """
    Check new alpha against all submitted alphas.

    Pass criteria (per existing alpha):
        corr < 0.7  OR  (corr >= 0.7 AND improvement >= 10%)

    Overall pass: ALL individual checks pass

    Args:
        new_pnl: Daily returns of the new alpha
        new_sharpe: Sharpe ratio of the new alpha

    Returns:
        CheckResult with passed flag, full details, and blocking alphas
    """
    submitted = get_submitted()

    results = []
    for alpha in submitted:
        pnl = load_pnl(alpha.id)
        corr = new_pnl.corr(pnl)
        improv = (new_sharpe - alpha.sharpe) / alpha.sharpe * 100
        passed = corr < 0.7 or improv >= 10

        results.append({
            "exist_id": alpha.id,
            "corr": corr,
            "new_sharpe": new_sharpe,
            "exist_sharpe": alpha.sharpe,
            "improv": improv,
            "passed": passed
        })

    blocking = [r for r in results if not r["passed"]]
    return CheckResult(passed=len(blocking) == 0, details=results, blocking=blocking)
```

### Edge cases

- No submitted alphas → auto-pass
- corr=1.0 (identical) → must have 10%+ sharpe improvement
- NaN corr (no overlapping dates) → pass

---

## 3. API Endpoints

### Submit with correlation check

```
POST /api/alphas/{alpha_id}/submit
```

Runs correlation check, returns results. If passed, sets status='submitted' and inserts correlations.

**Response:**

```json
{
  "passed": false,
  "blocking": [
    {"exist_id": "mean_rev_02-v1", "corr": 0.82, "improv": 6.7}
  ],
  "details": [
    {"exist_id": "momentum_01-v2", "corr": 0.45, "improv": 12.3, "passed": true},
    {"exist_id": "mean_rev_02-v1", "corr": 0.82, "improv": 6.7, "passed": false}
  ]
}
```

### Force submit (override warning)

```
POST /api/alphas/{alpha_id}/submit/confirm
```

Submits despite warnings. Inserts correlations, sets status='submitted'.

### Get correlations

```
GET /api/alphas/{alpha_id}/correlations
```

Returns all correlation records where alpha appears as new_alpha_id or exist_alpha_id.

### Delete alpha (cascade)

```
DELETE /api/alphas/{alpha_id}
```

Deletes alpha, versions, and correlations:

```sql
DELETE FROM correlations
WHERE new_alpha_id LIKE '{alpha_id}-%' OR exist_alpha_id LIKE '{alpha_id}-%'
```

---

## 4. UI Components

### Submit flow modal

```
┌─────────────────────────────────────────────────────────────┐
│  Correlation Check Results                                  │
│                                                             │
│  ✓ momentum_01-v2    corr: 0.45    improv: +12.3%          │
│  ✗ mean_rev_02-v1    corr: 0.82    improv: +6.7%   BLOCKED │
│  ✓ value_03-v4       corr: 0.31    improv: -2.1%           │
│                                                             │
│  ⚠ Blocked by 1 alpha. Submit anyway?                      │
│                                                             │
│  [Cancel]                            [Submit Anyway]        │
└─────────────────────────────────────────────────────────────┘
```

### Alpha detail view — correlations tab

```
┌─────────────────────────────────────────────────────────────┐
│  Alpha: momentum_01-v3                                      │
│  ─────────────────────────────────────────────────────────  │
│  [Overview]  [Versions]  [Correlations]                     │
│                                                             │
│  Correlations at submission:                                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Alpha          │ Corr  │ Sharpe │ Improv │ Status  │   │
│  │ mean_rev_02-v1 │ 0.82  │ 1.50   │ +6.7%  │ Blocked │   │
│  │ value_03-v4    │ 0.31  │ 1.80   │ -2.1%  │ Passed  │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. PnL Storage

Daily returns stored as JSON in `versions.pnl_data`:

```json
[
  {"date": "2024-01-02", "ret": 0.012},
  {"date": "2024-01-03", "ret": -0.005},
  ...
]
```

**Storage estimate**: ~25KB per alpha version (250 days × 5 years × 20 bytes)

---

## References

- WorldQuant BRAIN correlation check (inspiration)
- Issue #44 (Alpha DSL roadmap)
- Issue #45 (Local dashboard design)
