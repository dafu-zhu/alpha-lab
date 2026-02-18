# AlphaLab Web Components Design

**Date**: 2025-02-17
**Status**: Approved
**Issue**: #45

## Overview

Two web components for AlphaLab:

1. **Documentation Site** — Static docs hosted on GitHub Pages
2. **Local Dashboard** — WQ BRAIN-like UI via `alab serve`

## Goals

- **Speed**: 2 min (WQ BRAIN) → 2 sec (local)
- **Transparency**: See intermediate values, not black box
- **Version management**: Parameter change = version, structure change = new alpha
- **Flexibility**: Tool empowers YOUR workflow, not prescriptive

## Non-Goals

- Not a hosted web service (computation stays local)
- Not an auto-alpha-miner
- Not a WQ BRAIN replica (complementary, not competing)

---

## 1. Documentation Site

### Tech Stack

- MkDocs Material
- GitHub Pages (auto-deploy via Actions)

### Structure

```
/
├── Getting Started
│   ├── Installation
│   ├── Quick Start
│   └── Data Setup
├── API Reference
│   ├── AlphaLabClient
│   ├── dsl.compute()
│   └── Operators (68)
├── Expression Guide
│   ├── Syntax
│   ├── Multi-line expressions
│   └── Examples
├── Data Fields (66)
├── Alpha Library
│   ├── Alpha101 formulas
│   └── Common patterns
└── CLI Reference
```

---

## 2. Local Dashboard

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Browser (localhost:8888)                               │
│  - Expression editor                                    │
│  - Metrics display                                      │
│  - PnL chart                                            │
│  - Alpha management                                     │
└─────────────────────┬───────────────────────────────────┘
                      │ HTTP/WebSocket
                      ▼
┌─────────────────────────────────────────────────────────┐
│  alab serve (FastAPI + Uvicorn)                         │
│  - /api/evaluate                                        │
│  - /api/alphas                                          │
│  - /api/versions                                        │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│  Local Storage                                          │
│  $LOCAL_STORAGE_PATH/                                   │
│  ├── data/           (market data)                      │
│  └── alphas/         (alpha management)                 │
│      ├── alphas.db   (SQLite)                           │
│      └── exports/    (optional)                         │
└─────────────────────────────────────────────────────────┘
```

### UI Layout

```
┌─────────────────────────────────────────────────────────────────┐
│  AlphaLab Dashboard                              localhost:8888 │
├─────────────────────────────────┬───────────────────────────────┤
│  Code                           │  Performance Summary          │
│  ┌───────────────────────────┐  │  Sharpe | Turnover | Fitness  │
│  │ Expression editor         │  │  1.60  | 13.8%    | 1.28     │
│  │ with syntax highlighting  │  │                               │
│  └───────────────────────────┘  │  Yearly Breakdown Table       │
│                                 │                               │
│  Settings (non-performance)     ├───────────────────────────────┤
│  - Date range                   │  PnL Chart                    │
│  - Universe                     │                               │
│                                 │                               │
├─────────────────────────────────┼───────────────────────────────┤
│  Alpha Properties               │  Correlation                  │
│  - Name, Category, Tags         │  Self-correlation display     │
│  - Version selector             │                               │
└─────────────────────────────────┴───────────────────────────────┘
```

### Design Principle: Settings in Expression

Performance-affecting settings belong IN the expression:

```python
# Neutralization
alpha = group_neutralize(raw_alpha, subindustry)

# Decay
alpha = decay_linear(alpha, 5)

# Truncation
alpha = truncate(alpha, 0.05)
```

UI settings are for non-performance config only:
- Date range for backtest
- Universe selection
- Display preferences

This keeps expressions **self-contained** — copy to WQ BRAIN and behavior is identical.

### Version Management

| Change Type | Result |
|-------------|--------|
| Parameter change (`ts_delta(close, 5)` → `ts_delta(close, 10)`) | New version of same alpha |
| Structure change (add/remove terms) | New alpha |

Stored in SQLite with full history.

### Storage Schema (alphas.db)

```sql
CREATE TABLE alphas (
    id TEXT PRIMARY KEY,
    name TEXT,
    category TEXT,
    tags TEXT,  -- JSON array
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);

CREATE TABLE versions (
    id TEXT PRIMARY KEY,
    alpha_id TEXT REFERENCES alphas(id),
    version_num INTEGER,
    expression TEXT,
    created_at TIMESTAMP,
    -- Cached metrics
    sharpe REAL,
    turnover REAL,
    fitness REAL,
    returns REAL,
    drawdown REAL
);

CREATE TABLE runs (
    id TEXT PRIMARY KEY,
    version_id TEXT REFERENCES versions(id),
    universe TEXT,
    start_date DATE,
    end_date DATE,
    run_at TIMESTAMP,
    metrics TEXT,  -- JSON blob
    pnl_data TEXT  -- JSON blob for chart
);
```

### Tech Stack

- **Backend**: FastAPI + Uvicorn
- **Frontend**: React or Svelte (TBD)
- **Charts**: Plotly or Lightweight Charts
- **Storage**: SQLite

---

## Open Questions

1. Frontend framework choice (React vs Svelte)?
2. Should Alpha101 formulas be bundled or fetched?
3. Custom domain for docs (alphalab.dev)?

---

## References

- [OpenBB Architecture](https://openbb.co/blog/exploring-the-architecture-behind-the-openbb-platform)
- [WorldQuant BRAIN](https://platform.worldquantbrain.com/) (UI inspiration)
- Issue #44 (parent roadmap)
- Issue #45 (this design)
