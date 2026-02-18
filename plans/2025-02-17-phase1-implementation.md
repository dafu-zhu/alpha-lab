# Phase 1: Examples & Documentation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create example notebooks and scripts, migrate docs to Quarto, deploy to GitHub Pages.

**Architecture:** Quarto renders both .qmd docs and .ipynb notebooks into a unified static site. Notebooks demonstrate API usage and validate functionality. Scripts provide copy-paste patterns.

**Tech Stack:** Quarto, Jupyter notebooks, GitHub Actions, GitHub Pages

---

## Task 1: Set Up Quarto Project

**Files:**
- Create: `docs/_quarto.yml`
- Create: `docs/index.qmd`
- Create: `.github/workflows/docs.yml`

**Step 1: Create Quarto configuration**

Create `docs/_quarto.yml`:

```yaml
project:
  type: website
  output-dir: _site

website:
  title: "AlphaLab"
  description: "Local-first alpha research platform"
  navbar:
    left:
      - text: "Getting Started"
        href: getting-started/installation.qmd
      - text: "Tutorials"
        href: tutorials/01_quickstart.ipynb
      - text: "API Reference"
        href: api/client.qmd
      - text: "Operators"
        href: operators.qmd
      - text: "Fields"
        href: fields.qmd
    right:
      - icon: github
        href: https://github.com/dafu-zhu/alpha-lab
  sidebar:
    - title: "Getting Started"
      contents:
        - getting-started/installation.qmd
        - getting-started/quickstart.qmd
        - getting-started/data-setup.qmd
    - title: "Tutorials"
      contents:
        - tutorials/01_quickstart.ipynb
        - tutorials/02_expressions.ipynb
        - tutorials/03_group_operations.ipynb
        - tutorials/04_standalone_dsl.ipynb
    - title: "API Reference"
      contents:
        - api/client.qmd
        - api/dsl.qmd
        - api/operators.qmd
    - title: "Reference"
      contents:
        - operators.qmd
        - fields.qmd
        - cli.qmd

format:
  html:
    theme: cosmo
    toc: true
    code-copy: true
    code-overflow: wrap

execute:
  freeze: auto
```

**Step 2: Create landing page**

Create `docs/index.qmd`:

```markdown
---
title: "AlphaLab"
subtitle: "Local-first alpha research platform"
---

A local-first alpha research platform that brings [WorldQuant BRAIN](https://platform.worldquantbrain.com/)-style development to your own machine — same expression syntax, same operators, no latency, no transfer costs, no platform limits.

## Quick Example

```python
from alphalab.api.client import AlphaLabClient

client = AlphaLabClient(data_path="/path/to/your/data")

alpha = client.query("""
regime = ts_rank(ts_sum(volume, 5)/ts_sum(volume, 60), 60) > 0.5;
fundamental = group_rank(ts_delta(income/sharesout, 21), subindustry);
timing = group_rank(-ts_delta(vwap, 5), subindustry);
trade_when(regime, 2*fundamental + timing, -1)
""")
```

## Features

- **68 operators** — time-series, cross-sectional, group operations
- **66 data fields** — price/volume, fundamentals, GICS classifications
- **Zero latency** — data lives on disk, expressions run in-process
- **Zero cost** — free data sources (Alpaca, SEC EDGAR)

## Get Started

1. [Installation](getting-started/installation.qmd)
2. [Quick Start](getting-started/quickstart.qmd)
3. [Tutorials](tutorials/01_quickstart.ipynb)
```

**Step 3: Verify Quarto installation**

Run: `quarto --version`
Expected: Version number (e.g., 1.4.x or higher)

If not installed: `brew install quarto` (macOS) or download from quarto.org

**Step 4: Test render**

Run: `cd /Users/zdf/Documents/GitHub/alpha-lab && quarto render docs --to html`
Expected: Creates `docs/_site/` directory

**Step 5: Commit**

```bash
git add docs/_quarto.yml docs/index.qmd
git commit -m "feat: initialize Quarto documentation"
```

---

## Task 2: Migrate Existing Docs to Quarto

**Files:**
- Create: `docs/getting-started/installation.qmd`
- Create: `docs/getting-started/quickstart.qmd`
- Create: `docs/getting-started/data-setup.qmd`
- Create: `docs/api/client.qmd`
- Create: `docs/api/dsl.qmd`
- Create: `docs/api/operators.qmd`
- Rename: `docs/OPERATORS.md` → `docs/operators.qmd`
- Rename: `docs/FIELDS.md` → `docs/fields.qmd`
- Rename: `docs/CLI.md` → `docs/cli.qmd`

**Step 1: Create directory structure**

```bash
mkdir -p docs/getting-started docs/api docs/tutorials
```

**Step 2: Create installation.qmd**

Create `docs/getting-started/installation.qmd`:

```markdown
---
title: "Installation"
---

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- Free [Alpaca](https://app.alpaca.markets/signup) account
- Email address (for SEC EDGAR User-Agent)

## Install

```bash
git clone https://github.com/dafu-zhu/alpha-lab.git
cd alpha-lab
uv sync
cp .env.example .env   # then fill in credentials
```

## Environment Variables

Copy `.env.example` to `.env` and fill in:

| Variable | Description |
|----------|-------------|
| `ALPACA_API_KEY` | Alpaca API key |
| `ALPACA_API_SECRET` | Alpaca API secret |
| `SEC_USER_AGENT` | Your email (required by SEC) |
| `LOCAL_STORAGE_PATH` | Path to store data |
```

**Step 3: Create quickstart.qmd**

Create `docs/getting-started/quickstart.qmd`:

```markdown
---
title: "Quick Start"
---

## Download Data

```bash
uv run alab --master                         # Build security master + calendar
uv run alab --all --start 2017 --end 2025    # Download everything (~45 min)
```

Or pick what you need: `--ticks`, `--fundamental`, `--top-3000`, `--features`.

## Your First Alpha

```python
from alphalab.api.client import AlphaLabClient

client = AlphaLabClient(data_path="/path/to/your/data")

# Simple momentum alpha
alpha = client.query("rank(-ts_delta(close, 5))")
print(alpha.head())
```

## Next Steps

- [Tutorials](../tutorials/01_quickstart.ipynb) — Step-by-step notebooks
- [API Reference](../api/client.qmd) — Full API documentation
- [Operators](../operators.qmd) — All 68 operators
```

**Step 4: Create data-setup.qmd**

Create `docs/getting-started/data-setup.qmd`:

```markdown
---
title: "Data Setup"
---

## Data Sources

| Source | Data | Coverage |
|--------|------|----------|
| Alpaca | Daily OHLCV | 2017+ |
| SEC EDGAR | Fundamentals | 2009+ |
| Security Master | Symbols, GICS | 1986+ |

## Download Commands

```bash
# Full backfill (recommended first time)
uv run alab --all --start 2017 --end 2025

# Individual data types
uv run alab --ticks          # Daily prices
uv run alab --fundamental    # SEC filings
uv run alab --top-3000       # Universe lists
uv run alab --features       # Pre-built matrices
```

## Storage Layout

```
$LOCAL_STORAGE_PATH/
├── data/
│   ├── meta/master/          # Security master, calendar
│   ├── meta/universe/        # Monthly top 3000
│   ├── raw/ticks/            # Daily OHLCV per security
│   ├── raw/fundamental/      # SEC filings per security
│   └── features/             # Wide matrices (Date x securities)
```
```

**Step 5: Create API docs from existing content**

Create `docs/api/client.qmd` by extracting from `docs/API.md` (AlphaLabClient section).

Create `docs/api/dsl.qmd` by extracting from `docs/API.md` (Standalone DSL section).

Create `docs/api/operators.qmd` as a summary linking to full operators.qmd.

**Step 6: Convert existing .md to .qmd**

```bash
cd /Users/zdf/Documents/GitHub/alpha-lab/docs
mv OPERATORS.md operators.qmd
mv FIELDS.md fields.qmd
mv CLI.md cli.qmd
```

Add YAML frontmatter to each:
```yaml
---
title: "Operators"
---
```

**Step 7: Test full render**

Run: `quarto render docs`
Expected: All pages render without errors

**Step 8: Commit**

```bash
git add docs/
git commit -m "docs: migrate existing docs to Quarto format"
```

---

## Task 3: Create Quickstart Notebook

**Files:**
- Create: `docs/tutorials/01_quickstart.ipynb`

**Step 1: Create notebook**

Create `docs/tutorials/01_quickstart.ipynb` with cells:

**Cell 1 (markdown):**
```markdown
# AlphaLab Quickstart

This notebook demonstrates basic AlphaLab usage.
```

**Cell 2 (code):**
```python
from alphalab.api.client import AlphaLabClient
import os

# Initialize client
data_path = os.environ.get("LOCAL_STORAGE_PATH", "/path/to/your/data")
client = AlphaLabClient(data_path=data_path)
```

**Cell 3 (markdown):**
```markdown
## Look Up Securities

Use `client.lookup()` to resolve symbols to security info.
```

**Cell 4 (code):**
```python
# Look up Apple
info = client.lookup("AAPL")
print(f"Symbol: {info.symbol}")
print(f"Company: {info.company}")
print(f"Sector: {info.sector}")
```

**Cell 5 (markdown):**
```markdown
## Load Data

Use `client.get()` to load pre-built feature tables.
```

**Cell 6 (code):**
```python
# Load closing prices for specific symbols
close = client.get("close", symbols=["AAPL", "MSFT", "GOOGL"], start="2024-01-01")
close.head()
```

**Cell 7 (markdown):**
```markdown
## Your First Alpha

Use `client.query()` to evaluate alpha expressions.
```

**Cell 8 (code):**
```python
# Simple momentum alpha: rank of 5-day price change (negated for reversal)
alpha = client.query(
    "rank(-ts_delta(close, 5))",
    symbols=["AAPL", "MSFT", "GOOGL"],
    start="2024-01-01"
)
alpha.head()
```

**Cell 9 (markdown):**
```markdown
## Next Steps

- [Multi-line Expressions](02_expressions.ipynb)
- [Group Operations](03_group_operations.ipynb)
- [Standalone DSL](04_standalone_dsl.ipynb)
```

**Step 2: Verify notebook renders**

Run: `quarto render docs/tutorials/01_quickstart.ipynb`
Expected: Creates HTML output without execution errors

**Step 3: Commit**

```bash
git add docs/tutorials/01_quickstart.ipynb
git commit -m "feat: add quickstart tutorial notebook"
```

---

## Task 4: Create Expressions Notebook

**Files:**
- Create: `docs/tutorials/02_expressions.ipynb`

**Step 1: Create notebook**

Create `docs/tutorials/02_expressions.ipynb` with cells:

**Cell 1 (markdown):**
```markdown
# Alpha Expressions

Learn to write multi-line expressions with variables.
```

**Cell 2 (code):**
```python
from alphalab.api.client import AlphaLabClient
import os

client = AlphaLabClient(data_path=os.environ.get("LOCAL_STORAGE_PATH"))
```

**Cell 3 (markdown):**
```markdown
## Single-Line Expressions

Simple expressions that fit on one line.
```

**Cell 4 (code):**
```python
# Daily returns
returns = client.query("close / ts_delay(close, 1) - 1", start="2024-01-01")
returns.head()
```

**Cell 5 (code):**
```python
# Z-score from 20-day moving average
zscore = client.query(
    "(close - ts_mean(close, 20)) / ts_std(close, 20)",
    start="2024-01-01"
)
zscore.head()
```

**Cell 6 (markdown):**
```markdown
## Multi-Line with Variables

Use semicolons to separate statements. The last expression is returned.
```

**Cell 7 (code):**
```python
alpha = client.query("""
momentum = ts_delta(close, 10);
volatility = ts_std(returns, 20);
rank(momentum / volatility)
""", start="2024-01-01")

alpha.head()
```

**Cell 8 (markdown):**
```markdown
## Boolean Conditions

Create conditional alphas with comparisons.
```

**Cell 9 (code):**
```python
# Only trade when volume is above average
alpha = client.query("""
volume_filter = volume > ts_mean(volume, 20);
raw_alpha = rank(-ts_delta(close, 5));
volume_filter * raw_alpha
""", start="2024-01-01")

alpha.head()
```

**Cell 10 (markdown):**
```markdown
## trade_when

Conditional entry/exit with carry-forward.
```

**Cell 11 (code):**
```python
alpha = client.query("""
regime = ts_rank(ts_sum(volume, 5)/ts_sum(volume, 60), 60) > 0.5;
signal = rank(-ts_delta(close, 5));
trade_when(regime, signal, -1)
""", start="2024-01-01")

alpha.head()
```

**Step 2: Verify notebook renders**

Run: `quarto render docs/tutorials/02_expressions.ipynb`

**Step 3: Commit**

```bash
git add docs/tutorials/02_expressions.ipynb
git commit -m "feat: add expressions tutorial notebook"
```

---

## Task 5: Create Group Operations Notebook

**Files:**
- Create: `docs/tutorials/03_group_operations.ipynb`

**Step 1: Create notebook**

Create `docs/tutorials/03_group_operations.ipynb` with cells:

**Cell 1 (markdown):**
```markdown
# Group Operations

Sector-relative operations: group_rank, group_neutralize, and more.
```

**Cell 2 (code):**
```python
from alphalab.api.client import AlphaLabClient
import os

client = AlphaLabClient(data_path=os.environ.get("LOCAL_STORAGE_PATH"))
```

**Cell 3 (markdown):**
```markdown
## Group Fields

Available group fields for within-group operations:
- `sector` — GICS sector
- `industry` — GICS industry
- `subindustry` — GICS sub-industry
- `exchange` — Exchange code
```

**Cell 4 (markdown):**
```markdown
## group_rank

Rank within each group instead of across all securities.
```

**Cell 5 (code):**
```python
# Rank momentum within each sector
alpha = client.query(
    "group_rank(ts_delta(close, 5), sector)",
    start="2024-01-01"
)
alpha.head()
```

**Cell 6 (markdown):**
```markdown
## group_neutralize

Subtract group mean to remove sector bias.
```

**Cell 7 (code):**
```python
# Sector-neutralized momentum
alpha = client.query("""
raw = rank(-ts_delta(close, 5));
group_neutralize(raw, sector)
""", start="2024-01-01")

alpha.head()
```

**Cell 8 (markdown):**
```markdown
## Combining Group Operations

Build sophisticated alphas with multiple group operations.
```

**Cell 9 (code):**
```python
# Industry-relative value factor, sector-neutralized
alpha = client.query("""
value = group_rank(book / close, industry);
group_neutralize(value, sector)
""", start="2024-01-01")

alpha.head()
```

**Step 2: Verify notebook renders**

Run: `quarto render docs/tutorials/03_group_operations.ipynb`

**Step 3: Commit**

```bash
git add docs/tutorials/03_group_operations.ipynb
git commit -m "feat: add group operations tutorial notebook"
```

---

## Task 6: Create Standalone DSL Notebook

**Files:**
- Create: `docs/tutorials/04_standalone_dsl.ipynb`

**Step 1: Create notebook**

Create `docs/tutorials/04_standalone_dsl.ipynb` with cells:

**Cell 1 (markdown):**
```markdown
# Standalone DSL

Use `dsl.compute()` with your own DataFrames — no AlphaLab data required.
```

**Cell 2 (code):**
```python
from alphalab.api.dsl import compute
import polars as pl
import numpy as np
```

**Cell 3 (markdown):**
```markdown
## Create Sample Data

Wide format: Date column + one column per security.
```

**Cell 4 (code):**
```python
# Create sample price data
dates = pl.date_range(pl.date(2024, 1, 1), pl.date(2024, 1, 31), eager=True)
np.random.seed(42)

prices = pl.DataFrame({
    "Date": dates,
    "AAPL": 180 + np.cumsum(np.random.randn(len(dates))),
    "MSFT": 380 + np.cumsum(np.random.randn(len(dates))),
    "GOOGL": 140 + np.cumsum(np.random.randn(len(dates))),
})
prices.head()
```

**Cell 5 (markdown):**
```markdown
## Single Variable

Pass your DataFrame as a named variable.
```

**Cell 6 (code):**
```python
# Rank of 5-day momentum
result = compute("rank(-ts_delta(x, 5))", x=prices)
result.head()
```

**Cell 7 (markdown):**
```markdown
## Multiple Variables

Pass multiple DataFrames for complex expressions.
```

**Cell 8 (code):**
```python
# Create volume data
volume = pl.DataFrame({
    "Date": dates,
    "AAPL": np.random.randint(1000000, 5000000, len(dates)),
    "MSFT": np.random.randint(2000000, 8000000, len(dates)),
    "GOOGL": np.random.randint(500000, 2000000, len(dates)),
})

# Volume-weighted momentum
result = compute(
    "rank(-ts_delta(price, 5)) * rank(vol)",
    price=prices,
    vol=volume
)
result.head()
```

**Cell 9 (markdown):**
```markdown
## Multi-Line Expressions

Same syntax as client.query().
```

**Cell 10 (code):**
```python
result = compute("""
momentum = ts_delta(price, 5);
volatility = ts_std(price, 10);
rank(momentum / volatility)
""", price=prices)

result.head()
```

**Cell 11 (markdown):**
```markdown
## When to Use

| Use Case | Function |
|----------|----------|
| AlphaLab data with auto-loading | `client.query()` |
| Custom DataFrames | `dsl.compute()` |
```

**Step 2: Verify notebook renders**

Run: `quarto render docs/tutorials/04_standalone_dsl.ipynb`

**Step 3: Commit**

```bash
git add docs/tutorials/04_standalone_dsl.ipynb
git commit -m "feat: add standalone DSL tutorial notebook"
```

---

## Task 7: Create Example Scripts

**Files:**
- Create: `examples/scripts/momentum.py`
- Create: `examples/scripts/mean_reversion.py`
- Create: `examples/scripts/quality_factor.py`
- Create: `examples/scripts/combined_alpha.py`

**Step 1: Create directory**

```bash
mkdir -p examples/scripts
```

**Step 2: Create momentum.py**

```python
"""Simple momentum alpha."""
from alphalab.api.client import AlphaLabClient
import os

def momentum_alpha(client, lookback=5):
    """5-day price momentum, cross-sectionally ranked."""
    return client.query(f"rank(-ts_delta(close, {lookback}))")

if __name__ == "__main__":
    client = AlphaLabClient(data_path=os.environ["LOCAL_STORAGE_PATH"])
    alpha = momentum_alpha(client)
    print(alpha.tail(10))
```

**Step 3: Create mean_reversion.py**

```python
"""Mean reversion alpha."""
from alphalab.api.client import AlphaLabClient
import os

def mean_reversion_alpha(client, lookback=20):
    """Buy oversold, sell overbought (z-score based)."""
    return client.query(f"rank(-ts_zscore(close, {lookback}))")

if __name__ == "__main__":
    client = AlphaLabClient(data_path=os.environ["LOCAL_STORAGE_PATH"])
    alpha = mean_reversion_alpha(client)
    print(alpha.tail(10))
```

**Step 4: Create quality_factor.py**

```python
"""Quality factor alpha using fundamentals."""
from alphalab.api.client import AlphaLabClient
import os

def quality_alpha(client):
    """ROA-based quality factor, sector-neutralized."""
    return client.query("""
roa = income / assets;
group_neutralize(rank(roa), sector)
""")

if __name__ == "__main__":
    client = AlphaLabClient(data_path=os.environ["LOCAL_STORAGE_PATH"])
    alpha = quality_alpha(client)
    print(alpha.tail(10))
```

**Step 5: Create combined_alpha.py**

```python
"""Combined multi-factor alpha."""
from alphalab.api.client import AlphaLabClient
import os

def combined_alpha(client):
    """Combine momentum, value, and quality signals."""
    return client.query("""
momentum = rank(-ts_delta(close, 20));
value = rank(book / close);
quality = rank(income / assets);
group_neutralize(momentum + value + quality, sector)
""")

if __name__ == "__main__":
    client = AlphaLabClient(data_path=os.environ["LOCAL_STORAGE_PATH"])
    alpha = combined_alpha(client)
    print(alpha.tail(10))
```

**Step 6: Commit**

```bash
git add examples/
git commit -m "feat: add example alpha scripts"
```

---

## Task 8: Set Up GitHub Actions for Docs

**Files:**
- Create: `.github/workflows/docs.yml`

**Step 1: Create workflow**

Create `.github/workflows/docs.yml`:

```yaml
name: Deploy Docs

on:
  push:
    branches: [main]
    paths:
      - 'docs/**'
      - '.github/workflows/docs.yml'
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: "pages"
  cancel-in-progress: true

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Quarto
        uses: quarto-dev/quarto-actions/setup@v2

      - name: Render Quarto Project
        run: quarto render docs

      - name: Upload artifact
        uses: actions/upload-pages-artifact@v3
        with:
          path: docs/_site

  deploy:
    needs: build
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v4
```

**Step 2: Enable GitHub Pages**

In GitHub repo settings:
1. Go to Settings → Pages
2. Set Source to "GitHub Actions"

**Step 3: Commit**

```bash
git add .github/workflows/docs.yml
git commit -m "ci: add GitHub Actions workflow for docs deployment"
```

---

## Task 9: Final Verification and Push

**Step 1: Full local render**

```bash
cd /Users/zdf/Documents/GitHub/alpha-lab
quarto render docs
```

Expected: All pages render, no errors

**Step 2: Preview locally**

```bash
quarto preview docs
```

Expected: Opens browser, all pages work, notebooks render

**Step 3: Clean up old docs (optional)**

Remove old .md files that were migrated:
```bash
rm docs/API.md docs/ALPHA-GUIDE.md docs/BACKTEST.md docs/STORAGE.md
```

Or keep them for reference and add to .gitignore.

**Step 4: Push all changes**

```bash
git push origin main
```

**Step 5: Verify deployment**

Check GitHub Actions for successful deployment.
Visit: `https://dafu-zhu.github.io/alpha-lab/`

---

## Summary

| Task | Description |
|------|-------------|
| 1 | Set up Quarto project structure |
| 2 | Migrate existing docs to .qmd |
| 3 | Create quickstart notebook |
| 4 | Create expressions notebook |
| 5 | Create group operations notebook |
| 6 | Create standalone DSL notebook |
| 7 | Create example scripts |
| 8 | Set up GitHub Actions |
| 9 | Final verification and push |
