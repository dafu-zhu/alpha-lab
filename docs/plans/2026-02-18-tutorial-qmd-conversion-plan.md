# Tutorial QMD Conversion Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Convert 4 Jupyter notebooks to Quarto markdown files for proper rendering.

**Architecture:** Replace `.ipynb` files with `.qmd` files containing the same content in native Quarto format. Update `_quarto.yml` to reference new files. Delete old notebooks.

**Tech Stack:** Quarto markdown, Python code blocks (display only)

---

### Task 1: Create 01_quickstart.qmd

**Files:**
- Create: `docs/tutorials/01_quickstart.qmd`

**Step 1: Create the file**

```markdown
---
title: "AlphaLab Quickstart"
---

This tutorial demonstrates basic AlphaLab usage.

```{python}
#| eval: false
from alphalab.api.client import AlphaLabClient
import os

data_path = os.environ.get("LOCAL_STORAGE_PATH", "/path/to/your/data")
client = AlphaLabClient(data_path=data_path)

# Date range: 1 year for quick tutorials
START = "2024-01-01"
END = "2024-12-31"
```

## Look Up Securities

Use `client.lookup()` to resolve symbols to security info.

```{python}
#| eval: false
# Look up a single symbol
aapl = client.lookup("AAPL")
print(aapl)

# Look up multiple symbols
for symbol in ["AAPL", "MSFT", "GOOGL"]:
    print(client.lookup(symbol))
```

## Load Data

Use `client.get()` to load pre-built feature tables.

```{python}
#| eval: false
# Load close prices (filtered by date)
close = client.get("close", start=START, end=END)
print(f"Shape: {close.shape}")
print(close.head())

# Load volume
volume = client.get("volume", start=START, end=END)
print(volume.head())
```

## Your First Alpha

Use `client.query()` to evaluate alpha expressions.

```{python}
#| eval: false
# Calculate 5-day price change
alpha = client.query("ts_delta(close, 5)", start=START, end=END)
print(alpha.head())

# Rank price change across all securities
ranked_alpha = client.query("rank(ts_delta(close, 5))", start=START, end=END)
print(ranked_alpha.head())
```

## Next Steps

- [Alpha Expressions](02_expressions.qmd) — Multi-line expressions with variables
- [Group Operations](03_group_operations.qmd) — Sector-relative operations
- [Standalone DSL](04_standalone_dsl.qmd) — Use DSL with your own DataFrames
```

**Step 2: Commit**

```bash
git add docs/tutorials/01_quickstart.qmd
git commit -m "docs: add 01_quickstart.qmd"
```

---

### Task 2: Create 02_expressions.qmd

**Files:**
- Create: `docs/tutorials/02_expressions.qmd`

**Step 1: Create the file**

```markdown
---
title: "Alpha Expressions"
---

Learn to write multi-line expressions with variables.

```{python}
#| eval: false
from alphalab.api.client import AlphaLabClient
import os

data_path = os.environ.get("LOCAL_STORAGE_PATH", "/path/to/your/data")
client = AlphaLabClient(data_path=data_path)

# Use a small subset for tutorials
SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM", "V", "JNJ"]
START = "2024-01-01"
END = "2024-12-31"
```

## Single-Line Expressions

```{python}
#| eval: false
# Daily returns (pre-computed field)
daily_returns = client.query("returns", symbols=SYMBOLS, start=START, end=END)
print(daily_returns.head())
```

```{python}
#| eval: false
# Volume momentum (is today's volume higher than average?)
volume_delta = client.query("ts_delta(volume, 5)", symbols=SYMBOLS, start=START, end=END)
print(volume_delta.head())
```

## Multi-Line with Variables

Use semicolons to separate statements.

```{python}
#| eval: false
# 20-day momentum, ranked
alpha = client.query("rank(ts_delta(close, 20))", symbols=SYMBOLS, start=START, end=END)
print(alpha.head())
```

## Intermediate Variables

Store intermediate results for cleaner code.

```{python}
#| eval: false
# Scale signal to sum to 1
alpha = client.query("scale(rank(ts_delta(close, 5)))", symbols=SYMBOLS, start=START, end=END)
print(alpha.head())
```

## Combining Signals

Add multiple ranked signals together.

```{python}
#| eval: false
# Combine momentum at different windows
alpha = client.query(
    "scale(rank(ts_delta(close, 5)) + rank(ts_delta(close, 20)))",
    symbols=SYMBOLS, start=START, end=END
)
print(alpha.head())
```

## Next Steps

- [Quickstart](01_quickstart.qmd) — Basic AlphaLab usage
- [Group Operations](03_group_operations.qmd) — Sector-relative operations
- [Standalone DSL](04_standalone_dsl.qmd) — Use DSL with your own DataFrames
```

**Step 2: Commit**

```bash
git add docs/tutorials/02_expressions.qmd
git commit -m "docs: add 02_expressions.qmd"
```

---

### Task 3: Create 03_group_operations.qmd

**Files:**
- Create: `docs/tutorials/03_group_operations.qmd`

**Step 1: Create the file**

```markdown
---
title: "Group Operations"
---

Sector-relative operations: `group_rank`, `group_neutralize`, and more.

```{python}
#| eval: false
from alphalab.api.client import AlphaLabClient
import os

data_path = os.environ.get("LOCAL_STORAGE_PATH", "/path/to/your/data")
client = AlphaLabClient(data_path=data_path)

# Use a small subset for tutorials
SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM", "V", "JNJ"]
START = "2024-01-01"
END = "2024-12-31"
```

## Group Fields

Available group fields: `sector`, `industry`, `subindustry`, `exchange`.

## group_rank

Rank within each group instead of across all securities.

```{python}
#| eval: false
# Rank momentum within each sector
# This finds the best momentum stocks relative to their sector peers
alpha = client.query("""
momentum = ts_delta(close, 20);
group_rank(momentum, sector)
""", symbols=SYMBOLS, start=START, end=END)
print(alpha.head())
```

## group_neutralize

Subtract group mean to remove sector bias.

```{python}
#| eval: false
# Sector-neutralized momentum
# Removes sector momentum, keeping only stock-specific signal
alpha = client.query("""
momentum = ts_delta(close, 20);
group_neutralize(momentum, sector)
""", symbols=SYMBOLS, start=START, end=END)
print(alpha.head())
```

## Combining Group Operations

```{python}
#| eval: false
# Industry-relative value within sectors
# 1. Neutralize by sector to remove sector effects
# 2. Rank within industry for finer-grained comparison
alpha = client.query("""
value = 1 / close;
sector_neutral = group_neutralize(value, sector);
group_rank(sector_neutral, industry)
""", symbols=SYMBOLS, start=START, end=END)
print(alpha.head())
```

## Next Steps

- [Quickstart](01_quickstart.qmd) — Basic AlphaLab usage
- [Alpha Expressions](02_expressions.qmd) — Multi-line expressions with variables
- [Standalone DSL](04_standalone_dsl.qmd) — Use DSL with your own DataFrames
```

**Step 2: Commit**

```bash
git add docs/tutorials/03_group_operations.qmd
git commit -m "docs: add 03_group_operations.qmd"
```

---

### Task 4: Create 04_standalone_dsl.qmd

**Files:**
- Create: `docs/tutorials/04_standalone_dsl.qmd`

**Step 1: Create the file**

```markdown
---
title: "Standalone DSL"
---

Use `dsl.compute()` with your own DataFrames.

```{python}
#| eval: false
from alphalab.api.dsl import compute
import polars as pl
import numpy as np
```

## Create Sample Data

Wide format: Date column + one column per security.

```{python}
#| eval: false
# Create sample price data (wide format)
np.random.seed(42)
dates = pl.date_range(pl.date(2024, 1, 1), pl.date(2024, 3, 31), eager=True)

# Generate random walk prices for 3 securities
n_days = len(dates)
price_data = {
    "Date": dates,
    "AAPL": 100 + np.cumsum(np.random.randn(n_days) * 2),
    "MSFT": 150 + np.cumsum(np.random.randn(n_days) * 2),
    "GOOGL": 120 + np.cumsum(np.random.randn(n_days) * 2),
}
prices = pl.DataFrame(price_data)
print(prices.head())
```

## Single Variable

```{python}
#| eval: false
# Calculate 5-day price change using standalone DSL
price_change = compute(
    "ts_delta(close, 5)",
    close=prices
)
print(price_change.head())
```

## Multiple Variables

```{python}
#| eval: false
# Create sample volume data
volume_data = {
    "Date": dates,
    "AAPL": np.random.randint(1000000, 5000000, n_days),
    "MSFT": np.random.randint(800000, 4000000, n_days),
    "GOOGL": np.random.randint(600000, 3000000, n_days),
}
volume = pl.DataFrame(volume_data)
print(volume.head())
```

```{python}
#| eval: false
# VWAP-like calculation with both price and volume
result = compute(
    "ts_mean(close * volume, 5) / ts_mean(volume, 5)",
    close=prices,
    volume=volume
)
print(result.head())
```

## Multi-Line Expressions

```{python}
#| eval: false
# Complex alpha with intermediate variables
alpha = compute(
    """
    momentum = ts_delta(close, 20);
    vol_signal = ts_zscore(volume, 20);
    rank(momentum) + rank(vol_signal)
    """,
    close=prices,
    volume=volume
)
print(alpha.head())
```

## When to Use Each Approach

| Approach | Use When |
|----------|----------|
| `client.query()` | Working with AlphaLab's pre-built feature tables |
| `dsl.compute()` | Using your own DataFrames or external data sources |

Both approaches support:

- All 68 operators (time-series, cross-sectional, group, etc.)
- Multi-line expressions with variables
- Boolean conditions and `trade_when`

## Next Steps

- [Quickstart](01_quickstart.qmd) — Basic AlphaLab usage
- [Alpha Expressions](02_expressions.qmd) — Multi-line expressions with variables
- [Group Operations](03_group_operations.qmd) — Sector-relative operations
```

**Step 2: Commit**

```bash
git add docs/tutorials/04_standalone_dsl.qmd
git commit -m "docs: add 04_standalone_dsl.qmd"
```

---

### Task 5: Update _quarto.yml and delete old notebooks

**Files:**
- Modify: `docs/_quarto.yml`
- Delete: `docs/tutorials/*.ipynb`

**Step 1: Update _quarto.yml**

Change all `.ipynb` references to `.qmd`:

```yaml
# In sidebar contents under "Tutorials":
- tutorials/01_quickstart.qmd
- tutorials/02_expressions.qmd
- tutorials/03_group_operations.qmd
- tutorials/04_standalone_dsl.qmd

# In navbar:
- text: "Tutorials"
  href: tutorials/01_quickstart.qmd
```

**Step 2: Delete old notebooks**

```bash
rm docs/tutorials/*.ipynb
```

**Step 3: Commit**

```bash
git add docs/_quarto.yml
git add -u docs/tutorials/
git commit -m "docs: switch tutorials from ipynb to qmd"
```

---

### Task 6: Test and deploy

**Step 1: Render locally (optional)**

```bash
cd docs && quarto render
```

**Step 2: Push and verify deployment**

```bash
git push origin main
gh run watch --exit-status
```

**Step 3: Verify site**

Open https://dafu-zhu.github.io/alpha-lab/tutorials/01_quickstart.html and confirm:

- Code blocks have proper newlines
- Headings are separate from body text
- Next Steps renders as bullet list
