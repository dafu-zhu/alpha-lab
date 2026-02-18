# Examples & Documentation Design

**Date**: 2025-02-17
**Status**: Approved
**Issue**: #44 (Phase 1)

## Overview

Create example notebooks and scripts demonstrating AlphaLab usage. Migrate documentation to Quarto for unified docs + notebook rendering.

## Goals

- Demonstrate core workflows with runnable examples
- Debug and validate DSL API through real usage
- Deploy beautiful documentation with embedded notebooks
- Provide copy-paste ready scripts for common patterns

## Stack

**Quarto** for everything:
- Documentation (replaces MkDocs plan)
- Notebook rendering (native .ipynb support)
- Deploy to GitHub Pages

## Structure

```
docs/
├── _quarto.yml              # Quarto config
├── index.qmd                # Landing page
├── getting-started/
│   ├── installation.qmd
│   ├── quickstart.qmd
│   └── data-setup.qmd
├── api/
│   ├── client.qmd           # AlphaLabClient reference
│   ├── dsl.qmd              # dsl.compute() reference
│   └── operators.qmd        # 68 operators
├── tutorials/               # Notebooks rendered by Quarto
│   ├── 01_quickstart.ipynb
│   ├── 02_expressions.ipynb
│   ├── 03_group_operations.ipynb
│   └── 04_standalone_dsl.ipynb
├── fields.qmd               # 66 data fields
└── cli.qmd                  # CLI reference

examples/
└── scripts/
    ├── momentum.py          # Simple momentum alpha
    ├── mean_reversion.py    # Mean reversion alpha
    ├── quality_factor.py    # Fundamental-based
    └── combined_alpha.py    # Multi-signal combination
```

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `01_quickstart.ipynb` | Basic client usage, first alpha, load data |
| `02_expressions.ipynb` | Multi-line expressions, variables, syntax |
| `03_group_operations.ipynb` | group_rank, group_neutralize, sector ops |
| `04_standalone_dsl.ipynb` | dsl.compute() with custom DataFrames |

## Scripts

Minimal, copy-paste ready patterns:

| Script | Alpha Type |
|--------|------------|
| `momentum.py` | Price momentum (ts_delta, ts_rank) |
| `mean_reversion.py` | Mean reversion (ts_zscore) |
| `quality_factor.py` | Fundamental ratios (income/assets) |
| `combined_alpha.py` | Multi-signal combination |

## Testing

- Notebooks executed in CI via `quarto render` or pytest
- Scripts have `if __name__ == "__main__"` block
- Failures indicate API regressions

## Deployment

```bash
quarto publish gh-pages
```

GitHub Actions workflow for auto-deploy on push to main.

## Migration

Existing `docs/*.md` files will be converted to `.qmd` format (mostly compatible, minimal changes needed).

## Open Questions

1. Custom domain (docs.alphalab.dev)?
2. Auto-generate operator docs from docstrings?

## References

- [Quarto Documentation](https://quarto.org/)
- [Quarto + Jupyter](https://quarto.org/docs/get-started/hello/jupyter.html)
- Issue #44 (Phase 1)
- Issue #45 (Phase 4 - dashboard)
