# Tutorial QMD Conversion Design

## Problem

Jupyter notebooks (`.ipynb`) render incorrectly in Quarto:
- Code blocks lack newlines (all code runs together)
- Markdown headings merge with body text
- Lists render inline instead of as bullets

Root cause: Notebooks store source as single strings with `\n` instead of standard Jupyter array format.

## Solution

Convert all 4 tutorial notebooks from `.ipynb` to `.qmd` (Quarto markdown).

## Files

| Current (.ipynb) | New (.qmd) |
|------------------|------------|
| `docs/tutorials/01_quickstart.ipynb` | `docs/tutorials/01_quickstart.qmd` |
| `docs/tutorials/02_expressions.ipynb` | `docs/tutorials/02_expressions.qmd` |
| `docs/tutorials/03_group_operations.ipynb` | `docs/tutorials/03_group_operations.qmd` |
| `docs/tutorials/04_standalone_dsl.ipynb` | `docs/tutorials/04_standalone_dsl.qmd` |

## Format

Each `.qmd` file uses:
- YAML front matter with title
- `## Heading` for sections (concise titles)
- Explanatory text below headings
- ```` ```{python} ```` blocks for code (non-executable, display only)
- Proper markdown lists with blank lines

## Content Fixes

1. **Concise headings** — Remove explanatory text from headings
2. **Explanatory text** — Move to paragraph below heading
3. **Next Steps** — Proper markdown list with links

## Config Update

Update `docs/_quarto.yml` sidebar to reference `.qmd` files instead of `.ipynb`.

## Benefits

- Native Quarto format — renders correctly
- Easier to edit — plain text, no JSON
- Cleaner git diffs
- No execution needed for static docs
