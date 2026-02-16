# Backtest Engine (Planned)

The backtest engine will simulate and evaluate alphas locally against the same criteria used by [WorldQuant BRAIN](https://platform.worldquantbrain.com/) for alpha submission.

## WQ BRAIN Submission Criteria

An alpha must pass all of the following to be accepted on the BRAIN platform:

| Metric | Requirement | Description |
|--------|------------|-------------|
| **Sharpe ratio** | > 1.25 | Risk-adjusted return over the simulation period |
| **Fitness** | > 1 | Combined metric balancing Sharpe, turnover, and returns |
| **Turnover** | 1% < T < 70% | Portfolio turnover must be within bounds |
| **Weight distribution** | Well-distributed | No excessive concentration in a few instruments |
| **Sub-universe Sharpe** | >= cutoff | Alpha must perform across sub-universes (see below) |

### Sub-Universe Sharpe Cutoff

The sub-universe Sharpe threshold is not a fixed constant — it scales with the relative size of the sub-universe:

```
subuniverse_sharpe >= 0.75 * sqrt(subuniverse_size / alpha_universe_size) * alpha_sharpe
```

This ensures the alpha generalizes across market segments rather than being driven by a single cluster of stocks.

**References:**
- [How to improve Sharpe](https://support.worldquantbrain.com/hc/en-us/articles/20251383456663-How-to-improve-Sharpe)
- [Sub-universe Sharpe cutoff](https://support.worldquantbrain.com/hc/en-us/articles/6568644868375-How-do-I-resolve-this-error-Sub-universe-Sharpe-NaN-is-not-above-cutoff)

## Planned Implementation

The local backtest engine will compute these metrics from alpha output and feature data, allowing you to validate alphas before submitting to WQ BRAIN.
