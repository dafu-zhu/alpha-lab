# Operators

68 operators aligned with [WorldQuant BRAIN conventions](https://platform.worldquantbrain.com/learn/documentation/discover-brain/operators-702).

All operators work on wide DataFrames where the first column is the date and remaining columns are security values. Parameters named `x`, `y` are `pl.DataFrame` unless noted otherwise.

---

## Time-Series (26)

Column-wise operations applied independently to each symbol over time.

### `ts_mean(x, d)`
Rolling mean over `d` periods. Partial windows allowed (min 1 sample).

### `ts_sum(x, d)`
Rolling sum over `d` periods. Partial windows allowed.

### `ts_std(x, d)`
Rolling standard deviation over `d` periods. Min 2 samples required.

### `ts_min(x, d)`
Rolling minimum over `d` periods. Partial windows allowed.

### `ts_max(x, d)`
Rolling maximum over `d` periods. Partial windows allowed.

### `ts_delta(x, d=1, lookback=None)`
Difference from `d` periods ago: `x - ts_delay(x, d)`.
- **lookback** (`pl.DataFrame | None`): Prior rows to avoid leading nulls. Result trimmed to `len(x)`.

### `ts_delay(x, d, lookback=None)`
Lag values by `d` periods.
- **lookback** (`pl.DataFrame | None`): Prior rows to avoid leading nulls. Result trimmed to `len(x)`.

### `ts_product(x, d)`
Rolling product over `d` periods. Partial windows allowed.

### `ts_count_nans(x, d)`
Count of null values in rolling window of `d` periods.

### `ts_zscore(x, d)`
Rolling z-score: `(x - rolling_mean) / rolling_std`. Min 2 samples for std.

### `ts_scale(x, d, constant=0)`
Scale to `[constant, 1+constant]` based on rolling min/max. Min 2 samples.
- **constant** (`float`): Offset added to the scaled range (default: 0).

### `ts_av_diff(x, d)`
Difference from rolling mean: `x - rolling_mean(x, d)`.

### `ts_step(x)`
Row counter replacing all values: 1, 2, 3, ... No window parameter.

### `ts_arg_max(x, d)`
Days since max in rolling window. 0 = today is max, `d-1` = oldest day was max. Returns `None` if window < `d`.

### `ts_arg_min(x, d)`
Days since min in rolling window. 0 = today is min, `d-1` = oldest day was min. Returns `None` if window < `d`.

### `ts_backfill(x, d)`
Forward-fill nulls with last valid value, limited to `d` periods.

### `kth_element(x, d, k)`
Get `k`-th lagged element. `k=0` is current, `k=1` is previous, etc. The `d` parameter is accepted but unused.

### `last_diff_value(x, d)`
Last value different from current within `d` periods. Returns `None` if no different value found.

### `days_from_last_change(x)`
Days since value changed. Resets to 0 on each change. No window parameter.

### `hump(x, hump=0.01)`
Limit per-row change magnitude. Max change = `hump * sum(|all values in row|)`. Stateful: output depends on previous output row.
- **hump** (`float`): Fraction controlling max change (default: 0.01).

### `ts_decay_linear(x, d, dense=False)`
Weighted average with linear decay weights `[1, 2, ..., d]`. Most recent value gets highest weight.
- **dense** (`bool`): If `True`, skip nulls and renormalize weights (default: `False`).

### `ts_rank(x, d, constant=0)`
Rank of current value in rolling window, scaled to `[constant, 1+constant]`. Partial windows allowed.
- **constant** (`float`): Offset added to the rank range (default: 0).

### `ts_corr(x, y, d)`
Rolling Pearson correlation between matching columns of `x` and `y` over `d` periods. Returns `None` when std = 0.

### `ts_covariance(x, y, d)`
Rolling covariance between matching columns of `x` and `y` over `d` periods.

### `ts_quantile(x, d, driver="gaussian")`
Rolling quantile transform: rank in window then apply inverse CDF.
- **driver** (`str`): `"gaussian"` (default) for inverse normal, `"uniform"` for linear mapping to `[-1, 1]`.

### `ts_regression(y, x, d, lag=0, rettype=0)`
Rolling OLS regression of `y` on `x`. Partial windows allowed (min 2 samples).
- **lag** (`int`): Lag applied to `x` before regression (default: 0).
- **rettype** (`int | str`): Output selection:

| Value | Alias | Returns |
|-------|-------|---------|
| 0 | `"resid"` | Residual (y - predicted) |
| 1 | `"beta"` | Slope |
| 2 | `"alpha"` | Intercept |
| 3 | `"predicted"` | alpha + beta * x |
| 4 | `"corr"` | Correlation |
| 5 | `"r_squared"` | R-squared |
| 6 | `"tstat_beta"` | t-stat for beta |
| 7 | `"tstat_alpha"` | t-stat for alpha |
| 8 | `"stderr_beta"` | Std error of beta |
| 9 | `"stderr_alpha"` | Std error of alpha |

---

## Cross-Sectional (7)

Row-wise operations applied across all symbols at each date.

### `rank(x, rate=2)`
Cross-sectional rank within each row, normalized to `[0.0, 1.0]`.
- **rate** (`int`): Ranking precision. `rate=0` uses exact sort O(N log N). `rate>0` uses bucket-based approximation O(N log B) for large universes. Default: 2.

### `zscore(x)`
Cross-sectional z-score: `(x - row_mean) / row_std` across symbols per date.

### `scale(x, scale=1.0, longscale=0.0, shortscale=0.0)`
Scale so `sum(|x|)` equals target book size.
- **scale** (`float`): Target sum of absolute values (default: 1.0). Ignored when `longscale`/`shortscale` are set.
- **longscale** (`float`): Target sum of positive values. 0 = disabled.
- **shortscale** (`float`): Target sum of |negative values|. 0 = disabled.

### `normalize(x, useStd=False, limit=0.0)`
Subtract row mean. Optionally divide by std and clip.
- **useStd** (`bool`): Divide by row std after subtracting mean (default: `False`).
- **limit** (`float`): If > 0, clip result to `[-limit, +limit]` (default: 0.0).

### `quantile(x, driver="gaussian", sigma=1.0)`
Cross-sectional quantile transform: rank, shift to avoid boundaries, apply inverse CDF.
- **driver** (`str`): Distribution type: `"gaussian"`, `"uniform"`, or `"cauchy"` (default: `"gaussian"`).
- **sigma** (`float`): Scale parameter for output (default: 1.0).

### `winsorize(x, std=4.0)`
Clip values to `[mean - std*SD, mean + std*SD]` within each row.
- **std** (`float`): Number of standard deviations for clipping bounds (default: 4.0).

### `bucket(x, range_spec)`
Assign values to discrete buckets. Each value maps to its bucket's lower bound.
- **range_spec** (`str`): Comma-separated `"start,end,step"` (e.g., `"0,1,0.25"` creates quartile buckets).

---

## Group (6)

Operations applied within groups defined by a separate group DataFrame. The `group` parameter is a wide DataFrame of the same shape containing group assignment IDs.

### `group_neutralize(x, group)`
Subtract group mean from each value.

### `group_zscore(x, group)`
Z-score within groups: `(x - group_mean) / group_std`.

### `group_scale(x, group)`
Min-max scale within groups to `[0, 1]`.

### `group_rank(x, group)`
Rank within groups, normalized to `[0, 1]`. Single-member groups return 0.5.

### `group_mean(x, weight, group)`
Weighted mean within groups: `sum(x * weight) / sum(weight)`. Result broadcast to all group members.
- **weight** (`pl.DataFrame`): Wide DataFrame of weights (same shape as `x`).

### `group_backfill(x, group, d, std=4.0)`
Fill nulls with winsorized group mean over `d` lookback days.
- **d** (`int`): Number of days to look back for group values.
- **std** (`float`): Standard deviations for winsorization of fill values (default: 4.0).

---

## Arithmetic (15)

Element-wise math operations. `abs`, `min`, `max`, `sign` also accept scalars and fall back to Python built-ins when no DataFrame is passed.

### `abs(x)`
Absolute value. Accepts DataFrame or scalar.

### `add(*args, filter=False)`
Element-wise addition of two or more DataFrames.
- **filter** (`bool`): If `True`, treat null as 0 (default: `False`).

### `subtract(x, y, filter=False)`
Element-wise subtraction: `x - y`.
- **filter** (`bool`): If `True`, treat null as 0 (default: `False`).

### `multiply(*args, filter=False)`
Element-wise multiplication of two or more values. Accepts mixed DataFrame/scalar inputs.
- **filter** (`bool`): If `True`, treat null as 1 (default: `False`).

### `divide(x, y)`
Safe element-wise division: `x / y`. Division by zero returns null.

### `inverse(x)`
Multiplicative inverse: `1/x`. Division by zero returns null.

### `log(x)`
Natural logarithm. Values <= 0 return null.

### `max(*args)`
Element-wise maximum across two or more DataFrames. Falls back to Python `max()` when no DataFrames are passed.

### `min(*args)`
Element-wise minimum across two or more DataFrames. Falls back to Python `min()` when no DataFrames are passed.

### `power(x, y)`
Element-wise power: `x^y`. Accepts mixed DataFrame/scalar for both `x` and `y`.

### `signed_power(x, y)`
Signed power: `sign(x) * |x|^y`. Preserves sign of `x`. Accepts mixed DataFrame/scalar.

### `sqrt(x)`
Square root. Negative values return null.

### `sign(x)`
Sign function: 1 for positive, -1 for negative, 0 for zero. Accepts DataFrame or scalar.

### `reverse(x)`
Negation: `-x`.

### `densify(x)`
Remap unique values to consecutive integers `0..n-1` per row using dense ranking.

---

## Logical (11)

Comparison and boolean operators. Comparison operators accept DataFrame or scalar for `y` and return boolean-typed DataFrames.

### `and_(x, y)`
Logical AND of two boolean DataFrames.

### `or_(x, y)`
Logical OR of two boolean DataFrames.

### `not_(x)`
Logical NOT of a boolean DataFrame.

### `if_else(cond, then_, else_)`
Conditional selection. `then_` and `else_` accept DataFrame or scalar.
- **cond** (`pl.DataFrame`): Boolean DataFrame.
- **then_** (`pl.DataFrame | float | int`): Value(s) when `True`.
- **else_** (`pl.DataFrame | float | int`): Value(s) when `False`.

### `is_nan(x)`
Check for null or NaN values. Returns boolean DataFrame (`True` where null/NaN).

### `lt(x, y)`
Less than: `x < y`. `y` accepts DataFrame or scalar.

### `le(x, y)`
Less than or equal: `x <= y`. `y` accepts DataFrame or scalar.

### `gt(x, y)`
Greater than: `x > y`. `y` accepts DataFrame or scalar.

### `ge(x, y)`
Greater than or equal: `x >= y`. `y` accepts DataFrame or scalar.

### `eq(x, y)`
Equality: `x == y`. `y` accepts DataFrame or scalar.

### `ne(x, y)`
Not equal: `x != y`. `y` accepts DataFrame or scalar.

---

## Transformational (1)

Stateful operators driven by trade signals.

### `trade_when(trigger_trade, alpha, trigger_exit)`
Conditional alpha with entry/exit signals and carry-forward.
- **trigger_trade** (`pl.DataFrame`): Entry signals (>0 = enter/update position).
- **alpha** (`pl.DataFrame`): Alpha values used on entry.
- **trigger_exit** (`pl.DataFrame | float | int`): Exit signals (>0 = exit, sets result to NaN). Scalar (e.g., `-1`) means never exit.

Per row per symbol: if exit fires, result = NaN. Else if trade fires, result = alpha. Else carry forward previous result.

---

## Vector (2)

Operators for list-typed columns (e.g., intraday bar vectors stored as Polars list fields).

### `vec_avg(x)`
Mean of each list element across the vector field.

### `vec_sum(x)`
Sum of each list element across the vector field.
