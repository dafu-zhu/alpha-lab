"""Alpha operators for wide table transformations.

All operators work on wide DataFrames where:
- First column is the date/timestamp
- Remaining columns are symbol values
"""

from alphalab.api.profiler import profiled

# Import raw operators with underscore prefix
from alphalab.api.operators.arithmetic import (
    abs as _abs,
    add as _add,
    densify as _densify,
    divide as _divide,
    inverse as _inverse,
    log as _log,
    max as _max,
    min as _min,
    multiply as _multiply,
    power as _power,
    reverse as _reverse,
    sign as _sign,
    signed_power as _signed_power,
    sqrt as _sqrt,
    subtract as _subtract,
)
from alphalab.api.operators.cross_sectional import (
    bucket as _bucket,
    normalize as _normalize,
    quantile as _quantile,
    rank as _rank,
    scale as _scale,
    winsorize as _winsorize,
    zscore as _zscore,
)
from alphalab.api.operators.group import (
    group_backfill as _group_backfill,
    group_mean as _group_mean,
    group_neutralize as _group_neutralize,
    group_rank as _group_rank,
    group_scale as _group_scale,
    group_zscore as _group_zscore,
)
from alphalab.api.operators.logical import (
    and_ as _and_,
    eq as _eq,
    ge as _ge,
    gt as _gt,
    if_else as _if_else,
    is_nan as _is_nan,
    le as _le,
    lt as _lt,
    ne as _ne,
    not_ as _not_,
    or_ as _or_,
)
from alphalab.api.operators.time_series import (
    days_from_last_change as _days_from_last_change,
    hump as _hump,
    kth_element as _kth_element,
    last_diff_value as _last_diff_value,
    ts_arg_max as _ts_arg_max,
    ts_arg_min as _ts_arg_min,
    ts_av_diff as _ts_av_diff,
    ts_backfill as _ts_backfill,
    ts_corr as _ts_corr,
    ts_count_nans as _ts_count_nans,
    ts_covariance as _ts_covariance,
    ts_decay_linear as _ts_decay_linear,
    ts_delay as _ts_delay,
    ts_delta as _ts_delta,
    ts_max as _ts_max,
    ts_mean as _ts_mean,
    ts_min as _ts_min,
    ts_product as _ts_product,
    ts_quantile as _ts_quantile,
    ts_rank as _ts_rank,
    ts_regression as _ts_regression,
    ts_scale as _ts_scale,
    ts_std as _ts_std,
    ts_step as _ts_step,
    ts_sum as _ts_sum,
    ts_zscore as _ts_zscore,
)
from alphalab.api.operators.transformational import trade_when as _trade_when
from alphalab.api.operators.vector import vec_avg as _vec_avg, vec_sum as _vec_sum

# Wrap all operators with profiler
# Arithmetic
abs = profiled(_abs)
add = profiled(_add)
densify = profiled(_densify)
divide = profiled(_divide)
inverse = profiled(_inverse)
log = profiled(_log)
max = profiled(_max)
min = profiled(_min)
multiply = profiled(_multiply)
power = profiled(_power)
reverse = profiled(_reverse)
sign = profiled(_sign)
signed_power = profiled(_signed_power)
sqrt = profiled(_sqrt)
subtract = profiled(_subtract)

# Cross-sectional
bucket = profiled(_bucket)
normalize = profiled(_normalize)
quantile = profiled(_quantile)
rank = profiled(_rank)
scale = profiled(_scale)
winsorize = profiled(_winsorize)
zscore = profiled(_zscore)

# Group
group_backfill = profiled(_group_backfill)
group_mean = profiled(_group_mean)
group_neutralize = profiled(_group_neutralize)
group_rank = profiled(_group_rank)
group_scale = profiled(_group_scale)
group_zscore = profiled(_group_zscore)

# Logical
and_ = profiled(_and_)
eq = profiled(_eq)
ge = profiled(_ge)
gt = profiled(_gt)
if_else = profiled(_if_else)
is_nan = profiled(_is_nan)
le = profiled(_le)
lt = profiled(_lt)
ne = profiled(_ne)
not_ = profiled(_not_)
or_ = profiled(_or_)

# Time-series
days_from_last_change = profiled(_days_from_last_change)
hump = profiled(_hump)
kth_element = profiled(_kth_element)
last_diff_value = profiled(_last_diff_value)
ts_arg_max = profiled(_ts_arg_max)
ts_arg_min = profiled(_ts_arg_min)
ts_av_diff = profiled(_ts_av_diff)
ts_backfill = profiled(_ts_backfill)
ts_corr = profiled(_ts_corr)
ts_count_nans = profiled(_ts_count_nans)
ts_covariance = profiled(_ts_covariance)
ts_decay_linear = profiled(_ts_decay_linear)
ts_delay = profiled(_ts_delay)
ts_delta = profiled(_ts_delta)
ts_max = profiled(_ts_max)
ts_mean = profiled(_ts_mean)
ts_min = profiled(_ts_min)
ts_product = profiled(_ts_product)
ts_quantile = profiled(_ts_quantile)
ts_rank = profiled(_ts_rank)
ts_regression = profiled(_ts_regression)
ts_scale = profiled(_ts_scale)
ts_std = profiled(_ts_std)
ts_step = profiled(_ts_step)
ts_sum = profiled(_ts_sum)
ts_zscore = profiled(_ts_zscore)

# Transformational
trade_when = profiled(_trade_when)

# Vector
vec_avg = profiled(_vec_avg)
vec_sum = profiled(_vec_sum)

__all__ = [
    # Time-series operators (basic)
    "ts_mean",
    "ts_sum",
    "ts_std",
    "ts_min",
    "ts_max",
    "ts_delta",
    "ts_delay",
    # Time-series operators (rolling)
    "ts_product",
    "ts_count_nans",
    "ts_zscore",
    "ts_scale",
    "ts_av_diff",
    "ts_step",
    # Time-series operators (arg)
    "ts_arg_max",
    "ts_arg_min",
    # Time-series operators (lookback)
    "ts_backfill",
    "kth_element",
    "last_diff_value",
    "days_from_last_change",
    # Time-series operators (stateful)
    "hump",
    "ts_decay_linear",
    "ts_rank",
    # Time-series operators (two-variable)
    "ts_corr",
    "ts_covariance",
    "ts_quantile",
    "ts_regression",
    # Cross-sectional operators
    "bucket",
    "rank",
    "zscore",
    "normalize",
    "scale",
    "quantile",
    "winsorize",
    # Group operators
    "group_rank",
    "group_zscore",
    "group_scale",
    "group_neutralize",
    "group_mean",
    "group_backfill",
    # Vector operators
    "vec_avg",
    "vec_sum",
    # Arithmetic operators
    "abs",
    "add",
    "subtract",
    "multiply",
    "divide",
    "inverse",
    "log",
    "max",
    "min",
    "power",
    "signed_power",
    "sqrt",
    "sign",
    "reverse",
    "densify",
    # Logical operators
    "and_",
    "or_",
    "not_",
    "if_else",
    "is_nan",
    "lt",
    "le",
    "gt",
    "ge",
    "eq",
    "ne",
    # Transformational operators
    "trade_when",
]
