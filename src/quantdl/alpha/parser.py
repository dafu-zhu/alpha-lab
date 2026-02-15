"""AST-based safe expression evaluator for alpha DSL.

Provides alpha_eval() for parsing string expressions like:
    alpha_eval("min(close, vwap)", {"close": close_df, "vwap": vwap_df})
    alpha_eval("rank(-ts_delta(close, 5))", {"close": close_df}, ops=quantdl.operators)

For GP/RL alpha mining, operator functions are injected directly into the namespace,
allowing clean expressions without 'ops.' prefix:
    alpha_eval("rank(ts_delta(close, 5))", {"close": close_df}, ops=quantdl.operators)
"""

from __future__ import annotations

import ast
import operator
from collections.abc import Callable
from typing import Any

import polars as pl

from quantdl.alpha.core import Alpha, _get_value_cols
from quantdl.alpha.types import AlphaLike


class AlphaParseError(Exception):
    """Raised when expression parsing fails."""


# Safe binary operators
_BINARY_OPS: dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.BitAnd: operator.and_,
    ast.BitOr: operator.or_,
}

# Safe unary operators
_UNARY_OPS: dict[type, Callable[[Any], Any]] = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    ast.Invert: operator.invert,
}

# Safe builtin functions (element-wise on DataFrames)
_BUILTINS: dict[str, Callable[..., Any]] = {
    "abs": abs,
    "min": lambda *args: _elem_min(*args),
    "max": lambda *args: _elem_max(*args),
    "log": lambda x: _apply_unary(x, "log"),
    "sqrt": lambda x: _apply_unary(x, "sqrt"),
    "sign": lambda x: _apply_unary(x, "sign"),
}


def _apply_unary(x: AlphaLike, op: str) -> Alpha:
    """Apply unary function to Alpha/DataFrame."""
    if isinstance(x, Alpha):
        df = x.data
    elif isinstance(x, pl.DataFrame):
        df = x
    else:
        raise TypeError(f"Cannot apply {op} to {type(x)}")

    date_col = df.columns[0]
    value_cols = _get_value_cols(df)

    if op == "log":
        exprs = [pl.col(c).log() for c in value_cols]
    elif op == "sqrt":
        exprs = [pl.col(c).sqrt() for c in value_cols]
    elif op == "sign":
        exprs = [pl.col(c).sign() for c in value_cols]
    else:
        raise ValueError(f"Unknown unary op: {op}")

    result = df.select(pl.col(date_col), *exprs)
    return Alpha(result)


def _elem_minmax(func_name: str, use_less_than: bool, args: tuple[AlphaLike, ...]) -> Alpha:
    """Element-wise min/max across arguments.

    Args:
        func_name: Function name for error messages ('min' or 'max')
        use_less_than: If True, use < comparison (min); if False, use > (max)
        args: Values to compare

    Returns:
        Alpha with element-wise min/max result
    """
    if len(args) < 2:
        raise ValueError(f"{func_name} requires at least 2 arguments")

    first = args[0]
    if isinstance(first, Alpha):
        base_df = first.data
    elif isinstance(first, pl.DataFrame):
        base_df = first
    else:
        raise TypeError(f"Cannot apply {func_name} to {type(first)}")

    date_col = base_df.columns[0]
    value_cols = _get_value_cols(base_df)
    exprs = {c: pl.col(c) for c in value_cols}

    for arg in args[1:]:
        if isinstance(arg, Alpha):
            other_df = arg.data
        elif isinstance(arg, pl.DataFrame):
            other_df = arg
        elif isinstance(arg, (int, float)):
            for c in value_cols:
                cond = exprs[c] < arg if use_less_than else exprs[c] > arg
                exprs[c] = pl.when(cond).then(exprs[c]).otherwise(arg)
            continue
        else:
            raise TypeError(f"Cannot apply {func_name} to {type(arg)}")

        for c in value_cols:
            other_col = other_df[c]
            cond = exprs[c] < other_col if use_less_than else exprs[c] > other_col
            exprs[c] = pl.when(cond).then(exprs[c]).otherwise(other_col)

    result = base_df.select(pl.col(date_col), *[exprs[c].alias(c) for c in value_cols])
    return Alpha(result)


def _elem_min(*args: AlphaLike) -> Alpha:
    """Element-wise minimum across arguments."""
    return _elem_minmax("min", True, args)


def _elem_max(*args: AlphaLike) -> Alpha:
    """Element-wise maximum across arguments."""
    return _elem_minmax("max", False, args)


class SafeEvaluator(ast.NodeVisitor):
    """AST visitor for safe expression evaluation.

    Only allows:
    - Variable references (from provided namespace)
    - Numeric literals
    - Binary/unary operators
    - Function calls (from ops namespace or builtins)
    - Attribute access (for ops.func style - backward compat)
    """

    def __init__(
        self,
        variables: dict[str, AlphaLike],
        ops: Any | None = None,
    ) -> None:
        """Initialize evaluator.

        Args:
            variables: Mapping of variable names to Alpha/DataFrame values
            ops: Module/namespace containing operator functions (e.g., quantdl.operators)
        """
        self.variables = variables
        self.ops = ops
        # Track which functions are operator functions (for Alpha unwrapping)
        self._op_funcs: set[int] = set()
        if ops is not None:
            for name in dir(ops):
                if not name.startswith('_'):
                    func = getattr(ops, name, None)
                    if callable(func):
                        self._op_funcs.add(id(func))

    def visit_Expression(self, node: ast.Expression) -> Any:
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> int | float:
        if isinstance(node.value, (int, float)):
            return node.value
        raise AlphaParseError(f"Unsupported constant type: {type(node.value)}")

    def visit_Num(self, node: ast.Num) -> int | float:  # Python 3.7 compat
        n = getattr(node, "n", None)
        if isinstance(n, (int, float)):
            return n
        raise AlphaParseError(f"Unsupported number type: {type(n)}")

    def visit_Name(self, node: ast.Name) -> AlphaLike:
        name = node.id
        if name in self.variables:
            val = self.variables[name]
            if isinstance(val, pl.DataFrame):
                return Alpha(val)
            return val
        if name in _BUILTINS:
            return _BUILTINS[name]  # type: ignore[return-value]
        # Check if it's an operator function (injected directly)
        if self.ops is not None and hasattr(self.ops, name):
            func = getattr(self.ops, name)
            if callable(func):
                return func  # type: ignore[no-any-return]
        raise AlphaParseError(f"Unknown variable: {name}")

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        # Handle ops.func_name
        value = self.visit(node.value)
        if value is self.ops:
            func = getattr(self.ops, node.attr, None)
            if func is None:
                raise AlphaParseError(f"Unknown operator: ops.{node.attr}")
            return func
        raise AlphaParseError(f"Unsupported attribute access: {ast.dump(node)}")

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        left = self.visit(node.left)
        right = self.visit(node.right)
        op_type = type(node.op)
        if op_type not in _BINARY_OPS:
            raise AlphaParseError(f"Unsupported operator: {op_type.__name__}")
        return _BINARY_OPS[op_type](left, right)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        operand = self.visit(node.operand)
        op_type = type(node.op)
        if op_type not in _UNARY_OPS:
            raise AlphaParseError(f"Unsupported unary operator: {op_type.__name__}")
        return _UNARY_OPS[op_type](operand)

    def visit_Compare(self, node: ast.Compare) -> Any:
        # Handle chained comparisons: a < b < c
        left = self.visit(node.left)
        result = None
        for op, comparator in zip(node.ops, node.comparators, strict=True):
            right = self.visit(comparator)
            op_type = type(op)
            if op_type not in _BINARY_OPS:
                raise AlphaParseError(f"Unsupported comparison: {op_type.__name__}")
            comp_result = _BINARY_OPS[op_type](left, right)
            result = comp_result if result is None else result & comp_result
            left = right
        return result

    def visit_BoolOp(self, node: ast.BoolOp) -> Any:
        # Handle and/or with multiple operands
        values = [self.visit(v) for v in node.values]
        if isinstance(node.op, ast.And):
            result = values[0]
            for v in values[1:]:
                result = result & v
            return result
        if isinstance(node.op, ast.Or):
            result = values[0]
            for v in values[1:]:
                result = result | v
            return result
        raise AlphaParseError(f"Unsupported bool op: {type(node.op).__name__}")

    def _is_ops_func(self, node: ast.AST) -> bool:
        """Check if node is ops.func_name (backward compat)."""
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            return node.value.id == "ops"
        return False

    def _is_operator_func(self, func: Any) -> bool:
        """Check if func is an operator function (for Alpha unwrapping)."""
        return id(func) in self._op_funcs

    def _unwrap_for_ops(self, val: Any) -> Any:
        """Unwrap Alpha to DataFrame for ops functions."""
        if isinstance(val, Alpha):
            return val.data
        return val

    def visit_Call(self, node: ast.Call) -> Any:
        func = self.visit(node.func)
        args = [self.visit(arg) for arg in node.args]
        kwargs = {kw.arg: self.visit(kw.value) for kw in node.keywords if kw.arg}

        # If calling operator function (directly or via ops.*), unwrap Alpha to DataFrame
        if self._is_ops_func(node.func) or self._is_operator_func(func):
            args = [self._unwrap_for_ops(a) for a in args]
            kwargs = {k: self._unwrap_for_ops(v) for k, v in kwargs.items()}
            if callable(func):
                result = func(*args, **kwargs)
                # Wrap result back to Alpha
                if isinstance(result, pl.DataFrame):
                    return Alpha(result)
                return result

        if callable(func):
            return func(*args, **kwargs)
        raise AlphaParseError(f"Not callable: {func}")

    def visit_IfExp(self, node: ast.IfExp) -> Alpha:
        # Ternary: body if test else orelse
        test = self.visit(node.test)
        body = self.visit(node.body)
        orelse = self.visit(node.orelse)
        return _if_else(test, body, orelse)

    def generic_visit(self, node: ast.AST) -> None:
        raise AlphaParseError(f"Unsupported syntax: {type(node).__name__}")


def _if_else(cond: Alpha, if_true: AlphaLike, if_false: AlphaLike) -> Alpha:
    """Element-wise if-else: if_true where cond != 0, else if_false."""
    if isinstance(cond, Alpha):
        cond_df = cond.data
    elif isinstance(cond, pl.DataFrame):
        cond_df = cond
    else:
        raise TypeError(f"Condition must be Alpha/DataFrame, got {type(cond)}")

    date_col = cond_df.columns[0]
    value_cols = _get_value_cols(cond_df)

    # Get true/false values
    if isinstance(if_true, Alpha):
        true_df = if_true.data
    elif isinstance(if_true, pl.DataFrame):
        true_df = if_true
    else:
        true_df = None
        true_scalar = if_true

    if isinstance(if_false, Alpha):
        false_df = if_false.data
    elif isinstance(if_false, pl.DataFrame):
        false_df = if_false
    else:
        false_df = None
        false_scalar = if_false

    exprs = []
    for c in value_cols:
        cond_expr = pl.col(c) != 0
        true_val = true_df[c] if true_df is not None else true_scalar
        false_val = false_df[c] if false_df is not None else false_scalar
        exprs.append(pl.when(cond_expr).then(true_val).otherwise(false_val).alias(c))

    result = cond_df.select(pl.col(date_col), *exprs)
    return Alpha(result)


def alpha_eval(
    expr: str,
    variables: dict[str, AlphaLike],
    ops: Any | None = None,
) -> Alpha:
    """Evaluate alpha expression string safely.

    Uses AST parsing for safe evaluation without exec/eval.
    Operator functions are available directly without 'ops.' prefix.

    Args:
        expr: Expression string, e.g., "rank(-ts_delta(close, 5))"
        variables: Mapping of variable names to Alpha/DataFrame values
        ops: Module containing operator functions (e.g., quantdl.operators)

    Returns:
        Alpha with computed result

    Example:
        >>> import quantdl.operators as ops
        >>> # Clean syntax (recommended for GP/RL):
        >>> result = alpha_eval(
        ...     "rank(-ts_delta(close, 5))",
        ...     {"close": close_df},
        ...     ops=ops,
        ... )

        >>> # Legacy syntax (still supported):
        >>> result = alpha_eval(
        ...     "ops.rank(-ops.ts_delta(close, 5))",
        ...     {"close": close_df},
        ...     ops=ops,
        ... )

        >>> # Builtins work without ops:
        >>> result = alpha_eval("min(close, vwap)", {"close": close_df, "vwap": vwap_df})
    """
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise AlphaParseError(f"Invalid expression syntax: {e}") from e

    # Inject ops as a variable if provided
    eval_vars: dict[str, Any] = dict(variables)
    if ops is not None:
        eval_vars["ops"] = ops

    evaluator = SafeEvaluator(eval_vars, ops)
    result = evaluator.visit(tree)

    if isinstance(result, Alpha):
        return result
    if isinstance(result, pl.DataFrame):
        return Alpha(result)
    raise AlphaParseError(f"Expression did not return Alpha/DataFrame: {type(result)}")
