"""Safe arithmetic evaluator built on Python's ``ast`` module (never ``eval``).

Only numeric literals, a fixed set of operators, a few constants and a whitelist
of math functions are allowed. Everything else (names, attributes, subscripts,
lambdas, comprehensions, keyword arguments...) is rejected before evaluation.
"""

from __future__ import annotations

import ast
import math
import operator
from collections.abc import Callable

MAX_EXPRESSION_LENGTH = 200
MAX_EXPONENT = 1000
MAX_RESULT_DIGITS = 1000
MAX_FACTORIAL = 170  # largest n whose factorial fits in a float

_BINARY_OPS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _factorial(n: float) -> float:
    if n != int(n) or n < 0 or n > MAX_FACTORIAL:
        raise ValueError(f"factorial needs an integer between 0 and {MAX_FACTORIAL}")
    return math.factorial(int(n))


_FUNCTIONS: dict[str, Callable[..., float]] = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "log": math.log,  # natural log, or log(x, base)
    "log10": math.log10,
    "log2": math.log2,
    "exp": math.exp,
    "abs": abs,
    "round": round,
    "floor": math.floor,
    "ceil": math.ceil,
    "factorial": _factorial,
    "radians": math.radians,
    "degrees": math.degrees,
}
_CONSTANTS: dict[str, float] = {"pi": math.pi, "e": math.e, "tau": math.tau}


class CalculatorError(ValueError):
    """Raised for invalid, unsafe or unsupported expressions."""


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise CalculatorError(f"unsupported literal: {node.value!r}")
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow):
            # Bound the result size too, so chained powers like (10**999)**999 can't explode.
            if abs(right) > MAX_EXPONENT or (
                abs(left) > 1 and right > 0 and right * math.log10(abs(left)) > MAX_RESULT_DIGITS
            ):
                raise CalculatorError("result would be too large")
        return _BINARY_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Name) and node.id in _CONSTANTS:
        return _CONSTANTS[node.id]
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCTIONS:
            raise CalculatorError(f"unsupported function; allowed: {', '.join(sorted(_FUNCTIONS))}")
        if node.keywords:
            raise CalculatorError("keyword arguments are not supported")
        return _FUNCTIONS[node.func.id](*(_eval(arg) for arg in node.args))
    raise CalculatorError(f"unsupported syntax: {type(node).__name__}")


def safe_eval(expression: str) -> float:
    """Evaluate a math expression safely and return the numeric result.

    ``^`` is accepted as a power operator because speech transcripts often use it.

    Raises:
        CalculatorError: if the expression is empty, too long, unsafe or invalid.
    """
    expression = expression.strip().replace("^", "**")
    if not expression:
        raise CalculatorError("expression is empty")
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise CalculatorError(f"expression too long (max {MAX_EXPRESSION_LENGTH} characters)")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise CalculatorError(f"invalid expression: {exc.msg}") from exc
    try:
        result = _eval(tree)
    except CalculatorError:
        raise
    except (ArithmeticError, ValueError, TypeError) as exc:
        raise CalculatorError(f"math error: {exc}") from exc
    if isinstance(result, float) and (math.isnan(result) or math.isinf(result)):
        raise CalculatorError("result is not a finite number")
    return result
