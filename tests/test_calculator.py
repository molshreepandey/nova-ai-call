"""The calculator must evaluate real math and reject anything that isn't."""

import math

import pytest

from mcp_server.calculator import CalculatorError, safe_eval


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("sqrt(1764) * 3.5", 147.0),
        ("2 + 3 * 4", 14),
        ("(2 + 3) * 4", 20),
        ("2 ** 10", 1024),
        ("2 ^ 10", 1024),  # spoken/typed caret means power
        ("-5 + +3", -2),
        ("7 // 2 + 7 % 2", 4),
        ("10 / 4", 2.5),
        ("log(e)", 1.0),
        ("log(8, 2)", 3.0),
        ("log10(1000)", 3.0),
        ("cos(0) + sin(0)", 1.0),
        ("round(pi, 2)", 3.14),
        ("factorial(5)", 120),
        ("abs(-3.5)", 3.5),
    ],
)
def test_valid_expressions(expression: str, expected: float) -> None:
    assert safe_eval(expression) == pytest.approx(expected)


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('echo pwned')",
        "open('/etc/passwd').read()",
        "().__class__.__bases__[0].__subclasses__()",
        "[x for x in range(10)]",
        "lambda: 1",
        "eval('1+1')",
        "x + 1",
        "math.sqrt(4)",
        "'a' * 3",
        "True + 1",
        "sqrt(x=4)",
        "1 if 1 else 2",
        "1 < 2",
    ],
)
def test_unsafe_or_unsupported_input_is_rejected(expression: str) -> None:
    with pytest.raises(CalculatorError):
        safe_eval(expression)


@pytest.mark.parametrize(
    "expression",
    ["9 ** 9 ** 9", "((10 ** 999) ** 999) ** 999", "2 ** 5000", "factorial(1000)", "1" * 201],
)
def test_resource_exhaustion_is_rejected(expression: str) -> None:
    with pytest.raises(CalculatorError):
        safe_eval(expression)


@pytest.mark.parametrize("expression", ["1 / 0", "sqrt(-1)", "log(0)", "", "   ", "2 +", "exp(1000)"])
def test_math_and_syntax_errors_become_calculator_errors(expression: str) -> None:
    with pytest.raises(CalculatorError):
        safe_eval(expression)


def test_result_is_finite_number() -> None:
    assert math.isfinite(safe_eval("tan(1.5)"))
