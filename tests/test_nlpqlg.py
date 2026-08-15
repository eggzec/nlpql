"""Tests of the successive restart driver NLPQLG."""

# ruff:file-ignore[assert, magic-value-comparison]  (tests use bare asserts and literal reference values)

from __future__ import annotations

import nlpql
import numpy as np
import pytest


def _rastrigin(x: np.ndarray) -> float:
    """Return the two dimensional Rastrigin function.

    Args:
        x: Point at which the function is evaluated.

    Returns:
        The function value.
    """
    return float(np.sum(x**2 - 5.0 * np.cos(2.0 * np.pi * x)))


def _rastrigin_grad(x: np.ndarray) -> np.ndarray:
    """Return the gradient of the Rastrigin function.

    Args:
        x: Point at which the gradient is evaluated.

    Returns:
        The gradient vector.
    """
    return 2.0 * x + 10.0 * np.pi * np.sin(2.0 * np.pi * x)


def test_improves_a_poor_local_minimum() -> None:
    """Successive restarts escape from a local minimum."""
    kwargs = {"jac": _rastrigin_grad, "bounds": [(-5.0, 5.0)] * 2}
    local = nlpql.minimize(_rastrigin, [2.7, 3.2], method="NLPQLP", **kwargs)
    globl = nlpql.minimize(
        _rastrigin,
        [2.7, 3.2],
        method="NLPQLG",
        options={"ncycle": 25},
        **kwargs,
    )
    assert local.success
    assert globl.success
    assert globl.fun < local.fun
    assert globl.fun == pytest.approx(-10.0, abs=1.0e-6)
    assert globl.x == pytest.approx([0.0, 0.0], abs=1.0e-6)


def test_is_deterministic() -> None:
    """The restart sequence does not use a random number generator."""
    kwargs = {
        "jac": _rastrigin_grad,
        "bounds": [(-5.0, 5.0)] * 2,
        "method": "NLPQLG",
        "options": {"ncycle": 7},
    }
    first = nlpql.minimize(_rastrigin, [1.4, -2.3], **kwargs)
    second = nlpql.minimize(_rastrigin, [1.4, -2.3], **kwargs)
    assert first.x == pytest.approx(second.x, abs=0.0)
    assert first.fun == pytest.approx(second.fun, abs=0.0)


def test_single_cycle_equals_nlpqlp(tp37: dict) -> None:
    """One cycle reproduces a plain NLPQLP run."""
    common = {
        "jac": tp37["jac"],
        "bounds": tp37["bounds"],
        "constraints": tp37["constraints"],
        "options": {"acc": 1.0e-11, "maxfun": 10, "maxnm": 0},
    }
    first = nlpql.minimize(tp37["fun"], tp37["x0"], method="NLPQLG", **common)
    second = nlpql.minimize(tp37["fun"], tp37["x0"], method="NLPQLP", **common)
    assert first.x == pytest.approx(second.x, abs=1.0e-12)
    assert first.fun == pytest.approx(second.fun, abs=1.0e-12)


def test_constrained_multistart() -> None:
    """Restarts respect the constraints of the problem."""
    res = nlpql.minimize(
        _rastrigin,
        [1.6, 1.6],
        method="NLPQLG",
        jac=_rastrigin_grad,
        bounds=[(-5.0, 5.0)] * 2,
        constraints={
            "type": "ineq",
            "fun": lambda x: np.array([x[0] + x[1] - 1.0]),
        },
        options={"ncycle": 30, "acc": 1.0e-10},
    )
    assert res.success
    assert res.constr[0] >= -1.0e-8
