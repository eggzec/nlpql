"""Tests of the easy-to-use version NLPQLY.

The reference values are taken from the user's guide of NLPQLY,
K. Schittkowski, November 2012.
"""

# ruff:file-ignore[assert, magic-value-comparison]  (tests use bare asserts and literal reference values)

from __future__ import annotations

import nlpql
import numpy as np
import pytest


def test_tp37_with_internal_differences(tp37: dict) -> None:
    """Reproduce the example of the NLPQLY user's guide."""
    res = nlpql.minimize(
        tp37["fun"],
        tp37["x0"],
        method="NLPQLY",
        bounds=tp37["bounds"],
        constraints=tp37["constraints"],
        options={"acc": 1.0e-8, "maxiter": 100},
    )
    assert res.success
    assert res.fun == pytest.approx(-3456.0, abs=1.0e-4)
    assert res.x == pytest.approx([24.0, 12.0, 12.0], abs=1.0e-4)


def test_number_of_function_calls(tp37: dict) -> None:
    """One iteration needs one line search plus n difference calls."""
    res = nlpql.minimize(
        tp37["fun"],
        tp37["x0"],
        method="NLPQLY",
        bounds=tp37["bounds"],
        constraints=tp37["constraints"],
        options={"acc": 1.0e-8, "maxiter": 100},
    )
    assert res.nfev >= res.nit * len(tp37["x0"])
    assert res.njev == 0


def test_supplied_gradients_are_ignored(tp37: dict) -> None:
    """NLPQLY always differentiates internally."""
    res = nlpql.minimize(
        tp37["fun"],
        tp37["x0"],
        method="NLPQLY",
        jac=tp37["jac"],
        bounds=tp37["bounds"],
        constraints=tp37["constraints"],
        options={"acc": 1.0e-8},
    )
    assert res.success
    assert res.jac == pytest.approx(np.zeros(3))


def test_unconstrained_problem() -> None:
    """A problem without any constraint is accepted."""
    res = nlpql.minimize(
        lambda x: (x[0] - 3.0) ** 2 + (x[1] + 1.0) ** 2,
        [0.0, 0.0],
        method="NLPQLY",
        options={"acc": 1.0e-9},
    )
    assert res.success
    assert res.x == pytest.approx([3.0, -1.0], abs=1.0e-5)


def test_equality_constrained_problem() -> None:
    """Equality constraints are handled by the easy-to-use version."""
    res = nlpql.minimize(
        lambda x: x[0] ** 2 + x[1] ** 2,
        [2.0, 0.0],
        method="NLPQLY",
        constraints={
            "type": "eq",
            "fun": lambda x: np.array([x[0] + x[1] - 2.0]),
        },
        options={"acc": 1.0e-9},
    )
    assert res.success
    assert res.x == pytest.approx([1.0, 1.0], abs=1.0e-5)
    assert abs(res.constr[0]) < 1.0e-7
