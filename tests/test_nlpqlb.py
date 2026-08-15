"""Tests of the active set code NLPQLB.

The reference values are taken from the user's guide of NLPQLB,
K. Schittkowski, November 2010.  Problem P3 of that guide is the
semi-infinite program

    min  exp(x1) + exp(x2) + exp(x3)
         x1 + x2 y + x3 y^2 - 1/(1+y^2) >= 0  for all y in [0,1]

which is discretized by m equidistant points.
"""

# ruff:file-ignore[assert, magic-value-comparison]  (tests use bare asserts and literal reference values)

from __future__ import annotations

import nlpql
import numpy as np
import pytest


def _p3(m: int) -> dict:
    """Return the discretized semi-infinite test problem P3.

    Args:
        m: Number of equidistant discretization points of ``[0, 1]``.

    Returns:
        Dictionary describing the problem for :func:`nlpql.minimize`.
    """
    y = np.arange(m) / (m - 1.0)
    return {
        "fun": lambda x: float(np.sum(np.exp(x))),
        "jac": np.exp,
        "x0": [1.0, 0.5, 0.0],
        "bounds": [(-100.0, 100.0)] * 3,
        "constraints": {
            "type": "ineq",
            "fun": lambda x: x[0] + x[1] * y + x[2] * y**2 - 1.0 / (1.0 + y**2),
            "jac": lambda x: np.column_stack([np.ones_like(y), y, y**2]),
        },
    }


def test_p3_matches_users_guide() -> None:
    """Reproduce the example of the NLPQLB user's guide."""
    problem = _p3(10000)
    res = nlpql.minimize(
        problem["fun"],
        problem["x0"],
        method="NLPQLB",
        jac=problem["jac"],
        bounds=problem["bounds"],
        constraints=problem["constraints"],
        options={
            "acc": 1.0e-10,
            "accqp": 1.0e-12,
            "mw": 20,
            "maxnm": 20,
            "rho": 0.1,
            "maxiter": 500,
            "maxfun": 20,
        },
    )
    assert res.success
    assert res.fun == pytest.approx(4.3011838, abs=1.0e-6)
    assert res.x == pytest.approx(
        [1.0066048, -0.12688079, -0.379724], abs=1.0e-5
    )
    assert res.constr.size == 10000
    assert res.constr.min() > -1.0e-8


def test_working_set_contains_the_active_constraints() -> None:
    """The returned working set covers all active constraints."""
    problem = _p3(2000)
    res = nlpql.minimize(
        problem["fun"],
        problem["x0"],
        method="NLPQLB",
        jac=problem["jac"],
        bounds=problem["bounds"],
        constraints=problem["constraints"],
        options={"acc": 1.0e-10, "mw": 20, "rho": 0.1},
    )
    assert res.success
    active = np.flatnonzero(res.constr <= 1.0e-10)
    assert set(active.tolist()).issubset(set(res.working_set.tolist()))


def test_large_discretization() -> None:
    """A much finer discretization does not change the solution.

    The working set has to grow with the number of discretization
    points, because more and more constraints become nearly active.
    """
    m = 100000
    problem = _p3(m)
    res = nlpql.minimize(
        problem["fun"],
        problem["x0"],
        method="NLPQLB",
        jac=problem["jac"],
        bounds=problem["bounds"],
        constraints=problem["constraints"],
        options={"acc": 1.0e-10, "mw": 20 + m // 500, "rho": 0.1},
    )
    assert res.success
    assert res.constr.size == m
    assert res.fun == pytest.approx(4.3011838, abs=1.0e-5)


def test_agrees_with_the_dense_solver() -> None:
    """For mw = m the active set strategy reduces to plain NLPQLP."""
    problem = _p3(50)
    common = {
        "jac": problem["jac"],
        "bounds": problem["bounds"],
        "constraints": problem["constraints"],
    }
    reduced = nlpql.minimize(
        problem["fun"],
        problem["x0"],
        method="NLPQLB",
        options={"acc": 1.0e-10, "mw": 50, "rho": 0.1},
        **common,
    )
    dense = nlpql.minimize(
        problem["fun"],
        problem["x0"],
        method="NLPQLP",
        options={"acc": 1.0e-10},
        **common,
    )
    assert reduced.success
    assert dense.success
    assert reduced.fun == pytest.approx(dense.fun, abs=1.0e-8)
