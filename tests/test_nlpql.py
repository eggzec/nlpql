"""Tests of the original SQP code NLPQL."""

# ruff:file-ignore[assert]  (tests use bare asserts and literal reference values)

from __future__ import annotations

import nlpql
import numpy as np
import pytest


def test_tp37(tp37: dict) -> None:
    """NLPQL solves Rosenbrock's post office problem."""
    res = nlpql.minimize(
        tp37["fun"],
        tp37["x0"],
        method="NLPQL",
        jac=tp37["jac"],
        bounds=tp37["bounds"],
        constraints=tp37["constraints"],
        options={"acc": 1.0e-11, "maxfun": 10, "maxiter": 100},
    )
    assert res.success
    assert res.method == "NLPQL"
    assert res.fun == pytest.approx(-3456.0, abs=1.0e-6)
    assert res.x == pytest.approx([24.0, 12.0, 12.0], abs=1.0e-6)
    assert res.multipliers[:2] == pytest.approx([0.0, 144.0], abs=1.0e-6)


def test_agrees_with_nlpqlp(tp37: dict) -> None:
    """For a serial monotone line search both codes coincide."""
    common = {
        "jac": tp37["jac"],
        "bounds": tp37["bounds"],
        "constraints": tp37["constraints"],
    }
    first = nlpql.minimize(
        tp37["fun"],
        tp37["x0"],
        method="NLPQL",
        options={"acc": 1.0e-11, "maxfun": 10},
        **common,
    )
    second = nlpql.minimize(
        tp37["fun"],
        tp37["x0"],
        method="NLPQLP",
        options={
            "acc": 1.0e-11,
            "maxfun": 10,
            "maxnm": 0,
            "rho": 0.0,
            "accqp": 0.0,
        },
        **common,
    )
    assert first.nit == second.nit
    assert first.x == pytest.approx(second.x, abs=1.0e-12)


def test_unconstrained_rosenbrock() -> None:
    """Without constraints the code reduces to a quasi-Newton method."""
    res = nlpql.minimize(
        lambda x: 100.0 * (x[1] - x[0] ** 2) ** 2 + (1.0 - x[0]) ** 2,
        [-1.2, 1.0],
        method="NLPQL",
        jac=lambda x: np.array([
            -400.0 * x[0] * (x[1] - x[0] ** 2) - 2.0 * (1.0 - x[0]),
            200.0 * (x[1] - x[0] ** 2),
        ]),
        options={"acc": 1.0e-12, "maxiter": 200},
    )
    assert res.success
    assert res.x == pytest.approx([1.0, 1.0], abs=1.0e-6)


def test_multipliers_of_the_bounds() -> None:
    """Active bounds produce non-negative multipliers."""
    res = nlpql.minimize(
        lambda x: (x[0] - 5.0) ** 2,
        [0.5],
        method="NLPQL",
        jac=lambda x: np.array([2.0 * (x[0] - 5.0)]),
        bounds=[(0.0, 1.0)],
        options={"acc": 1.0e-12},
    )
    assert res.success
    assert res.x == pytest.approx([1.0], abs=1.0e-10)
    # layout of U: m constraints, n lower bounds, n upper bounds,
    # here m = 0 and n = 1, so the upper bound multiplier is U(2)
    assert res.multipliers[0] == pytest.approx(0.0, abs=1.0e-8)
    assert res.multipliers[1] == pytest.approx(8.0, abs=1.0e-6)
