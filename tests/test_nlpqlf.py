"""Tests of the feasible SQP code NLPQLF.

The objective function of the example contains a square root which can
only be evaluated inside the unit disc, so that every iterate at which
the objective is requested has to satisfy the feasibility constraint.
"""

# ruff:file-ignore[assert]

from __future__ import annotations

import nlpql
import numpy as np
import pytest


class Counter:
    """Objective function which refuses to leave the unit disc.

    Attributes:
        outside: Number of evaluations that were requested outside of
            the feasible set.
        calls: Total number of evaluations.
    """

    def __init__(self) -> None:
        self.outside = 0
        self.calls = 0

    def __call__(self, x: np.ndarray) -> float:
        """Return the objective function value.

        Args:
            x: Point at which the objective is evaluated.

        Returns:
            The function value.
        """
        self.calls += 1
        radius = 1.0 - x[0] ** 2 - x[1] ** 2
        if radius < 0.0:
            self.outside += 1
            return float("nan")
        return -np.sqrt(radius) + (x[0] - 2.0) ** 2 + (x[1] - 2.0) ** 2


def _feasibility(x: np.ndarray) -> np.ndarray:
    """Return the feasibility constraint of the unit disc.

    Args:
        x: Point at which the constraint is evaluated.

    Returns:
        Array with the single constraint value.
    """
    return np.array([1.0 - x[0] ** 2 - x[1] ** 2])


def test_objective_is_never_evaluated_outside() -> None:
    """The objective is only requested at feasible arguments."""
    fun = Counter()
    res = nlpql.minimize(
        fun,
        [0.0, 0.0],
        method="NLPQLF",
        bounds=[(-2.0, 2.0)] * 2,
        options={
            "acc": 1.0e-10,
            "feasibility": {"type": "ineq", "fun": _feasibility},
        },
    )
    assert res.success
    assert fun.outside == 0
    assert fun.calls > 0
    assert res.feasibility[0] >= -1.0e-12  # ruff:ignore[magic-value-comparison]


def test_solution_of_the_disc_problem() -> None:
    """The known stationary point of the example is reproduced."""
    res = nlpql.minimize(
        Counter(),
        [0.0, 0.0],
        method="NLPQLF",
        bounds=[(-2.0, 2.0)] * 2,
        options={
            "acc": 1.0e-10,
            "feasibility": {"type": "ineq", "fun": _feasibility},
        },
    )
    assert res.success
    assert res.x == pytest.approx([0.68292977, 0.68292977], abs=1.0e-6)
    assert res.fun == pytest.approx(3.21009162, abs=1.0e-7)


def test_ordinary_constraints_are_handled_as_well() -> None:
    """Feasibility constraints may be combined with ordinary ones."""
    res = nlpql.minimize(
        Counter(),
        [0.0, 0.0],
        method="NLPQLF",
        bounds=[(-2.0, 2.0)] * 2,
        constraints={"type": "ineq", "fun": lambda x: np.array([0.5 - x[0]])},
        options={
            "acc": 1.0e-10,
            "feasibility": {"type": "ineq", "fun": _feasibility},
        },
    )
    assert res.success
    assert res.constr[0] >= -1.0e-8  # ruff:ignore[magic-value-comparison]
    assert res.x[0] <= 0.5 + 1.0e-8


def test_infeasible_starting_point_is_reported() -> None:
    """A starting point outside of the feasible set gives IFAIL = 8."""
    res = nlpql.minimize(
        lambda x: float(x @ x),
        [1.5, 1.5],
        method="NLPQLF",
        bounds=[(-2.0, 2.0)] * 2,
        options={"feasibility": {"type": "ineq", "fun": _feasibility}},
    )
    assert not res.success
    assert res.status == 8  # ruff:ignore[magic-value-comparison]
