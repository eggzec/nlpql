"""Tests of the SQP code NLPQLP.

The reference values are taken from the user's guide of NLPQLP,
Version 4.2, K. Schittkowski, July 2015.
"""

# ruff:file-ignore[assert, magic-value-comparison]  (tests use bare asserts and literal reference values)

from __future__ import annotations

import nlpql
import numpy as np
import pytest


def max_violation(g: np.ndarray, me: int) -> float:
    """Return the maximum constraint violation of a solution.

    Args:
        g: Constraint values, equalities first.
        me: Number of equality constraints.

    Returns:
        The maximum norm of the violated constraints.
    """
    viol = [abs(float(v)) for v in g[:me]]
    viol += [max(0.0, -float(v)) for v in g[me:]]
    return max(viol) if viol else 0.0


def test_tp37_matches_users_guide(tp37: dict) -> None:
    """Reproduce the demonstration example of the user's guide."""
    res = nlpql.minimize(
        tp37["fun"],
        tp37["x0"],
        jac=tp37["jac"],
        bounds=tp37["bounds"],
        constraints=tp37["constraints"],
        options={
            "acc": 1.0e-11,
            "accqp": 1.0e-12,
            "maxfun": 10,
            "maxiter": 100,
            "maxnm": 0,
            "rho": 0.0,
        },
    )
    assert res.success
    assert res.status == 0
    assert res.fun == pytest.approx(-3456.0, abs=1.0e-6)
    assert res.x == pytest.approx([24.0, 12.0, 12.0], abs=1.0e-6)
    # multipliers of the two constraints, cf. the printed solution
    assert res.multipliers[:2] == pytest.approx([0.0, 144.0], abs=1.0e-6)
    assert res.nit == 7
    assert res.nfev == 8
    assert res.njev == 7
    assert res.nqp == 7


def test_tp37_active_set(tp37: dict) -> None:
    """Only the second constraint is active at the solution."""
    res = nlpql.minimize(
        tp37["fun"],
        tp37["x0"],
        jac=tp37["jac"],
        bounds=tp37["bounds"],
        constraints=tp37["constraints"],
    )
    assert res.active.tolist() == [False, True]


def test_non_evaluable_functions() -> None:
    """Solve the second example of the user's guide.

    The logarithm of the first constraint can only be evaluated inside
    the disc ``x1^2 + x2^2 < 2``, so that the objective is guarded and
    the solution lies close to a singular point.
    """
    eps = 1.0e-7

    def fun(x: np.ndarray) -> float:
        return 100.0 * (x[1] - x[0] ** 2) ** 2 + (x[0] - 1.0) ** 2

    def jac(x: np.ndarray) -> np.ndarray:
        return np.array([
            -400.0 * x[0] * (x[1] - x[0] ** 2) + 2.0 * (x[0] - 1.0),
            200.0 * (x[1] - x[0] ** 2),
        ])

    def con(x: np.ndarray) -> np.ndarray:
        a = max(2.0 - x[0] ** 2 - x[1] ** 2, eps)
        return np.array([x[0] - np.log(a), a])

    res = nlpql.minimize(
        fun,
        [0.0, 0.0],
        jac=jac,
        bounds=[(-2.0, 2.0)] * 2,
        constraints={"type": "ineq", "fun": con},
        options={
            "acc": 1.0e-14,
            "accqp": 1.0e-15,
            "stpmin": 1.0e-10,
            "rho": 1.0e3,
            "maxiter": 500,
            "maxfun": 20,
            "maxnm": 20,
        },
    )
    assert res.success
    assert res.fun == pytest.approx(0.0, abs=1.0e-10)
    assert res.x == pytest.approx([1.0, 1.0], abs=1.0e-6)


def test_equality_and_inequality_mixed() -> None:
    """Equality constraints are sorted in front of the inequalities."""
    res = nlpql.minimize(
        lambda x: x[0] ** 2 + x[1] ** 2,
        [3.0, 1.0],
        jac=lambda x: 2.0 * x,
        constraints=[
            {"type": "ineq", "fun": lambda x: np.array([x[0] - 1.0])},
            {"type": "eq", "fun": lambda x: np.array([x[0] + x[1] - 3.0])},
        ],
    )
    assert res.success
    assert res.x == pytest.approx([1.5, 1.5], abs=1.0e-7)
    assert max_violation(res.constr, 1) < 1.0e-9


def test_bounds_are_never_violated() -> None:
    """All iterates stay inside the box given by the bounds."""
    seen: list[np.ndarray] = []
    res = nlpql.minimize(
        lambda x: (x[0] - 5.0) ** 2 + (x[1] + 5.0) ** 2,
        [0.5, 0.5],
        jac=lambda x: np.array([2.0 * (x[0] - 5.0), 2.0 * (x[1] + 5.0)]),
        bounds=[(0.0, 1.0), (0.0, 1.0)],
        callback=seen.append,
    )
    assert res.success
    assert res.x == pytest.approx([1.0, 0.0], abs=1.0e-10)
    for point in seen:
        assert np.all(point >= -1.0e-12)
        assert np.all(point <= 1.0 + 1.0e-12)


def test_numerical_gradients_reach_the_same_solution(tp37: dict) -> None:
    """Forward differences give the documented solution as well."""
    res = nlpql.minimize(
        tp37["fun"],
        tp37["x0"],
        bounds=tp37["bounds"],
        constraints={"type": "ineq", "fun": tp37["constraints"]["fun"]},
        options={"acc": 1.0e-9},
    )
    assert res.success
    assert res.fun == pytest.approx(-3456.0, abs=1.0e-5)


def test_distributed_line_search(tp37: dict) -> None:
    """The parallel line search finds the same optimum."""
    res = nlpql.minimize(
        tp37["fun"],
        tp37["x0"],
        jac=tp37["jac"],
        bounds=tp37["bounds"],
        constraints=tp37["constraints"],
        options={"nproc": 7, "stpmin": 1.0e-10, "acc": 1.0e-10},
    )
    assert res.success
    assert res.fun == pytest.approx(-3456.0, abs=1.0e-6)


def test_starting_point_outside_the_bounds() -> None:
    """A starting point violating a bound is rejected with IFAIL = 8."""
    res = nlpql.minimize(
        lambda x: x[0] ** 2, [5.0], jac=lambda x: 2.0 * x, bounds=[(0.0, 1.0)]
    )
    assert not res.success
    assert res.status == 8
    assert "bound" in res.message


def test_maximum_number_of_iterations() -> None:
    """Stopping after MAXIT iterations is reported as IFAIL = 1."""
    res = nlpql.minimize(
        lambda x: 100.0 * (x[1] - x[0] ** 2) ** 2 + (1.0 - x[0]) ** 2,
        [-1.2, 1.0],
        options={"maxiter": 2, "acc": 1.0e-14},
    )
    assert res.status == 1
    assert not res.success
