"""Tests of the data fitting codes.

All of them proceed from the same residual vector, Rosenbrock's
function written as

    f_1(x) = 10 (x2 - x1^2) ,   f_2(x) = 1 - x1 ,

so that the sum of squares, the sum of absolute values and the maximum
of the absolute values all vanish at ``x = (1, 1)``.
"""

# ruff:file-ignore[assert]

from __future__ import annotations

import nlpql
import numpy as np
import pytest


def _residual(x: np.ndarray) -> np.ndarray:
    """Return the residual vector of Rosenbrock's function.

    Args:
        x: Point at which the residuals are evaluated.

    Returns:
        Array with the two residuals.
    """
    return np.array([10.0 * (x[1] - x[0] ** 2), 1.0 - x[0]])


def _residual_jac(x: np.ndarray) -> np.ndarray:
    """Return the Jacobian of the residual vector.

    Args:
        x: Point at which the Jacobian is evaluated.

    Returns:
        Array of shape ``(2, 2)``.
    """
    return np.array([[-20.0 * x[0], 10.0], [-1.0, 0.0]])


@pytest.mark.parametrize("method", ["NLPLSQ", "NLPLSX", "NLPL1", "NLPINF"])
def test_norms_vanish_at_the_solution(method: str) -> None:
    """Every norm of the residual vanishes at the minimum."""
    res = nlpql.fit(
        _residual,
        [-1.2, 1.0],
        method=method,
        jac=_residual_jac,
        bounds=[(-10.0, 10.0)] * 2,
        options={"acc": 1.0e-10},
    )
    assert res.success, f"{method}: {res.message}"
    assert res.fun == pytest.approx(0.0, abs=1.0e-8)
    assert res.x == pytest.approx([1.0, 1.0], abs=1.0e-4)
    assert res.residuals == pytest.approx([0.0, 0.0], abs=1.0e-4)


def test_least_squares_matches_the_scalar_solver() -> None:
    """NLPLSQ and NLPQLP find the same minimum of the sum of squares."""
    fit = nlpql.fit(
        _residual,
        [-1.2, 1.0],
        method="NLPLSQ",
        jac=_residual_jac,
        options={"acc": 1.0e-12},
    )
    direct = nlpql.minimize(
        lambda x: float(np.sum(_residual(x) ** 2)),
        [-1.2, 1.0],
        options={"acc": 1.0e-12, "maxiter": 300},
    )
    assert fit.success
    assert direct.success
    assert fit.x == pytest.approx(direct.x, abs=1.0e-4)


def test_numerical_derivatives() -> None:
    """The residual Jacobian may be approximated by differences."""
    res = nlpql.fit(
        _residual, [-1.2, 1.0], method="NLPLSQ", options={"acc": 1.0e-9}
    )
    assert res.success
    assert res.fun == pytest.approx(0.0, abs=1.0e-8)


def test_min_max_respects_bounds_and_constraints() -> None:
    """NLPMMX minimizes the maximum of the individual functions."""
    res = nlpql.fit(
        _residual,
        [-1.2, 1.0],
        method="NLPMMX",
        jac=_residual_jac,
        bounds=[(-10.0, 10.0)] * 2,
        constraints={
            "type": "ineq",
            "fun": lambda x: np.array([2.0 - x[0] - x[1]]),
        },
        options={"acc": 1.0e-10},
    )
    assert res.success
    assert res.fun == pytest.approx(max(res.residuals), abs=1.0e-8)
    assert res.constr[0] >= -1.0e-8  # ruff:ignore[magic-value-comparison]
    assert np.all(res.x >= -10.0 - 1.0e-10)
    assert np.all(res.x <= 10.0 + 1.0e-10)


def test_maximum_norm_of_an_approximation_problem() -> None:
    """Fit a straight line to data in the maximum norm.

    The best Chebyshev approximation of the three points below by a
    straight line equioscillates, so that the optimal deviation equals
    one quarter of the absolute second difference of the data, here
    |0 - 2 + 0| / 4 = 0.5.
    """
    t = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 1.0, 0.0])
    res = nlpql.fit(
        lambda x: x[0] + x[1] * t - y,
        [0.0, 0.0],
        method="NLPINF",
        jac=lambda x: np.column_stack([np.ones_like(t), t]),
        options={"acc": 1.0e-12},
    )
    assert res.success
    assert res.fun == pytest.approx(0.5, abs=1.0e-8)
    assert res.x == pytest.approx([0.5, 0.0], abs=1.0e-6)


def test_l1_norm_of_an_approximation_problem() -> None:
    """The L1 fit of the same data interpolates two of the points."""
    t = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 1.0, 0.0])
    res = nlpql.fit(
        lambda x: x[0] + x[1] * t - y,
        [0.5, 0.5],
        method="NLPL1",
        jac=lambda x: np.column_stack([np.ones_like(t), t]),
        options={"acc": 1.0e-12},
    )
    assert res.success
    assert res.fun == pytest.approx(1.0, abs=1.0e-6)


def test_unknown_method_is_rejected() -> None:
    """A method that is not a data fitting code raises a ValueError."""
    with pytest.raises(ValueError, match="unknown method"):
        nlpql.fit(_residual, [0.0, 0.0], method="NLPQLP")
