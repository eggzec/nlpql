"""Tests of the multicriteria code NLPJOB.

The example is taken from the user's guide of NLPJOB,

    min ( (x1+3)^2 + 1 , x2 )
    x1^2 + x2^2 <= 9 ,  x1 + x2 <= 1 ,  -10 <= x <= 10

whose individual minima are f1* = 1 and f2* = -3.
"""

# ruff:file-ignore[assert]

from __future__ import annotations

import nlpql
import numpy as np
import pytest


def _objectives(x: np.ndarray) -> np.ndarray:
    """Return the two criteria of the example.

    Args:
        x: Point at which the criteria are evaluated.

    Returns:
        Array with the two objective function values.
    """
    return np.array([(x[0] + 3.0) ** 2 + 1.0, x[1]])


def _objectives_jac(x: np.ndarray) -> np.ndarray:
    """Return the Jacobian of the two criteria.

    Args:
        x: Point at which the Jacobian is evaluated.

    Returns:
        Array of shape ``(2, 2)``.
    """
    return np.array([[2.0 * (x[0] + 3.0), 0.0], [0.0, 1.0]])


CONSTRAINTS = {
    "type": "ineq",
    "fun": lambda x: np.array([9.0 - x[0] ** 2 - x[1] ** 2, 1.0 - x[0] - x[1]]),
    "jac": lambda x: np.array([[-2.0 * x[0], -2.0 * x[1]], [-1.0, -1.0]]),
}

COMMON = {
    "jac": _objectives_jac,
    "bounds": [(-10.0, 10.0)] * 2,
    "constraints": CONSTRAINTS,
    "options": {"acc": 1.0e-10, "maxiter": 200},
}


def _run(model: int, **kwargs: object) -> nlpql.OptimizeResult:
    """Solve the example with one transformation.

    Args:
        model: Number of the scalar transformation.
        kwargs: Additional arguments of :func:`nlpql.minimize_multi`.

    Returns:
        The result of the run.
    """
    args = dict(COMMON)
    args.update(kwargs)
    return nlpql.minimize_multi(_objectives, [0.0, 0.0], model=model, **args)


@pytest.mark.parametrize("model", sorted(nlpql.JOB_MODELS))
def test_every_model_converges(model: int) -> None:
    """All sixteen transformations produce a feasible solution."""
    res = _run(model, imin=1, weights=[1.0, 1.0], fk=[1.0, -3.0])
    assert res.success, f"model {model}: {res.message}"
    # the codes stop as soon as the sum of constraint violations drops
    # below the square root of the termination accuracy
    assert res.constr[0] >= -1.0e-5  # ruff:ignore[magic-value-comparison]
    assert res.constr[1] >= -1.0e-5  # ruff:ignore[magic-value-comparison]
    assert res.objectives.size == 2  # ruff:ignore[magic-value-comparison]


def test_individual_minimum() -> None:
    """Model 0 minimizes the selected criterion only."""
    res = _run(0, imin=1, weights=[1.0, 1.0], fk=[1.0, -3.0])
    assert res.success
    assert res.fun == pytest.approx(1.0, abs=1.0e-7)
    assert res.x == pytest.approx([-3.0, 0.0], abs=1.0e-5)

    res = _run(0, imin=2, weights=[1.0, 1.0], fk=[1.0, -3.0])
    assert res.success
    assert res.fun == pytest.approx(-3.0, abs=1.0e-6)


def test_weighted_sum() -> None:
    """Model 1 minimizes the weighted sum of the criteria."""
    res = _run(1, imin=1, weights=[1.0, 1.0], fk=[1.0, -3.0])
    assert res.success
    assert res.fun == pytest.approx(float(np.sum(res.objectives)), abs=1.0e-8)


def test_min_max_of_relative_distances() -> None:
    """Model 12 equioscillates between the two relative distances.

    The example of the user's guide uses this transformation with unit
    weights, which leads to the scalar problem
    ``min max{ (x1+3)^2, (x2+3)/3 }``.
    """
    res = _run(12, imin=1, weights=[1.0, 1.0], fk=[1.0, -3.0])
    assert res.success
    first = (res.x[0] + 3.0) ** 2
    second = (res.x[1] + 3.0) / 3.0
    assert first == pytest.approx(second, abs=1.0e-6)
    assert res.fun == pytest.approx(first, abs=1.0e-6)
    # the circle constraint is active at the solution
    assert res.constr[0] == pytest.approx(0.0, abs=1.0e-7)


def test_distance_functions() -> None:
    """Models 4 and 5 minimize the distance from the given goals."""
    goals = [1.0, -3.0]
    l1 = _run(4, imin=1, weights=goals, fk=[1.0, -3.0])
    l2 = _run(5, imin=1, weights=goals, fk=[1.0, -3.0])
    assert l1.success
    assert l2.success
    dev1 = np.abs(l1.objectives - np.array(goals))
    dev2 = (l2.objectives - np.array(goals)) ** 2
    assert l1.fun == pytest.approx(float(np.sum(dev1)), abs=1.0e-6)
    assert l2.fun == pytest.approx(float(np.sum(dev2)), abs=1.0e-6)


def test_zero_reference_values_are_reported() -> None:
    """A vanishing entry of FK is reported as IFAIL = 11."""
    res = _run(6, imin=1, weights=[1.0, 1.0], fk=[1.0, 0.0])
    assert res.status == 11  # ruff:ignore[magic-value-comparison]
    assert not res.success


def test_unknown_model_is_rejected() -> None:
    """A transformation number outside 0..15 raises a ValueError."""
    with pytest.raises(ValueError, match="unknown model"):
        _run(16, imin=1, weights=[1.0, 1.0], fk=[1.0, -3.0])
