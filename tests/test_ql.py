"""Tests of the convex quadratic programming solver QL.

The reference values are taken from the example of the QL user's guide
of K. Schittkowski, February 2011.
"""

# ruff:file-ignore[assert]  (tests use bare asserts and literal reference values)

from __future__ import annotations

import nlpql
import numpy as np
import pytest


def test_users_guide_example() -> None:
    """Reproduce the quadratic program of the QL user's guide."""
    n = 5
    c = np.eye(n)
    d = np.array([-21.98, -1.26, 61.39, 5.3, 101.3])
    a = np.zeros((1, n))
    a[0, 0] = -7.56
    a[0, 4] = 0.5
    b = np.array([39.1])
    res = nlpql.solve_qp(
        c, d, a, b, me=0, xl=np.full(n, -100.0), xu=np.full(n, 100.0)
    )
    assert res.success
    expected = np.array([
        -1.42539840706855,
        1.26,
        -61.39,
        -5.3,
        -99.7520239148764,
    ])
    assert res.x == pytest.approx(expected, abs=1.0e-9)
    assert res.fun == pytest.approx(-6996.50559772314, abs=1.0e-8)


def test_unconstrained_minimum() -> None:
    """A quadratic program without constraints is solved directly."""
    c = np.array([[2.0, 0.0], [0.0, 4.0]])
    d = np.array([-2.0, -8.0])
    res = nlpql.solve_qp(c, d)
    assert res.success
    assert res.x == pytest.approx([1.0, 2.0], abs=1.0e-12)


def test_equality_constraint_is_satisfied() -> None:
    """An equality constrained quadratic program is solved exactly."""
    c = np.eye(3)
    d = np.zeros(3)
    a = np.array([[1.0, 1.0, 1.0]])
    b = np.array([-3.0])
    res = nlpql.solve_qp(c, d, a, b, me=1)
    assert res.success
    assert res.x == pytest.approx([1.0, 1.0, 1.0], abs=1.0e-10)


def test_active_bound_multiplier() -> None:
    """Bounds are handled separately and return their multipliers."""
    c = np.eye(2)
    d = np.array([-5.0, -5.0])
    res = nlpql.solve_qp(c, d, xl=np.array([0.0, 0.0]), xu=np.array([1.0, 1.0]))
    assert res.success
    assert res.x == pytest.approx([1.0, 1.0], abs=1.0e-12)
    # layout of U: m constraints, then n lower, then n upper bounds,
    # here m = 0, so the lower bounds start at index 0
    assert res.multipliers[0:2] == pytest.approx([0.0, 0.0], abs=1.0e-12)
    assert res.multipliers[2:4] == pytest.approx([4.0, 4.0], abs=1.0e-10)
