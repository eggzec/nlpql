"""Shared fixtures and helpers of the NLPQL test suite."""

from __future__ import annotations

import numpy as np
import pytest


nlpql = pytest.importorskip("nlpql")


@pytest.fixture
def tp37() -> dict:
    """Return Rosenbrock's post office problem TP37.

    The problem is used as the demonstration example of the NLPQLP and
    NLPQLY user's guides, its solution is ``x = (24, 12, 12)`` with
    ``f = -3456`` and the multipliers ``u = (0, 144)``.

    Returns:
        Dictionary describing the problem for :func:`nlpql.minimize`.
    """
    return {
        "fun": lambda x: -x[0] * x[1] * x[2],
        "jac": lambda x: np.array([-x[1] * x[2], -x[0] * x[2], -x[0] * x[1]]),
        "x0": [10.0, 10.0, 10.0],
        "bounds": [(0.0, 42.0)] * 3,
        "constraints": {
            "type": "ineq",
            "fun": lambda x: np.array([
                x[0] + 2.0 * x[1] + 2.0 * x[2],
                72.0 - x[0] - 2.0 * x[1] - 2.0 * x[2],
            ]),
            "jac": lambda x: np.array([[1.0, 2.0, 2.0], [-1.0, -2.0, -2.0]]),
        },
    }
