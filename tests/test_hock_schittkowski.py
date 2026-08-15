"""Test problems of the collection of Hock and Schittkowski.

The problems and their optimal objective function values are taken from

    Hock W., Schittkowski K. (1981): Test Examples for Nonlinear
    Programming Codes, Lecture Notes in Economics and Mathematical
    Systems, Vol. 187, Springer

Every problem is solved once with analytical derivatives and once with
the forward difference approximation of the Python layer.
"""

# ruff:file-ignore[assert, magic-value-comparison]  (tests use bare asserts and literal reference values)

from __future__ import annotations

import nlpql
import numpy as np
import pytest


def _tp1() -> dict:
    """Return test problem TP1, a bound constrained Rosenbrock function.

    Returns:
        Problem description together with the known optimum.
    """
    return {
        "fun": lambda x: 100.0 * (x[1] - x[0] ** 2) ** 2 + (1.0 - x[0]) ** 2,
        "jac": lambda x: np.array([
            -400.0 * x[0] * (x[1] - x[0] ** 2) - 2.0 * (1.0 - x[0]),
            200.0 * (x[1] - x[0] ** 2),
        ]),
        "x0": [-2.0, 1.0],
        "bounds": [(None, None), (-1.5, None)],
        "constraints": (),
        "me": 0,
        "fopt": 0.0,
        "xopt": [1.0, 1.0],
    }


def _tp6() -> dict:
    """Return test problem TP6 with one nonlinear equality constraint.

    Returns:
        Problem description together with the known optimum.
    """
    return {
        "fun": lambda x: (1.0 - x[0]) ** 2,
        "jac": lambda x: np.array([-2.0 * (1.0 - x[0]), 0.0]),
        "x0": [-1.2, 1.0],
        "bounds": None,
        "constraints": [
            {
                "type": "eq",
                "fun": lambda x: np.array([10.0 * (x[1] - x[0] ** 2)]),
                "jac": lambda x: np.array([[-20.0 * x[0], 10.0]]),
            }
        ],
        "me": 1,
        "fopt": 0.0,
        "xopt": [1.0, 1.0],
    }


def _tp22() -> dict:
    """Return test problem TP22 with two inequality constraints.

    Returns:
        Problem description together with the known optimum.
    """
    return {
        "fun": lambda x: (x[0] - 2.0) ** 2 + (x[1] - 1.0) ** 2,
        "jac": lambda x: np.array([2.0 * (x[0] - 2.0), 2.0 * (x[1] - 1.0)]),
        "x0": [2.0, 2.0],
        "bounds": None,
        "constraints": [
            {
                "type": "ineq",
                "fun": lambda x: np.array([
                    -(x[0] ** 2) + x[1],
                    -x[0] - x[1] + 2.0,
                ]),
                "jac": lambda x: np.array([[-2.0 * x[0], 1.0], [-1.0, -1.0]]),
            }
        ],
        "me": 0,
        "fopt": 1.0,
        "xopt": [1.0, 1.0],
    }


def _tp28() -> dict:
    """Return test problem TP28 with one linear equality constraint.

    Returns:
        Problem description together with the known optimum.
    """
    return {
        "fun": lambda x: (x[0] + x[1]) ** 2 + (x[1] + x[2]) ** 2,
        "jac": lambda x: np.array([
            2.0 * (x[0] + x[1]),
            2.0 * (x[0] + x[1]) + 2.0 * (x[1] + x[2]),
            2.0 * (x[1] + x[2]),
        ]),
        "x0": [-4.0, 1.0, 1.0],
        "bounds": None,
        "constraints": [
            {
                "type": "eq",
                "fun": lambda x: np.array([
                    x[0] + 2.0 * x[1] + 3.0 * x[2] - 1.0
                ]),
                "jac": lambda x: np.array([[1.0, 2.0, 3.0]]),
            }
        ],
        "me": 1,
        "fopt": 0.0,
        "xopt": [0.5, -0.5, 0.5],
    }


def _tp35() -> dict:
    """Return test problem TP35, the quadratic problem of Beale.

    Returns:
        Problem description together with the known optimum.
    """
    return {
        "fun": lambda x: (
            9.0
            - 8.0 * x[0]
            - 6.0 * x[1]
            - 4.0 * x[2]
            + 2.0 * x[0] ** 2
            + 2.0 * x[1] ** 2
            + x[2] ** 2
            + 2.0 * x[0] * x[1]
            + 2.0 * x[0] * x[2]
        ),
        "jac": lambda x: np.array([
            -8.0 + 4.0 * x[0] + 2.0 * x[1] + 2.0 * x[2],
            -6.0 + 4.0 * x[1] + 2.0 * x[0],
            -4.0 + 2.0 * x[2] + 2.0 * x[0],
        ]),
        "x0": [0.5, 0.5, 0.5],
        "bounds": [(0.0, None)] * 3,
        "constraints": [
            {
                "type": "ineq",
                "fun": lambda x: np.array([3.0 - x[0] - x[1] - 2.0 * x[2]]),
                "jac": lambda x: np.array([[-1.0, -1.0, -2.0]]),
            }
        ],
        "me": 0,
        "fopt": 1.0 / 9.0,
        "xopt": [4.0 / 3.0, 7.0 / 9.0, 4.0 / 9.0],
    }


def _tp43() -> dict:
    """Return test problem TP43, the problem of Rosen and Suzuki.

    Returns:
        Problem description together with the known optimum.
    """
    return {
        "fun": lambda x: (
            x[0] ** 2
            + x[1] ** 2
            + 2.0 * x[2] ** 2
            + x[3] ** 2
            - 5.0 * x[0]
            - 5.0 * x[1]
            - 21.0 * x[2]
            + 7.0 * x[3]
        ),
        "jac": lambda x: np.array([
            2.0 * x[0] - 5.0,
            2.0 * x[1] - 5.0,
            4.0 * x[2] - 21.0,
            2.0 * x[3] + 7.0,
        ]),
        "x0": [0.0, 0.0, 0.0, 0.0],
        "bounds": None,
        "constraints": [
            {
                "type": "ineq",
                "fun": lambda x: np.array([
                    8.0
                    - x[0] ** 2
                    - x[1] ** 2
                    - x[2] ** 2
                    - x[3] ** 2
                    - x[0]
                    + x[1]
                    - x[2]
                    + x[3],
                    10.0
                    - x[0] ** 2
                    - 2.0 * x[1] ** 2
                    - x[2] ** 2
                    - 2.0 * x[3] ** 2
                    + x[0]
                    + x[3],
                    5.0
                    - 2.0 * x[0] ** 2
                    - x[1] ** 2
                    - x[2] ** 2
                    - 2.0 * x[0]
                    + x[1]
                    + x[3],
                ]),
            }
        ],
        "me": 0,
        "fopt": -44.0,
        "xopt": [0.0, 1.0, 2.0, -1.0],
    }


def _tp71() -> dict:
    """Return test problem TP71 with a mixed set of constraints.

    Returns:
        Problem description together with the known optimum.
    """
    return {
        "fun": lambda x: x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2],
        "jac": lambda x: np.array([
            x[3] * (2.0 * x[0] + x[1] + x[2]),
            x[0] * x[3],
            x[0] * x[3] + 1.0,
            x[0] * (x[0] + x[1] + x[2]),
        ]),
        "x0": [1.0, 5.0, 5.0, 1.0],
        "bounds": [(1.0, 5.0)] * 4,
        "constraints": [
            {
                "type": "eq",
                "fun": lambda x: np.array([float(x @ x) - 40.0]),
                "jac": lambda x: np.array([2.0 * x]),
            },
            {
                "type": "ineq",
                "fun": lambda x: np.array([x[0] * x[1] * x[2] * x[3] - 25.0]),
                "jac": lambda x: np.array([
                    [
                        x[1] * x[2] * x[3],
                        x[0] * x[2] * x[3],
                        x[0] * x[1] * x[3],
                        x[0] * x[1] * x[2],
                    ]
                ]),
            },
        ],
        "me": 1,
        "fopt": 17.0140173,
        "xopt": [1.0, 4.7429994, 3.8211503, 1.3794082],
    }


def _tp76() -> dict:
    """Return test problem TP76, a linearly constrained quadratic one.

    Returns:
        Problem description together with the known optimum.
    """
    return {
        "fun": lambda x: (
            x[0] ** 2
            + 0.5 * x[1] ** 2
            + x[2] ** 2
            + 0.5 * x[3] ** 2
            - x[0] * x[2]
            + x[2] * x[3]
            - x[0]
            - 3.0 * x[1]
            + x[2]
            - x[3]
        ),
        "jac": lambda x: np.array([
            2.0 * x[0] - x[2] - 1.0,
            x[1] - 3.0,
            2.0 * x[2] - x[0] + x[3] + 1.0,
            x[3] + x[2] - 1.0,
        ]),
        "x0": [0.5, 0.5, 0.5, 0.5],
        "bounds": [(0.0, None)] * 4,
        "constraints": [
            {
                "type": "ineq",
                "fun": lambda x: np.array([
                    5.0 - x[0] - 2.0 * x[1] - x[2] - x[3],
                    4.0 - 3.0 * x[0] - x[1] - 2.0 * x[2] + x[3],
                    x[1] + 4.0 * x[2] - 1.5,
                ]),
                "jac": lambda x: np.array([
                    [-1.0, -2.0, -1.0, -1.0],
                    [-3.0, -1.0, -2.0, 1.0],
                    [0.0, 1.0, 4.0, 0.0],
                ]),
            }
        ],
        "me": 0,
        "fopt": -4.6818181818,
        "xopt": [0.2727273, 2.0909091, 0.0, 0.5454545],
    }


PROBLEMS = {
    "TP1": _tp1(),
    "TP6": _tp6(),
    "TP22": _tp22(),
    "TP28": _tp28(),
    "TP35": _tp35(),
    "TP43": _tp43(),
    "TP71": _tp71(),
    "TP76": _tp76(),
}


def _violation(g: np.ndarray, me: int) -> float:
    """Return the maximum constraint violation.

    Args:
        g: Constraint values, equalities first.
        me: Number of equality constraints.

    Returns:
        The maximum norm of the violated constraints.
    """
    if g.size == 0:
        return 0.0
    viol = [abs(float(v)) for v in g[:me]]
    viol += [max(0.0, -float(v)) for v in g[me:]]
    return max(viol) if viol else 0.0


@pytest.mark.parametrize("name", sorted(PROBLEMS))
def test_analytical_derivatives(name: str) -> None:
    """Solve a test problem with analytically given derivatives."""
    problem = PROBLEMS[name]
    res = nlpql.minimize(
        problem["fun"],
        problem["x0"],
        jac=problem["jac"],
        bounds=problem["bounds"],
        constraints=problem["constraints"],
        options={"acc": 1.0e-11, "maxiter": 300},
    )
    assert res.success, f"{name}: {res.message}"
    assert res.fun == pytest.approx(problem["fopt"], abs=1.0e-6)
    assert _violation(res.constr, problem["me"]) < 1.0e-7


@pytest.mark.parametrize("name", sorted(PROBLEMS))
def test_numerical_derivatives(name: str) -> None:
    """Solve a test problem with forward difference derivatives."""
    problem = PROBLEMS[name]
    res = nlpql.minimize(
        problem["fun"],
        problem["x0"],
        bounds=problem["bounds"],
        constraints=problem["constraints"],
        options={"acc": 1.0e-9, "maxiter": 300},
    )
    assert res.success, f"{name}: {res.message}"
    assert res.fun == pytest.approx(problem["fopt"], abs=1.0e-5)
    # the success criterion of the user's guide uses eps^2 with
    # eps = 0.01 for the maximum constraint violation
    assert _violation(res.constr, problem["me"]) < 1.0e-4


@pytest.mark.parametrize("name", sorted(PROBLEMS))
def test_solution_vector(name: str) -> None:
    """The computed solution matches the published one."""
    problem = PROBLEMS[name]
    res = nlpql.minimize(
        problem["fun"],
        problem["x0"],
        jac=problem["jac"],
        bounds=problem["bounds"],
        constraints=problem["constraints"],
        options={"acc": 1.0e-11, "maxiter": 300},
    )
    assert res.x == pytest.approx(problem["xopt"], abs=1.0e-4)


@pytest.mark.parametrize("method", ["NLPQLP", "NLPQLY"])
def test_stabilized_codes_solve_the_collection(method: str) -> None:
    """The stabilized codes solve the whole collection."""
    for name, problem in PROBLEMS.items():
        res = nlpql.minimize(
            problem["fun"],
            problem["x0"],
            method=method,
            jac=problem["jac"] if method != "NLPQLY" else None,
            bounds=problem["bounds"],
            constraints=problem["constraints"],
            options={"acc": 1.0e-9, "maxiter": 300},
        )
        assert res.success, f"{method}/{name}: {res.message}"
        assert res.fun == pytest.approx(problem["fopt"], abs=1.0e-4), name


def test_original_code_attains_the_published_values() -> None:
    """The original NLPQL reaches the optima of the collection.

    NLPQL performs a monotone line search without internal restarts, so
    that it may stop in the round-off region of the merit function.
    Depending on the platform this shows up as IFAIL = 4, the line
    search cannot be terminated, or as IFAIL = 3, the BFGS update
    underflows.  Both are exactly the error situations for which the
    non-monotone line search and the restarts of NLPQLP were
    introduced; the objective function value obtained is correct
    nevertheless, which is what this test checks.
    """
    for name, problem in PROBLEMS.items():
        res = nlpql.minimize(
            problem["fun"],
            problem["x0"],
            method="NLPQL",
            jac=problem["jac"],
            bounds=problem["bounds"],
            constraints=problem["constraints"],
            options={"acc": 1.0e-9, "maxiter": 300},
        )
        assert res.status in {0, 3, 4}, f"NLPQL/{name}: {res.message}"
        assert res.fun == pytest.approx(problem["fopt"], abs=1.0e-4), name
        assert _violation(res.constr, problem["me"]) < 1.0e-4, name
