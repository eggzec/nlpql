"""Translation of a user supplied problem into the NLPQL formulation.

The NLPQL family expects the problem in the form::

    min  f(x)
         g_j(x)  = 0 ,  j = 1, ..., me
         g_j(x) >= 0 ,  j = me+1, ..., m
         xl <= x <= xu

so that equality constraints only have to be sorted to the front of
the constraint vector.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


_BIG = 1.0e30


class Problem:
    """Objective, constraints and bounds in the NLPQL layout.

    Args:
        fun: Objective function ``f(x, *args)``.
        x0: Starting point.
        args: Extra arguments passed to every user supplied callable.
        jac: Gradient of the objective.  If ``None`` the gradient is
            approximated by forward differences.
        bounds: Sequence of ``(lower, upper)`` pairs or an object with
            the attributes ``lb`` and ``ub``.
        constraints: Constraint dictionary or a sequence of them, using
            the keys ``type`` (``"eq"`` or ``"ineq"``), ``fun``, ``jac``
            and ``args``.
        eps: Increment of the forward difference formula.
    """

    def __init__(  # ruff:ignore[too-many-arguments, too-many-positional-arguments]
        self,
        fun: Callable[..., float],
        x0: Any,  # ruff:ignore[any-type]
        args: tuple = (),
        jac: Callable[..., Any] | None = None,
        bounds: Any = None,  # ruff:ignore[any-type]
        constraints: Any = (),  # ruff:ignore[any-type]
        eps: float | None = None,
    ) -> None:
        self.x0 = np.ascontiguousarray(x0, dtype=np.float64).ravel()
        self.n = self.x0.size
        self._fun = fun
        self._jac = jac
        self._args = tuple(args)
        self.eps = (
            float(np.sqrt(np.finfo(float).eps)) if eps is None else float(eps)
        )
        self.nfev = 0
        self.njev = 0
        self.xl, self.xu = _parse_bounds(bounds, self.n)
        self._eq, self._ineq = _parse_constraints(constraints)
        self.me = sum(c["size"] for c in self._eq)
        self.m = self.me + sum(c["size"] for c in self._ineq)

    def f(self, x: Any) -> float:  # ruff:ignore[any-type]
        """Return the objective function value.

        Args:
            x: Point at which the objective is evaluated.

        Returns:
            The value of the objective function.
        """
        self.nfev += 1
        return float(self._fun(np.asarray(x), *self._args))

    def g(self, x: Any) -> np.ndarray:  # ruff:ignore[any-type]
        """Return all constraint values, equalities first.

        Args:
            x: Point at which the constraints are evaluated.

        Returns:
            Array of length ``m`` with the constraint values.
        """
        x = np.asarray(x)
        if self.m == 0:
            return np.zeros(1, dtype=np.float64)
        out = np.empty(self.m, dtype=np.float64)
        pos = 0
        for con in self._eq + self._ineq:
            val = np.atleast_1d(
                np.asarray(con["fun"](x, *con["args"]), dtype=np.float64)
            ).ravel()
            out[pos : pos + val.size] = val
            pos += val.size
        return out

    def df(self, x: Any) -> np.ndarray:  # ruff:ignore[any-type]
        """Return the gradient of the objective function.

        Args:
            x: Point at which the gradient is evaluated.

        Returns:
            Array of length ``n`` with the partial derivatives.
        """
        x = np.asarray(x, dtype=np.float64)
        if self._jac is not None:
            return np.ascontiguousarray(
                np.asarray(self._jac(x, *self._args), dtype=np.float64)
            ).ravel()
        return self._forward(x, self._fun, self._args, 1)[0]

    def dg(self, x: Any) -> np.ndarray:  # ruff:ignore[any-type]
        """Return the Jacobian of all constraints, equalities first.

        Args:
            x: Point at which the Jacobian is evaluated.

        Returns:
            Array of shape ``(m, n)`` with the constraint gradients.
        """
        x = np.asarray(x, dtype=np.float64)
        if self.m == 0:
            return np.zeros((1, self.n), dtype=np.float64)
        out = np.empty((self.m, self.n), dtype=np.float64)
        pos = 0
        for con in self._eq + self._ineq:
            size = con["size"]
            if con["jac"] is not None:
                blk = np.atleast_2d(
                    np.asarray(con["jac"](x, *con["args"]), dtype=np.float64)
                )
            else:
                blk = self._forward(x, con["fun"], con["args"], size)
            out[pos : pos + size, :] = blk.reshape(size, self.n)
            pos += size
        return out

    def _forward(
        self, x: np.ndarray, func: Callable[..., Any], args: tuple, size: int
    ) -> np.ndarray:
        """Approximate a Jacobian block by forward differences.

        Args:
            x: Point at which the derivatives are approximated.
            func: Callable returning a scalar or a vector.
            args: Extra arguments of ``func``.
            size: Number of components returned by ``func``.

        Returns:
            Array of shape ``(size, n)`` with the partial derivatives.
        """
        base = np.atleast_1d(
            np.asarray(func(x, *args), dtype=np.float64)
        ).ravel()
        out = np.empty((size, self.n), dtype=np.float64)
        for i in range(self.n):
            step = self.eps * max(1.0e-5, abs(x[i]))
            if x[i] + step > self.xu[i]:
                step = -step
            xp = x.copy()
            xp[i] = x[i] + step
            self.nfev += 1
            val = np.atleast_1d(
                np.asarray(func(xp, *args), dtype=np.float64)
            ).ravel()
            out[:, i] = (val - base) / step
        return out


def _parse_bounds(bounds: Any, n: int) -> tuple[np.ndarray, np.ndarray]:  # ruff:ignore[any-type]
    """Convert bounds of any accepted form into two arrays.

    Args:
        bounds: ``None``, a sequence of pairs or an object carrying the
            attributes ``lb`` and ``ub``.
        n: Number of optimization variables.

    Returns:
        The lower and the upper bound as arrays of length ``n``.

    Raises:
        ValueError: If the number of bounds does not match ``n``.
    """
    xl = np.full(n, -_BIG, dtype=np.float64)
    xu = np.full(n, _BIG, dtype=np.float64)
    if bounds is None:
        return xl, xu
    if hasattr(bounds, "lb") and hasattr(bounds, "ub"):
        low = np.broadcast_to(np.asarray(bounds.lb, dtype=np.float64), (n,))
        upp = np.broadcast_to(np.asarray(bounds.ub, dtype=np.float64), (n,))
        pairs = list(zip(low, upp, strict=True))
    else:
        pairs = list(bounds)
    if len(pairs) != n:
        msg = f"expected {n} bounds, got {len(pairs)}"
        raise ValueError(msg)
    for i, (low_i, upp_i) in enumerate(pairs):
        if low_i is not None and np.isfinite(low_i):
            xl[i] = float(low_i)
        if upp_i is not None and np.isfinite(upp_i):
            xu[i] = float(upp_i)
    return xl, xu


def _parse_constraints(constraints: Any) -> tuple[list, list]:  # ruff:ignore[any-type]
    """Split constraints into equalities and inequalities.

    Args:
        constraints: Constraint dictionary or a sequence of them.

    Returns:
        Two lists of normalized constraint descriptions.

    Raises:
        ValueError: If a constraint has an unknown type.
    """
    if constraints is None:
        constraints = ()
    if isinstance(constraints, dict):
        constraints = (constraints,)
    eqs: list = []
    ineqs: list = []
    for con in constraints:
        kind = str(con.get("type", "ineq")).lower()
        entry = {
            "fun": con["fun"],
            "jac": con.get("jac"),
            "args": tuple(con.get("args", ())),
            "size": int(con.get("size", 0)),
        }
        if kind.startswith("eq"):
            eqs.append(entry)
        elif kind.startswith("ineq"):
            ineqs.append(entry)
        else:
            msg = f"unknown constraint type {kind!r}"
            raise ValueError(msg)
    return eqs, ineqs


def size_constraints(problem: Problem, x: Any) -> None:  # ruff:ignore[any-type]
    """Determine the number of components of every constraint.

    Args:
        problem: Problem whose constraint sizes are unknown.
        x: Point at which the constraints are evaluated once.
    """
    x = np.asarray(x, dtype=np.float64)
    total = 0
    for con in problem._eq + problem._ineq:
        if con["size"] <= 0:
            val = np.atleast_1d(
                np.asarray(con["fun"](x, *con["args"]), dtype=np.float64)
            ).ravel()
            con["size"] = val.size
        total += con["size"]
    problem.me = sum(c["size"] for c in problem._eq)
    problem.m = total


class VectorProblem(Problem):
    """A problem whose objective is a vector of individual functions.

    The class is used by the data fitting codes, where the objective is
    built from L residuals, and by the multicriteria code, where it is
    built from L objective functions.  Constraints and bounds are
    handled exactly as for a scalar problem.

    Args:
        fun: Callable returning the vector of individual functions.
        x0: Starting point.
        args: Extra arguments passed to every user supplied callable.
        jac: Callable returning the Jacobian of ``fun`` of shape
            ``(L, n)``.  If ``None`` it is approximated by forward
            differences.
        bounds: Sequence of ``(lower, upper)`` pairs or an object with
            the attributes ``lb`` and ``ub``.
        constraints: Constraint dictionary or a sequence of them.
        eps: Increment of the forward difference formula.
    """

    def __init__(  # ruff:ignore[too-many-arguments, too-many-positional-arguments]
        self,
        fun: Callable[..., Any],
        x0: Any,  # ruff:ignore[any-type]
        args: tuple = (),
        jac: Callable[..., Any] | None = None,
        bounds: Any = None,  # ruff:ignore[any-type]
        constraints: Any = (),  # ruff:ignore[any-type]
        eps: float | None = None,
    ) -> None:
        super().__init__(
            _zero_objective, x0, args, None, bounds, constraints, eps
        )
        self._vfun = fun
        self._vjac = jac
        self.nres = int(
            np.atleast_1d(
                np.asarray(fun(self.x0, *self._args), dtype=np.float64)
            ).size
        )

    def fvec(self, x: Any) -> np.ndarray:  # ruff:ignore[any-type]
        """Return the vector of individual functions.

        Args:
            x: Point at which the functions are evaluated.

        Returns:
            Array of length ``nres``.
        """
        self.nfev += 1
        return np.ascontiguousarray(
            np.atleast_1d(
                np.asarray(
                    self._vfun(np.asarray(x), *self._args), dtype=np.float64
                )
            ).ravel()
        )

    def jvec(self, x: Any) -> np.ndarray:  # ruff:ignore[any-type]
        """Return the Jacobian of the individual functions.

        Args:
            x: Point at which the Jacobian is evaluated.

        Returns:
            Array of shape ``(nres, n)``.
        """
        x = np.asarray(x, dtype=np.float64)
        if self._vjac is not None:
            return np.ascontiguousarray(
                np.atleast_2d(
                    np.asarray(self._vjac(x, *self._args), dtype=np.float64)
                )
            ).reshape(self.nres, self.n)
        return self._forward(x, self._vfun, self._args, self.nres)


def _zero_objective(x: Any, *args: Any) -> float:  # ruff:ignore[any-type]
    """Return zero, the placeholder objective of a vector problem.

    Args:
        x: Point at which the objective is evaluated.
        args: Ignored extra arguments.

    Returns:
        Always ``0.0``.
    """
    del x, args
    return 0.0
