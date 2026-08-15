"""High level front end of the NLPQL family of SQP codes."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from . import _status
from ._drivers import (
    solve_fit,
    solve_nlpjob,
    solve_nlpql,
    solve_nlpqlb,
    solve_nlpqlf,
    solve_nlpqlg,
    solve_nlpqlp,
    solve_nlpqly,
)
from ._problem import Problem, VectorProblem, size_constraints


_SOLVERS = {
    "nlpqlp": solve_nlpqlp,
    "nlpql": solve_nlpql,
    "nlpqly": solve_nlpqly,
    "nlpqlb": solve_nlpqlb,
    "nlpqlg": solve_nlpqlg,
    "nlpqlf": solve_nlpqlf,
}

_DEFAULTS: dict[str, Any] = {
    "acc": 1.0e-9,
    "accqp": 0.0,
    "stpmin": 0.0,
    "maxiter": 100,
    "maxfun": 20,
    "maxnm": 10,
    "rho": 100.0,
    "mode": 0,
    "iprint": 0,
    "iout": 6,
    "nproc": 1,
    "lql": True,
    "ncycle": 1,
    "mw": None,
    "eps": None,
    "disp": False,
    "accf": 0.0,
    "model": 1,
    "imin": 1,
    "weights": None,
    "fk": None,
    "feasibility": None,
    "method": "nlpqlp",
}


class OptimizeResult(dict):
    """Container for the result of an optimization run.

    The class behaves like a dictionary whose entries are in addition
    accessible as attributes.
    """

    def __getattr__(self, name: str) -> Any:  # ruff:ignore[any-type]
        """Return an entry of the result as an attribute.

        Args:
            name: Name of the entry.

        Returns:
            The stored value.

        Raises:
            AttributeError: If the entry does not exist.
        """
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    __setattr__ = dict.__setitem__  # type: ignore[assignment]
    __delattr__ = dict.__delitem__  # type: ignore[assignment]

    def __repr__(self) -> str:
        """Return a readable multi line representation.

        Returns:
            One ``key: value`` line per entry.
        """
        if not self:
            return f"{self.__class__.__name__}()"
        width = max(len(k) for k in self)
        items = sorted(self.items())
        return "\n".join(f"{k:>{width}}: {v!r}" for k, v in items)


def minimize(  # ruff:ignore[too-many-arguments, too-many-positional-arguments]
    fun: Callable[..., float],
    x0: Any,  # ruff:ignore[any-type]
    args: tuple = (),
    method: str = "NLPQLP",
    jac: Callable[..., Any] | None = None,
    bounds: Any = None,  # ruff:ignore[any-type]
    constraints: Any = (),  # ruff:ignore[any-type]
    tol: float | None = None,
    callback: Callable[[np.ndarray], None] | None = None,
    options: dict[str, Any] | None = None,
) -> OptimizeResult:
    """Minimize a smooth function subject to nonlinear constraints.

    The problem solved is

    .. code-block:: text

        min  f(x)
             g_j(x)  = 0 ,  j = 1, ..., me
             g_j(x) >= 0 ,  j = me+1, ..., m
             xl <= x <= xu

    where the constraints are described by dictionaries carrying the
    keys ``type``, ``fun``, ``jac`` and ``args``.

    Args:
        fun: Objective function ``f(x, *args)`` returning a scalar.
        x0: Starting point.
        args: Extra arguments passed to ``fun``, ``jac`` and to every
            constraint callable that does not define its own ``args``.
        method: One of ``"NLPQLP"``, ``"NLPQL"``, ``"NLPQLY"``,
            ``"NLPQLB"`` or ``"NLPQLG"``.
        jac: Gradient of the objective.  If ``None``, gradients are
            approximated by forward differences.
        bounds: Sequence of ``(lower, upper)`` pairs, or an object with
            the attributes ``lb`` and ``ub``.
        constraints: Constraint dictionary or a sequence of them with
            the keys ``type`` (``"eq"`` or ``"ineq"``), ``fun``, ``jac``
            and ``args``.
        tol: Desired final accuracy.  Overrides ``options["acc"]``.
        callback: Called with the current iterate after every iteration.
        options: Solver options.  Recognized keys are ``acc``,
            ``accqp``, ``stpmin``, ``maxiter``, ``maxfun``, ``maxnm``,
            ``rho``, ``mode``, ``iprint``, ``iout``, ``nproc``, ``lql``,
            ``ncycle``, ``mw``, ``eps`` and ``disp``.

    Returns:
        An :class:`OptimizeResult` with the fields ``x``, ``fun``,
        ``jac``, ``constr``, ``multipliers``, ``active``, ``status``,
        ``success``, ``message``, ``nit``, ``nfev``, ``njev`` and
        ``nqp``.

    Raises:
        ValueError: If ``method`` is not a member of the NLPQL family.

    Example:
        >>> import numpy as np
        >>> from nlpql import minimize
        >>> res = minimize(
        ...     lambda x: -x[0] * x[1] * x[2],
        ...     [10.0, 10.0, 10.0],
        ...     jac=lambda x: np.array([
        ...         -x[1] * x[2],
        ...         -x[0] * x[2],
        ...         -x[0] * x[1],
        ...     ]),
        ...     bounds=[(0.0, 42.0)] * 3,
        ...     constraints={
        ...         "type": "ineq",
        ...         "fun": lambda x: np.array([
        ...             x[0] + 2 * x[1] + 2 * x[2],
        ...             72 - x[0] - 2 * x[1] - 2 * x[2],
        ...         ]),
        ...         "jac": lambda x: np.array([
        ...             [1.0, 2.0, 2.0],
        ...             [-1.0, -2.0, -2.0],
        ...         ]),
        ...     },
        ... )
        >>> bool(res.success), round(res.fun, 6)
        (True, -3456.0)
    """
    key = str(method).lower()
    if key not in _SOLVERS:
        msg = f"unknown method {method!r}, expected one of {sorted(_SOLVERS)}"
        raise ValueError(msg)

    opt = _prepare(options, tol)
    problem = Problem(fun, x0, args, jac, bounds, constraints, opt["eps"])
    size_constraints(problem, problem.x0)
    if key == "nlpqlf":
        feas = Problem(
            fun, x0, args, None, bounds, opt["feasibility"], opt["eps"]
        )
        size_constraints(feas, feas.x0)
        opt["feasibility"] = feas
    raw = _SOLVERS[key](problem, opt, callback)
    return _result(raw, problem, method)


FIT_METHODS = ("NLPLSQ", "NLPLSX", "NLPL1", "NLPINF", "NLPMMX")

#: Scalar transformations of a multicriteria problem, see NLPJOB.
JOB_MODELS = {
    0: "individual minimum",
    1: "weighted sum",
    2: "hierarchical optimization",
    3: "trade-off method",
    4: "distance function in the L1 norm",
    5: "distance function in the L2 norm",
    6: "global criterion",
    7: "global criterion in the L2 norm",
    8: "min-max of absolute values",
    9: "min-max",
    10: "min-max of absolute distances from goals",
    11: "min-max of relative distances from ideal values",
    12: "min-max of weighted relative distances",
    13: "min-max of weighted objectives",
    14: "weighted global criterion",
    15: "weighted global criterion in the L2 norm",
}


def _prepare(
    options: dict[str, Any] | None, tol: float | None
) -> dict[str, Any]:
    """Merge user options with the defaults and normalize the types.

    Args:
        options: User supplied options or ``None``.
        tol: Desired accuracy overriding ``options["acc"]``.

    Returns:
        The complete option dictionary.
    """
    opt = dict(_DEFAULTS)
    if options:
        opt.update(options)
    if tol is not None:
        opt["acc"] = float(tol)
    if opt["disp"] and opt["iprint"] == 0:
        opt["iprint"] = 2
    for key in ("acc", "accqp", "accf", "stpmin", "rho"):
        opt[key] = float(opt[key])
    for key in (
        "maxiter",
        "maxfun",
        "maxnm",
        "mode",
        "iprint",
        "iout",
        "nproc",
        "ncycle",
        "model",
        "imin",
    ):
        opt[key] = int(opt[key])
    opt["lql"] = bool(opt["lql"])
    return opt


def _result(
    raw: dict[str, Any], problem: Problem, method: str
) -> OptimizeResult:
    """Assemble the public result object.

    Args:
        raw: Dictionary returned by one of the drivers.
        problem: Problem that has been solved, for the counters.
        method: Name of the code that has been used.

    Returns:
        The populated :class:`OptimizeResult`.
    """
    ifail = raw["status"]
    out = OptimizeResult(
        x=raw["x"],
        fun=raw["fun"],
        jac=raw["jac"],
        constr=raw["constr"],
        multipliers=raw["multipliers"],
        active=raw["active"],
        status=ifail,
        success=_status.success(ifail),
        message=_status.message(ifail),
        nit=raw["nit"],
        nfev=problem.nfev,
        njev=problem.njev,
        nqp=raw["nqp"],
        method=method.upper(),
    )
    for key in ("working_set", "residuals", "objectives", "feasibility"):
        if key in raw:
            out[key] = raw[key]
    return out


def fit(  # ruff:ignore[too-many-arguments, too-many-positional-arguments]
    fun: Callable[..., Any],
    x0: Any,  # ruff:ignore[any-type]
    args: tuple = (),
    method: str = "NLPLSQ",
    jac: Callable[..., Any] | None = None,
    bounds: Any = None,  # ruff:ignore[any-type]
    constraints: Any = (),  # ruff:ignore[any-type]
    tol: float | None = None,
    callback: Callable[[np.ndarray], None] | None = None,
    options: dict[str, Any] | None = None,
) -> OptimizeResult:
    """Solve a constrained data fitting problem.

    Depending on ``method`` the vector of individual functions returned
    by ``fun`` is combined by one of the norms

    =========  ===========================================
    NLPLSQ     sum of squares
    NLPLSX     sum of squares, very many terms
    NLPL1      sum of absolute values
    NLPINF     maximum of absolute values
    NLPMMX     maximum of the functions
    =========  ===========================================

    subject to the constraints and bounds of the problem.

    Args:
        fun: Callable returning the vector ``(f_1(x), ..., f_L(x))``.
        x0: Starting point.
        args: Extra arguments of ``fun``, ``jac`` and the constraints.
        method: One of ``"NLPLSQ"``, ``"NLPLSX"``, ``"NLPL1"``,
            ``"NLPINF"`` or ``"NLPMMX"``.
        jac: Callable returning the Jacobian of ``fun`` of shape
            ``(L, n)``.  If ``None`` it is approximated by forward
            differences.
        bounds: Sequence of ``(lower, upper)`` pairs.
        constraints: Constraint dictionary or a sequence of them.
        tol: Desired final accuracy.
        callback: Called with the current iterate after every iteration.
        options: Solver options, see :func:`minimize`.

    Returns:
        An :class:`OptimizeResult` with the additional field
        ``residuals`` holding the individual function values at the
        solution.

    Raises:
        ValueError: If ``method`` is not a data fitting code.

    Example:
        >>> import numpy as np
        >>> from nlpql import fit
        >>> res = fit(
        ...     lambda x: np.array([10.0 * (x[1] - x[0] ** 2), 1.0 - x[0]]),
        ...     [-1.2, 1.0],
        ... )
        >>> bool(res.success), bool(round(res.fun, 8) == 0.0)
        (True, True)
    """
    key = str(method).upper()
    if key not in FIT_METHODS:
        msg = f"unknown method {method!r}, expected one of {FIT_METHODS}"
        raise ValueError(msg)
    opt = _prepare(options, tol)
    opt["method"] = key.lower()
    problem = VectorProblem(fun, x0, args, jac, bounds, constraints, opt["eps"])
    size_constraints(problem, problem.x0)
    raw = solve_fit(problem, opt, callback)
    return _result(raw, problem, key)


def minimize_multi(  # ruff:ignore[too-many-arguments, too-many-positional-arguments]
    fun: Callable[..., Any],
    x0: Any,  # ruff:ignore[any-type]
    args: tuple = (),
    model: int = 1,
    imin: int = 1,
    weights: Any = None,  # ruff:ignore[any-type]
    fk: Any = None,  # ruff:ignore[any-type]
    jac: Callable[..., Any] | None = None,
    bounds: Any = None,  # ruff:ignore[any-type]
    constraints: Any = (),  # ruff:ignore[any-type]
    tol: float | None = None,
    callback: Callable[[np.ndarray], None] | None = None,
    options: dict[str, Any] | None = None,
) -> OptimizeResult:
    """Solve a multicriteria problem with ``NLPJOB``.

    The vector of objectives returned by ``fun`` is reduced to a scalar
    objective by one of sixteen transformations selected by ``model``,
    see :data:`JOB_MODELS`, and the resulting scalar program is solved
    by the SQP method.

    Args:
        fun: Callable returning the vector ``(f_1(x), ..., f_L(x))``.
        x0: Starting point.
        args: Extra arguments of ``fun``, ``jac`` and the constraints.
        model: Number of the scalar transformation, ``0 <= model <= 15``.
        imin: Index of the objective used by the models 0, 2 and 3,
            counted from one.
        weights: Weights, bounds or goal values, depending on ``model``.
        fk: Individual minima or goal values, depending on ``model``.
            The entries must be different from zero for the models 6, 7,
            11, 12, 14 and 15.
        jac: Callable returning the Jacobian of ``fun`` of shape
            ``(L, n)``.  If ``None`` it is approximated by forward
            differences.
        bounds: Sequence of ``(lower, upper)`` pairs.
        constraints: Constraint dictionary or a sequence of them.
        tol: Desired final accuracy.
        callback: Called with the current iterate after every iteration.
        options: Solver options, see :func:`minimize`.

    Returns:
        An :class:`OptimizeResult` with the additional field
        ``objectives`` holding the values of all criteria at the
        solution.

    Raises:
        ValueError: If ``model`` is not between 0 and 15.

    Example:
        >>> import numpy as np
        >>> from nlpql import minimize_multi
        >>> res = minimize_multi(
        ...     lambda x: np.array([(x[0] + 3.0) ** 2 + 1.0, x[1]]),
        ...     [0.0, 0.0],
        ...     model=0,
        ...     imin=1,
        ...     bounds=[(-10.0, 10.0)] * 2,
        ...     constraints={
        ...         "type": "ineq",
        ...         "fun": lambda x: np.array([9.0 - x[0] ** 2 - x[1] ** 2]),
        ...     },
        ... )
        >>> bool(res.success), round(res.fun, 6)
        (True, 1.0)
    """
    if int(model) not in JOB_MODELS:
        msg = f"unknown model {model!r}, expected 0 <= model <= 15"
        raise ValueError(msg)
    opt = _prepare(options, tol)
    opt["model"] = int(model)
    opt["imin"] = int(imin)
    opt["weights"] = weights
    opt["fk"] = fk
    problem = VectorProblem(fun, x0, args, jac, bounds, constraints, opt["eps"])
    size_constraints(problem, problem.x0)
    raw = solve_nlpjob(problem, opt, callback)
    return _result(raw, problem, "NLPJOB")
