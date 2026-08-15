"""Reverse communication drivers for the Fortran routines.

Every routine of the NLPQL family returns control to the calling
program whenever new function or gradient values are required.  The
protocol is always the same: call the routine, look at the returned
flag, provide what it asks for and call it again.  :func:`_protocol`
implements that loop once, the individual drivers only describe which
arrays a routine exchanges and how the requested values are computed.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from . import _nlpql
from ._problem import Problem, VectorProblem
from ._workspace import (
    fit_sizes,
    nlpjob_sizes,
    nlpql_sizes,
    nlpqlb_sizes,
    nlpqlf_sizes,
    nlpqlg_sizes,
    nlpqlp_sizes,
    nlpqly_sizes,
)


#: The Fortran routines request new function values with this flag.
NEED_FUNCTIONS = -1

#: The Fortran routines request new gradient values with this flag.
NEED_GRADIENTS = -2

#: Only the feasibility constraints are requested by NLPQLF.
NEED_FEASIBILITY = -3

#: Value used for a bound that is not present.
BIG = 1.0e30


def _empty(shape: tuple[int, ...]) -> np.ndarray:
    """Return a zeroed Fortran ordered double precision array.

    Args:
        shape: Shape of the array.

    Returns:
        The newly allocated array.
    """
    return np.zeros(shape, dtype=np.float64, order="F")


def _ints(size: int) -> np.ndarray:
    """Return a zeroed integer working array.

    Args:
        size: Length of the array.

    Returns:
        The newly allocated array.
    """
    return np.zeros(size, dtype=np.int32)


def _bounds(problem: Problem, nmax: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the bounds padded to the row dimension of a routine.

    The routines expect the bounds in arrays of the declared row
    dimension, the entries beyond the ``n`` optimization variables
    belong to the additional variables and are set to infinity.

    Args:
        problem: Problem carrying the bounds of the variables.
        nmax: Length of the arrays expected by the Fortran routine.

    Returns:
        The lower and the upper bound.
    """
    lower = np.full(nmax, -BIG, dtype=np.float64, order="F")
    upper = np.full(nmax, BIG, dtype=np.float64, order="F")
    lower[: problem.n] = problem.xl
    upper[: problem.n] = problem.xu
    return lower, upper


def _protocol(
    call: Callable[[], int],
    handlers: dict[int, Callable[[], None]],
    after_gradients: Callable[[], None] | None = None,
) -> int:
    """Run the reverse communication loop of one Fortran routine.

    Args:
        call: Invokes the routine once and returns the new flag.
        handlers: Maps a request flag to the action that satisfies it.
        after_gradients: Called once a new iterate has been accepted.

    Returns:
        The termination flag of the routine.
    """
    while True:
        ifail = call()
        handler = handlers.get(ifail)
        if handler is None:
            return ifail
        handler()
        if ifail == NEED_GRADIENTS and after_gradients is not None:
            after_gradients()


class _Session:
    """Arrays exchanged with one Fortran routine.

    The routines are pure, they return new objects for every array that
    they modify.  The session keeps the current objects under a name, so
    that the drivers can read and write them without threading a long
    tuple through the reverse communication loop.

    The container is heterogeneous on purpose: it holds arrays of both
    element types, scalars and the termination flag, and the argument
    lists of the routines mix all of them.  ``Any`` is therefore the
    accurate annotation for the accessors below.

    Args:
        routine: Name of the routine in the compiled extension.
        returns: Names of the values returned by the routine, in the
            order in which they appear.
    """

    def __init__(self, routine: str, returns: Sequence[str]) -> None:
        self._fun = getattr(_nlpql, routine)
        self._returns = tuple(returns)
        self.arrays: dict[str, Any] = {"ifail": 0}

    def __getitem__(self, name: str) -> Any:  # ruff:ignore[any-type]
        """Return one of the exchanged values.

        Args:
            name: Name of the value.

        Returns:
            The current object stored under that name.
        """
        return self.arrays[name]

    def add(self, **arrays: Any) -> None:  # ruff:ignore[any-type]
        """Register additional values.

        Args:
            arrays: Values to be stored, given by keyword.
        """
        self.arrays.update(arrays)

    def call(self, *args: Any) -> int:  # ruff:ignore[any-type]
        """Invoke the routine once and store everything it returns.

        Args:
            args: Positional arguments of the routine.

        Returns:
            The flag returned by the routine.
        """
        out = self._fun(*args)
        for name, value in zip(self._returns, out, strict=True):
            self.arrays[name] = value
        return int(self.arrays["ifail"])


def _statistics(session: _Session, problem: Problem) -> dict[str, Any]:
    """Return the counters reported by a routine.

    Args:
        session: Session holding the integer working array.
        problem: Problem carrying the evaluation counters.

    Returns:
        Dictionary with the statistics common to all drivers.
    """
    kwa = session["kwa"]
    return {
        "status": int(session["ifail"]),
        "nit": int(kwa[2]),
        "nqp": int(kwa[3]),
        "nfev": problem.nfev,
        "njev": problem.njev,
    }


def _no_constraints() -> np.ndarray:
    """Return the empty array used when a problem has no constraints.

    Returns:
        An empty double precision array.
    """
    return np.zeros(0)


# ---------------------------------------------------------------------
# scalar objective, the classical codes
# ---------------------------------------------------------------------


def _seed_scalar(
    problem: Problem,
    g: np.ndarray,
    df: np.ndarray,
    dg: np.ndarray,
    *,
    column: bool,
) -> float:
    """Evaluate the model functions at the starting point.

    Args:
        problem: Problem to be solved.
        g: Array receiving the constraint values.
        df: Array receiving the gradient of the objective.
        dg: Array receiving the Jacobian of the constraints.
        column: Whether ``g`` is a matrix whose first column is used.

    Returns:
        The objective function value at the starting point.
    """
    n, m = problem.n, problem.m
    fval = problem.f(problem.x0)
    if m > 0:
        values = problem.g(problem.x0)
        if column:
            g[:m, 0] = values
        else:
            g[:m] = values
        dg[:m, :n] = problem.dg(problem.x0)
    df[:n] = problem.df(problem.x0)
    problem.njev += 1
    return fval


def _scalar_result(
    session: _Session,
    problem: Problem,
    x: np.ndarray,
    fval: float,
    parts: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Assemble the result of a driver with a scalar objective.

    Args:
        session: Session holding the exchanged arrays.
        problem: Problem that has been solved.
        x: Final iterate.
        fval: Final objective function value.
        parts: Holds ``constr``, the constraint values at the final
            iterate, and ``active``, the mask of the constraints that
            the code considers active.

    Returns:
        Dictionary with the solution and the statistics.
    """
    out = {
        "x": np.array(x),
        "fun": float(fval),
        "constr": parts["constr"],
        "jac": np.array(session["df"][: problem.n]),
        "multipliers": np.array(session["u"]),
        "active": parts["active"],
    }
    out.update(_statistics(session, problem))
    return out


def solve_nlpqlp(
    problem: Problem,
    opt: dict[str, Any],
    callback: Callable[[np.ndarray], None] | None = None,
) -> dict[str, Any]:
    """Run ``NLPQLP`` on a prepared problem.

    Args:
        problem: Problem description in the NLPQL layout.
        opt: Solver options, see :func:`nlpql.minimize`.
        callback: Called with the current iterate after every iteration.

    Returns:
        Dictionary with the solution and the solver statistics.
    """
    return _solve_parallel("nlpqlpq", problem, opt, callback)


def solve_nlpqlg(
    problem: Problem,
    opt: dict[str, Any],
    callback: Callable[[np.ndarray], None] | None = None,
) -> dict[str, Any]:
    """Run ``NLPQLG``, i.e. NLPQLP restarted from several points.

    Args:
        problem: Problem description in the NLPQL layout.
        opt: Solver options, ``ncycle`` selects the number of restarts.
        callback: Called with the current iterate after every iteration.

    Returns:
        Dictionary with the best local solution found.
    """
    return _solve_parallel("nlpqlgq", problem, opt, callback)


def _solve_parallel(
    routine: str,
    problem: Problem,
    opt: dict[str, Any],
    callback: Callable[[np.ndarray], None] | None,
) -> dict[str, Any]:
    """Drive one of the codes that accept simultaneous evaluations.

    ``NLPQLP`` and ``NLPQLG`` share their calling sequence apart from
    the number of cycles, which only the latter accepts.

    Args:
        routine: Name of the routine in the compiled extension.
        problem: Problem description in the NLPQL layout.
        opt: Solver options.
        callback: Called with the current iterate after every iteration.

    Returns:
        Dictionary with the solution and the solver statistics.
    """
    n, m, me = problem.n, problem.m, problem.me
    nproc = int(opt["nproc"])
    sizes = nlpqlg_sizes if routine == "nlpqlgq" else nlpqlp_sizes
    dim = sizes(n, m, nproc)
    nmax, mmax = dim["nmax"], dim["mmax"]

    session = _Session(
        routine,
        ("x", "f", "g", "df", "dg", "u", "c", "d", "ifail", "wa", "kwa", "act"),
    )
    lower, upper = _bounds(problem, nmax)
    session.add(
        x=_empty((nmax, nproc)),
        f=_empty((nproc,)),
        g=_empty((mmax, nproc)),
        df=_empty((nmax,)),
        dg=_empty((mmax, nmax)),
        u=_empty((dim["mnn2"],)),
        c=_empty((nmax, nmax)),
        d=_empty((nmax,)),
        wa=_empty((dim["lwa"],)),
        kwa=_ints(dim["lkwa"]),
        act=_ints(dim["lact"]),
    )
    session["x"][:n, 0] = problem.x0
    session["f"][0] = _seed_scalar(
        problem, session["g"], session["df"], session["dg"], column=True
    )

    def call() -> int:
        """Invoke the routine once.

        Returns:
            The flag returned by the routine.
        """
        extra = (opt["ncycle"],) if routine == "nlpqlgq" else ()
        return session.call(
            m,
            me,
            n,
            session["x"],
            session["f"],
            session["g"],
            session["df"],
            session["dg"],
            session["u"],
            lower,
            upper,
            session["c"],
            session["d"],
            opt["acc"],
            opt["accqp"],
            opt["stpmin"],
            opt["maxfun"],
            opt["maxiter"],
            opt["maxnm"],
            opt["rho"],
            *extra,
            opt["iprint"],
            opt["mode"],
            opt["iout"],
            session["ifail"],
            session["wa"],
            session["kwa"],
            session["act"],
            opt["lql"],
        )

    def functions() -> None:
        """Evaluate the model functions at every test point."""
        for k in range(nproc):
            point = np.ascontiguousarray(session["x"][:n, k])
            session["f"][k] = problem.f(point)
            if m > 0:
                session["g"][:m, k] = problem.g(point)

    def gradients() -> None:
        """Evaluate the derivatives at the accepted iterate."""
        point = np.ascontiguousarray(session["x"][:n, 0])
        session["df"][:n] = problem.df(point)
        if m > 0:
            session["dg"][:m, :n] = problem.dg(point)
        problem.njev += 1

    def report() -> None:
        """Hand the accepted iterate to the callback."""
        if callback is not None:
            callback(np.array(session["x"][:n, 0]))

    _protocol(
        call, {NEED_FUNCTIONS: functions, NEED_GRADIENTS: gradients}, report
    )
    constr = np.array(session["g"][:m, 0]) if m > 0 else _no_constraints()
    active = (
        np.array(session["act"][:m], dtype=bool)
        if m > 0
        else np.zeros(0, dtype=bool)
    )
    return _scalar_result(
        session,
        problem,
        session["x"][:n, 0],
        session["f"][0],
        {"constr": constr, "active": active},
    )


def solve_nlpql(
    problem: Problem,
    opt: dict[str, Any],
    callback: Callable[[np.ndarray], None] | None = None,
) -> dict[str, Any]:
    """Run the original ``NLPQL`` on a prepared problem.

    Args:
        problem: Problem description in the NLPQL layout.
        opt: Solver options, see :func:`nlpql.minimize`.
        callback: Called with the current iterate after every iteration.

    Returns:
        Dictionary with the solution and the solver statistics.
    """
    n, m, me = problem.n, problem.m, problem.me
    dim = nlpql_sizes(n, m)
    nmax, mmax = dim["nmax"], dim["mmax"]

    session = _Session(
        "nlpql",
        ("x", "f", "g", "df", "dg", "u", "c", "d", "ifail", "wa", "kwa", "act"),
    )
    lower, upper = _bounds(problem, nmax)
    session.add(
        x=_empty((nmax,)),
        g=_empty((mmax,)),
        df=_empty((nmax,)),
        dg=_empty((mmax, nmax)),
        u=_empty((dim["mnn"],)),
        c=_empty((nmax, nmax)),
        d=_empty((nmax,)),
        wa=_empty((dim["lwa"],)),
        kwa=_ints(dim["lkwa"]),
        act=_ints(dim["lact"]),
    )
    session["x"][:n] = problem.x0
    session.add(
        f=_seed_scalar(
            problem, session["g"], session["df"], session["dg"], column=False
        )
    )

    def call() -> int:
        """Invoke the routine once.

        Returns:
            The flag returned by the routine.
        """
        return session.call(
            m,
            me,
            n,
            session["x"],
            session["f"],
            session["g"],
            session["df"],
            session["dg"],
            session["u"],
            lower,
            upper,
            session["c"],
            session["d"],
            opt["acc"],
            opt["maxfun"],
            opt["maxiter"],
            opt["iprint"],
            opt["iout"],
            session["ifail"],
            session["wa"],
            session["kwa"],
            session["act"],
        )

    def functions() -> None:
        """Evaluate the model functions at the test point."""
        point = np.ascontiguousarray(session["x"][:n])
        session.add(f=problem.f(point))
        if m > 0:
            session["g"][:m] = problem.g(point)

    def gradients() -> None:
        """Evaluate the derivatives at the accepted iterate."""
        point = np.ascontiguousarray(session["x"][:n])
        session["df"][:n] = problem.df(point)
        if m > 0:
            session["dg"][:m, :n] = problem.dg(point)
        problem.njev += 1

    def report() -> None:
        """Hand the accepted iterate to the callback."""
        if callback is not None:
            callback(np.array(session["x"][:n]))

    _protocol(
        call, {NEED_FUNCTIONS: functions, NEED_GRADIENTS: gradients}, report
    )
    constr = np.array(session["g"][:m]) if m > 0 else _no_constraints()
    active = (
        np.array(session["act"][:m], dtype=bool)
        if m > 0
        else np.zeros(0, dtype=bool)
    )
    return _scalar_result(
        session,
        problem,
        session["x"][:n],
        session["f"],
        {"constr": constr, "active": active},
    )


def solve_nlpqly(
    problem: Problem,
    opt: dict[str, Any],
    callback: Callable[[np.ndarray], None] | None = None,
) -> dict[str, Any]:
    """Run the easy-to-use version ``NLPQLY`` on a prepared problem.

    Args:
        problem: Problem description in the NLPQL layout.
        opt: Solver options, see :func:`nlpql.minimize`.
        callback: Called with the current iterate after every request.

    Returns:
        Dictionary with the solution and the solver statistics.
    """
    n, m, me = problem.n, problem.m, problem.me
    dim = nlpqly_sizes(n, m)

    session = _Session("nlpqly", ("x", "f", "g", "ifail", "wa", "kwa", "act"))
    session.add(
        x=np.ascontiguousarray(problem.x0.copy()),
        g=_empty((m,)),
        wa=_empty((dim["lwa"],)),
        kwa=_ints(dim["lkwa"]),
        act=_ints(dim["lact"]),
    )
    session.add(f=problem.f(session["x"]))
    if m > 0:
        session["g"][:m] = problem.g(session["x"])

    def call() -> int:
        """Invoke the routine once.

        Returns:
            The flag returned by the routine.
        """
        return session.call(
            me,
            session["x"],
            session["f"],
            session["g"],
            problem.xl,
            problem.xu,
            opt["acc"],
            opt["maxiter"],
            opt["iprint"],
            opt["iout"],
            session["ifail"],
            session["wa"],
            session["kwa"],
            session["act"],
        )

    def functions() -> None:
        """Evaluate the model functions at the requested point."""
        session.add(f=problem.f(session["x"]))
        if m > 0:
            session["g"][:m] = problem.g(session["x"])
        if callback is not None:
            callback(np.array(session["x"]))

    # every negative flag of NLPQLY asks for function values
    handlers = {flag: functions for flag in range(-20, 0)}
    _protocol(call, handlers)

    constr = np.array(session["g"][:m]) if m > 0 else _no_constraints()
    active = (
        np.array(session["act"][:m], dtype=bool)
        if m > 0
        else np.zeros(0, dtype=bool)
    )
    out = {
        "x": np.array(session["x"][:n]),
        "fun": float(session["f"]),
        "constr": constr,
        "jac": np.zeros(n),
        "multipliers": np.zeros(0),
        "active": active,
    }
    out.update(_statistics(session, problem))
    return out


def solve_nlpqlb(
    problem: Problem,
    opt: dict[str, Any],
    callback: Callable[[np.ndarray], None] | None = None,
) -> dict[str, Any]:
    """Run ``NLPQLB`` on a problem with very many constraints.

    Args:
        problem: Problem description in the NLPQL layout.
        opt: Solver options, ``mw`` selects the size of the working set.
        callback: Called with the current iterate after every iteration.

    Returns:
        Dictionary with the solution and the solver statistics.

    Raises:
        ValueError: If more constraints are active at the starting point
            than the working set can hold.
    """
    n, m, me = problem.n, problem.m, problem.me
    mw = int(opt.get("mw") or min(m, max(n + 1, 2 * n)))
    mw = max(min(mw, m), n + 1 if m > n else m)
    dim = nlpqlb_sizes(n, m, mw)
    nmax, mwmax = dim["nmax"], dim["mwmax"]

    session = _Session(
        "nlpqlbq",
        ("x", "f", "g", "df", "dg", "u", "c", "d", "ifail", "wa", "kwa", "act"),
    )
    lower, upper = _bounds(problem, nmax)
    session.add(
        x=_empty((nmax,)),
        g=_empty((m,)),
        df=_empty((nmax,)),
        dg=_empty((mwmax, nmax)),
        u=_empty((dim["mnn2"],)),
        c=_empty((nmax, nmax)),
        d=_empty((nmax,)),
        wa=_empty((dim["lwa"],)),
        kwa=_ints(dim["lkwa"]),
        act=_ints(dim["lact"]),
    )
    session["x"][:n] = problem.x0
    session.add(f=problem.f(problem.x0))
    session["g"][:m] = problem.g(problem.x0)

    order = _working_set(session["g"][:m], me, mw, opt["acc"])
    if order is None:
        msg = (
            "too many active constraints at the starting point, "
            "increase the option 'mw'"
        )
        raise ValueError(msg)
    session["kwa"][:mw] = order + 1
    session["act"][:m] = 1
    session["df"][:n] = problem.df(problem.x0)
    session["dg"][:mw, :n] = problem.dg(problem.x0)[order, :]
    problem.njev += 1

    def call() -> int:
        """Invoke the routine once.

        Returns:
            The flag returned by the routine.
        """
        return session.call(
            me,
            mw,
            n,
            session["x"],
            session["f"],
            session["g"],
            session["df"],
            session["dg"],
            session["u"],
            lower,
            upper,
            session["c"],
            session["d"],
            opt["acc"],
            opt["accqp"],
            opt["maxfun"],
            opt["maxiter"],
            opt["maxnm"],
            opt["rho"],
            opt["iprint"],
            opt["iout"],
            session["ifail"],
            session["wa"],
            session["kwa"],
            session["act"],
        )

    def functions() -> None:
        """Evaluate the model functions at the test point."""
        point = np.ascontiguousarray(session["x"][:n])
        session.add(f=problem.f(point))
        session["g"][:m] = problem.g(point)

    def gradients() -> None:
        """Evaluate the derivatives of the actual working set."""
        point = np.ascontiguousarray(session["x"][:n])
        rows = np.asarray(session["kwa"][:mw], dtype=np.intp) - 1
        session["df"][:n] = problem.df(point)
        session["dg"][:mw, :n] = problem.dg(point)[rows, :]
        problem.njev += 1
        if callback is not None:
            callback(np.array(point))

    _protocol(call, {NEED_FUNCTIONS: functions, NEED_GRADIENTS: gradients})

    out = _scalar_result(
        session,
        problem,
        session["x"][:n],
        session["f"],
        {
            "constr": np.array(session["g"][:m]),
            "active": np.array(session["act"][:m], dtype=bool),
        },
    )
    out["working_set"] = np.asarray(session["kwa"][:mw], dtype=np.intp) - 1
    return out


def solve_nlpqlf(
    problem: Problem,
    opt: dict[str, Any],
    callback: Callable[[np.ndarray], None] | None = None,
) -> dict[str, Any]:
    """Run ``NLPQLF`` on a problem with feasibility constraints.

    Args:
        problem: Problem holding the ordinary constraints.
        opt: Solver options, ``feasibility`` holds the problem
            describing the constraints that define the evaluable set.
        callback: Called with the current iterate after every iteration.

    Returns:
        Dictionary with the solution and the solver statistics.
    """
    feas = opt["feasibility"]
    n, mf, mef = problem.n, problem.m, problem.me
    m = feas.m
    dim = nlpqlf_sizes(n, mf, m)
    nmax, mmax = dim["nmax"], dim["mmax"]

    session = _Session(
        "nlpqlfq",
        ("x", "f", "g", "df", "dg", "u", "c", "d", "ifail", "wa", "kwa", "act"),
    )
    lower, upper = _bounds(problem, nmax)
    session.add(
        x=_empty((nmax,)),
        g=_empty((mmax,)),
        df=_empty((nmax,)),
        dg=_empty((mmax, nmax)),
        u=_empty((dim["mnn2"],)),
        c=_empty((nmax, nmax)),
        d=_empty((nmax,)),
        wa=_empty((dim["lwa"],)),
        kwa=_ints(dim["lkwa"]),
        act=_ints(dim["lact"]),
    )
    session["x"][:n] = problem.x0
    session.add(f=problem.f(problem.x0))
    if mf > 0:
        session["g"][:mf] = problem.g(problem.x0)
        session["dg"][:mf, :n] = problem.dg(problem.x0)
    if m > 0:
        session["g"][mf : mf + m] = feas.g(problem.x0)
        session["dg"][mf : mf + m, :n] = feas.dg(problem.x0)
    session["df"][:n] = problem.df(problem.x0)
    problem.njev += 1

    def call() -> int:
        """Invoke the routine once.

        Returns:
            The flag returned by the routine.
        """
        return session.call(
            mf,
            mef,
            m,
            n,
            session["x"],
            session["f"],
            session["g"],
            session["df"],
            session["dg"],
            session["u"],
            lower,
            upper,
            session["c"],
            session["d"],
            opt["acc"],
            opt["accf"],
            opt["accqp"],
            opt["maxfun"],
            opt["maxiter"],
            opt["maxnm"],
            opt["rho"],
            opt["iprint"],
            opt["iout"],
            session["ifail"],
            session["wa"],
            session["kwa"],
            session["act"],
        )

    def feasibility() -> None:
        """Evaluate the cheap feasibility constraints only."""
        if m > 0:
            point = np.ascontiguousarray(session["x"][:n])
            session["g"][mf : mf + m] = feas.g(point)

    def functions() -> None:
        """Evaluate the objective at a point known to be feasible."""
        point = np.ascontiguousarray(session["x"][:n])
        session.add(f=problem.f(point))
        if mf > 0:
            session["g"][:mf] = problem.g(point)

    def gradients() -> None:
        """Evaluate all derivatives at the accepted iterate."""
        point = np.ascontiguousarray(session["x"][:n])
        session["df"][:n] = problem.df(point)
        if mf > 0:
            session["dg"][:mf, :n] = problem.dg(point)
        if m > 0:
            session["dg"][mf : mf + m, :n] = feas.dg(point)
        problem.njev += 1
        if callback is not None:
            callback(np.array(point))

    _protocol(
        call,
        {
            NEED_FEASIBILITY: feasibility,
            NEED_FUNCTIONS: functions,
            NEED_GRADIENTS: gradients,
        },
    )

    total = mf + m
    out = _scalar_result(
        session,
        problem,
        session["x"][:n],
        session["f"],
        {
            "constr": np.array(session["g"][:mf])
            if mf > 0
            else _no_constraints(),
            "active": np.array(session["act"][:total], dtype=bool)
            if total > 0
            else np.zeros(0, dtype=bool),
        },
    )
    out["feasibility"] = (
        np.array(session["g"][mf : mf + m]) if m > 0 else _no_constraints()
    )
    return out


# ---------------------------------------------------------------------
# vector valued objectives
# ---------------------------------------------------------------------


def _seed_vector(problem: VectorProblem, g: np.ndarray, dg: np.ndarray) -> None:
    """Evaluate constraints and individual functions at the start.

    Args:
        problem: Problem whose objective is a vector.
        g: Array receiving the constraint and function values.
        dg: Array receiving the corresponding gradients.
    """
    n, m, ell = problem.n, problem.m, problem.nres
    if m > 0:
        g[:m] = problem.g(problem.x0)
        dg[:m, :n] = problem.dg(problem.x0)
    g[m : m + ell] = problem.fvec(problem.x0)
    dg[m : m + ell, :n] = problem.jvec(problem.x0)
    problem.njev += 1


def _vector_handlers(
    session: _Session,
    problem: VectorProblem,
    callback: Callable[[np.ndarray], None] | None,
) -> dict[int, Callable[[], None]]:
    """Return the request handlers of a vector valued problem.

    Args:
        session: Session holding the exchanged arrays.
        problem: Problem whose objective is a vector.
        callback: Called with the current iterate after every iteration.

    Returns:
        Mapping from a request flag to the action that satisfies it.
    """
    n, m, ell = problem.n, problem.m, problem.nres

    def functions() -> None:
        """Evaluate constraints and individual functions."""
        point = np.ascontiguousarray(session["x"][:n])
        if m > 0:
            session["g"][:m] = problem.g(point)
        session["g"][m : m + ell] = problem.fvec(point)

    def gradients() -> None:
        """Evaluate the corresponding gradients."""
        point = np.ascontiguousarray(session["x"][:n])
        if m > 0:
            session["dg"][:m, :n] = problem.dg(point)
        session["dg"][m : m + ell, :n] = problem.jvec(point)
        problem.njev += 1
        if callback is not None:
            callback(np.array(point))

    return {NEED_FUNCTIONS: functions, NEED_GRADIENTS: gradients}


def solve_fit(
    problem: VectorProblem,
    opt: dict[str, Any],
    callback: Callable[[np.ndarray], None] | None = None,
) -> dict[str, Any]:
    """Run one of the data fitting codes on a prepared problem.

    Args:
        problem: Problem whose objective is a vector of L individual
            functions.
        opt: Solver options, ``method`` selects the code.
        callback: Called with the current iterate after every iteration.

    Returns:
        Dictionary with the solution and the solver statistics.
    """
    method = opt["method"]
    n, m, me, ell = problem.n, problem.m, problem.me, problem.nres
    dim = fit_sizes(method, n, m, ell)
    lmmax, lnmax = dim["lmmax"], dim["lnmax"]

    session = _Session(
        method + "q",
        (
            "x",
            "f",
            "g",
            "df",
            "dg",
            "u",
            "xl",
            "xu",
            "c",
            "d",
            "ifail",
            "wa",
            "kwa",
            "act",
        ),
    )
    lower, upper = _bounds(problem, lnmax)
    session.add(
        x=_empty((lnmax,)),
        f=0.0,
        g=_empty((lmmax,)),
        df=_empty((lnmax,)),
        dg=_empty((lmmax, lnmax)),
        u=_empty((dim["lmnn2"],)),
        xl=lower,
        xu=upper,
        c=_empty((lnmax, lnmax)),
        d=_empty((lnmax,)),
        wa=_empty((dim["lwa"],)),
        kwa=_ints(dim["lkwa"]),
        act=_ints(dim["lact"]),
    )
    session["x"][:n] = problem.x0
    _seed_vector(problem, session["g"], session["dg"])

    def call() -> int:
        """Invoke the routine once.

        Returns:
            The flag returned by the routine.
        """
        return session.call(
            ell,
            m,
            me,
            n,
            session["x"],
            session["f"],
            session["g"],
            session["df"],
            session["dg"],
            session["u"],
            session["xl"],
            session["xu"],
            session["c"],
            session["d"],
            opt["acc"],
            opt["accqp"],
            opt["maxfun"],
            opt["maxiter"],
            opt["maxnm"],
            opt["rho"],
            opt["iprint"],
            opt["iout"],
            session["ifail"],
            session["wa"],
            session["kwa"],
            session["act"],
        )

    _protocol(call, _vector_handlers(session, problem, callback))

    point = np.ascontiguousarray(session["x"][:n])
    out = {
        "x": np.array(point),
        "fun": float(session["f"]),
        "residuals": problem.fvec(point),
        "constr": np.array(problem.g(point)) if m > 0 else _no_constraints(),
        "jac": np.zeros(n),
        "multipliers": np.array(session["u"]),
        "active": np.zeros(0, dtype=bool),
    }
    out.update(_statistics(session, problem))
    return out


def solve_nlpjob(
    problem: VectorProblem,
    opt: dict[str, Any],
    callback: Callable[[np.ndarray], None] | None = None,
) -> dict[str, Any]:
    """Run ``NLPJOB`` on a prepared multicriteria problem.

    Args:
        problem: Problem whose objective is a vector of L criteria.
        opt: Solver options, ``model``, ``imin``, ``weights`` and ``fk``
            select the scalar transformation.
        callback: Called with the current iterate after every iteration.

    Returns:
        Dictionary with the solution and the solver statistics.
    """
    n, m, me, ell = problem.n, problem.m, problem.me, problem.nres
    dim = nlpjob_sizes(n, m, ell)
    lmmax, lnmax = dim["lmmax"], dim["lnmax"]

    session = _Session(
        "nlpjob",
        (
            "x",
            "f",
            "g",
            "df",
            "dg",
            "u",
            "xl",
            "xu",
            "fw",
            "ifail",
            "wa",
            "kwa",
            "logwa",
        ),
    )
    lower, upper = _bounds(problem, lnmax)
    session.add(
        x=_empty((lnmax,)),
        f=0.0,
        g=_empty((lmmax,)),
        df=_empty((lnmax,)),
        dg=_empty((lmmax, lnmax)),
        u=_empty((dim["lmnn2"],)),
        xl=lower,
        xu=upper,
        fw=_empty((ell,)),
        wa=_empty((dim["lwa"],)),
        kwa=_ints(dim["lkwa"]),
        logwa=_ints(dim["llogwa"]),
    )
    weights = _vector_option(opt.get("weights"), ell)
    reference = _vector_option(opt.get("fk"), ell)
    session["x"][:n] = problem.x0
    _seed_vector(problem, session["g"], session["dg"])

    def call() -> int:
        """Invoke the routine once.

        Returns:
            The flag returned by the routine.
        """
        return session.call(
            ell,
            m,
            me,
            n,
            int(opt["model"]),
            int(opt["imin"]),
            session["x"],
            session["f"],
            session["g"],
            session["df"],
            session["dg"],
            session["u"],
            session["xl"],
            session["xu"],
            weights,
            reference,
            session["fw"],
            opt["acc"],
            opt["accqp"],
            opt["maxfun"],
            opt["maxiter"],
            opt["iprint"],
            opt["iout"],
            session["ifail"],
            session["wa"],
            session["kwa"],
            session["logwa"],
        )

    _protocol(call, _vector_handlers(session, problem, callback))

    point = np.ascontiguousarray(session["x"][:n])
    out = {
        "x": np.array(point),
        "fun": float(session["f"]),
        "objectives": np.array(session["fw"]),
        "constr": np.array(problem.g(point)) if m > 0 else _no_constraints(),
        "jac": np.zeros(n),
        "multipliers": np.array(session["u"]),
        "active": np.zeros(0, dtype=bool),
    }
    out.update(_statistics(session, problem))
    return out


def _vector_option(
    value: Sequence[float] | np.ndarray | None, size: int
) -> np.ndarray:
    """Return an option vector, defaulting to all ones.

    Args:
        value: User supplied sequence or ``None``.
        size: Expected length of the vector.

    Returns:
        A contiguous double precision array of the requested length.
    """
    out = np.ones(size, dtype=np.float64, order="F")
    if value is not None:
        out[:] = np.asarray(value, dtype=np.float64).ravel()
    return out


def _working_set(
    g: np.ndarray, me: int, mw: int, acc: float
) -> np.ndarray | None:
    """Return an initial working set of size ``mw``.

    All equality constraints are placed at the beginning, the remaining
    positions are filled with the indices of the inequality constraints
    that possess the smallest function values, i.e. which are active or
    closest to being active.

    Args:
        g: Constraint values at the starting point.
        me: Number of equality constraints.
        mw: Requested size of the working set.
        acc: Tolerance deciding whether a constraint is active.

    Returns:
        Zero based indices of the working set, or ``None`` if more than
        ``mw`` constraints are active.
    """
    m = g.size
    if me > mw:
        return None
    tail = g[me:]
    if int(np.count_nonzero(tail <= acc)) + me > mw:
        return None
    order = np.empty(mw, dtype=np.intp)
    order[:me] = np.arange(me)
    rest = mw - me
    if rest >= tail.size:
        order[me : me + tail.size] = np.arange(me, m)
    elif rest > 0:
        pick = np.argpartition(tail, rest - 1)[:rest]
        order[me:] = np.sort(pick) + me
    return order
