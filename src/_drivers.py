"""Reverse communication drivers for the Fortran routines.

Every routine of the NLPQL family returns control to the calling
program whenever new function or gradient values are required.  The
drivers below run that protocol and keep all working arrays alive
between two calls.
"""

from __future__ import annotations

from collections.abc import Callable
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


def _empty(shape: tuple[int, ...]) -> np.ndarray:
    """Return a zeroed Fortran ordered double precision array.

    Args:
        shape: Shape of the array.

    Returns:
        The newly allocated array.
    """
    return np.zeros(shape, dtype=np.float64, order="F")


def solve_nlpqlp(  # ruff:ignore[too-many-locals]
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
    n, m, me = problem.n, problem.m, problem.me
    nproc = int(opt["nproc"])
    dim = nlpqlp_sizes(n, m, nproc)
    nmax, mmax, mnn2 = dim["nmax"], dim["mmax"], dim["mnn2"]

    x = _empty((nmax, nproc))
    f = _empty((nproc,))
    g = _empty((mmax, nproc))
    df = _empty((nmax,))
    dg = _empty((mmax, nmax))
    u = _empty((mnn2,))
    c = _empty((nmax, nmax))
    d = _empty((nmax,))
    wa = _empty((dim["lwa"],))
    kwa = np.zeros(dim["lkwa"], dtype=np.int32)
    act = np.zeros(dim["lact"], dtype=np.int32)

    xlw = _empty((nmax,))
    xuw = _empty((nmax,))
    xlw[:n] = problem.xl
    xuw[:n] = problem.xu
    xlw[n:] = -1.0e30
    xuw[n:] = 1.0e30

    x[:n, 0] = problem.x0
    f[0] = problem.f(problem.x0)
    if m > 0:
        g[:m, 0] = problem.g(problem.x0)
    df[:n] = problem.df(problem.x0)
    if m > 0:
        dg[:m, :n] = problem.dg(problem.x0)
    problem.njev += 1

    ifail = 0
    while True:
        out = _nlpql.nlpqlpq(
            m,
            me,
            n,
            x,
            f,
            g,
            df,
            dg,
            u,
            xlw,
            xuw,
            c,
            d,
            opt["acc"],
            opt["accqp"],
            opt["stpmin"],
            opt["maxfun"],
            opt["maxiter"],
            opt["maxnm"],
            opt["rho"],
            opt["iprint"],
            opt["mode"],
            opt["iout"],
            ifail,
            wa,
            kwa,
            act,
            opt["lql"],
        )
        x, f, g, df, dg, u, c, d, ifail, wa, kwa, act = out
        if ifail == NEED_FUNCTIONS:
            for k in range(nproc):
                xk = np.ascontiguousarray(x[:n, k])
                f[k] = problem.f(xk)
                if m > 0:
                    g[:m, k] = problem.g(xk)
        elif ifail == NEED_GRADIENTS:
            xk = np.ascontiguousarray(x[:n, 0])
            df[:n] = problem.df(xk)
            if m > 0:
                dg[:m, :n] = problem.dg(xk)
            problem.njev += 1
            if callback is not None:
                callback(xk.copy())
        else:
            break

    return {
        "x": np.array(x[:n, 0]),
        "fun": float(f[0]),
        "constr": np.array(g[:m, 0]) if m > 0 else np.zeros(0),
        "jac": np.array(df[:n]),
        "multipliers": np.array(u[:mnn2]),
        "active": np.array(act[:m], dtype=bool)
        if m > 0
        else np.zeros(0, dtype=bool),
        "status": int(ifail),
        "nit": int(kwa[2]),
        "nqp": int(kwa[3]),
    }


def solve_nlpql(  # ruff:ignore[too-many-locals]
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
    nmax, mmax, mnn = dim["nmax"], dim["mmax"], dim["mnn"]

    x = _empty((nmax,))
    g = _empty((mmax,))
    df = _empty((nmax,))
    dg = _empty((mmax, nmax))
    u = _empty((mnn,))
    c = _empty((nmax, nmax))
    d = _empty((nmax,))
    wa = _empty((dim["lwa"],))
    kwa = np.zeros(dim["lkwa"], dtype=np.int32)
    act = np.zeros(dim["lact"], dtype=np.int32)

    xlw = _empty((nmax,))
    xuw = _empty((nmax,))
    xlw[:n] = problem.xl
    xuw[:n] = problem.xu
    xlw[n:] = -1.0e30
    xuw[n:] = 1.0e30

    x[:n] = problem.x0
    fval = problem.f(problem.x0)
    if m > 0:
        g[:m] = problem.g(problem.x0)
    df[:n] = problem.df(problem.x0)
    if m > 0:
        dg[:m, :n] = problem.dg(problem.x0)
    problem.njev += 1

    ifail = 0
    while True:
        out = _nlpql.nlpql(
            m,
            me,
            n,
            x,
            fval,
            g,
            df,
            dg,
            u,
            xlw,
            xuw,
            c,
            d,
            opt["acc"],
            opt["maxfun"],
            opt["maxiter"],
            opt["iprint"],
            opt["iout"],
            ifail,
            wa,
            kwa,
            act,
        )
        x, fval, g, df, dg, u, c, d, ifail, wa, kwa, act = out
        xk = np.ascontiguousarray(x[:n])
        if ifail == NEED_FUNCTIONS:
            fval = problem.f(xk)
            if m > 0:
                g[:m] = problem.g(xk)
        elif ifail == NEED_GRADIENTS:
            df[:n] = problem.df(xk)
            if m > 0:
                dg[:m, :n] = problem.dg(xk)
            problem.njev += 1
            if callback is not None:
                callback(xk.copy())
        else:
            break

    return {
        "x": np.array(x[:n]),
        "fun": float(fval),
        "constr": np.array(g[:m]) if m > 0 else np.zeros(0),
        "jac": np.array(df[:n]),
        "multipliers": np.array(u[:mnn]),
        "active": np.array(act[:m], dtype=bool)
        if m > 0
        else np.zeros(0, dtype=bool),
        "status": int(ifail),
        "nit": int(kwa[2]),
        "nqp": int(kwa[3]),
    }


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

    x = np.ascontiguousarray(problem.x0.copy())
    g = _empty((m,))
    wa = _empty((dim["lwa"],))
    kwa = np.zeros(dim["lkwa"], dtype=np.int32)
    act = np.zeros(dim["lact"], dtype=np.int32)

    fval = problem.f(x)
    if m > 0:
        g[:m] = problem.g(x)

    ifail = 0
    while True:
        out = _nlpql.nlpqly(
            me,
            x,
            fval,
            g,
            problem.xl,
            problem.xu,
            opt["acc"],
            opt["maxiter"],
            opt["iprint"],
            opt["iout"],
            ifail,
            wa,
            kwa,
            act,
        )
        x, fval, g, ifail, wa, kwa, act = out
        if ifail >= 0:
            break
        fval = problem.f(x)
        if m > 0:
            g[:m] = problem.g(x)
        if callback is not None:
            callback(np.array(x))

    return {
        "x": np.array(x[:n]),
        "fun": float(fval),
        "constr": np.array(g[:m]) if m > 0 else np.zeros(0),
        "jac": np.zeros(n),
        "multipliers": np.zeros(0),
        "active": np.array(act[:m], dtype=bool)
        if m > 0
        else np.zeros(0, dtype=bool),
        "status": int(ifail),
        "nit": int(kwa[2]),
        "nqp": int(kwa[3]),
    }


def solve_nlpqlb(  # ruff:ignore[too-many-locals, too-many-statements]
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
    nmax, mwmax, mnn2 = dim["nmax"], dim["mwmax"], dim["mnn2"]

    x = _empty((nmax,))
    g = _empty((m,))
    df = _empty((nmax,))
    dg = _empty((mwmax, nmax))
    u = _empty((mnn2,))
    c = _empty((nmax, nmax))
    d = _empty((nmax,))
    wa = _empty((dim["lwa"],))
    kwa = np.zeros(dim["lkwa"], dtype=np.int32)
    act = np.zeros(dim["lact"], dtype=np.int32)

    xlw = _empty((nmax,))
    xuw = _empty((nmax,))
    xlw[:n] = problem.xl
    xuw[:n] = problem.xu
    xlw[n:] = -1.0e30
    xuw[n:] = 1.0e30

    x[:n] = problem.x0
    fval = problem.f(problem.x0)
    g[:m] = problem.g(problem.x0)

    order = _working_set(g[:m], me, mw, opt["acc"])
    if order is None:
        msg = (
            "too many active constraints at the starting point, "
            "increase the option 'mw'"
        )
        raise ValueError(msg)
    kwa[:mw] = order + 1
    act[:m] = 1

    full = problem.dg(problem.x0)
    problem.njev += 1
    df[:n] = problem.df(problem.x0)
    dg[:mw, :n] = full[order, :]

    ifail = 0
    while True:
        out = _nlpql.nlpqlbq(
            me,
            mw,
            n,
            x,
            fval,
            g,
            df,
            dg,
            u,
            xlw,
            xuw,
            c,
            d,
            opt["acc"],
            opt["accqp"],
            opt["maxfun"],
            opt["maxiter"],
            opt["maxnm"],
            opt["rho"],
            opt["iprint"],
            opt["iout"],
            ifail,
            wa,
            kwa,
            act,
        )
        x, fval, g, df, dg, u, c, d, ifail, wa, kwa, act = out
        xk = np.ascontiguousarray(x[:n])
        if ifail == NEED_FUNCTIONS:
            fval = problem.f(xk)
            g[:m] = problem.g(xk)
        elif ifail == NEED_GRADIENTS:
            order = np.asarray(kwa[:mw], dtype=np.intp) - 1
            df[:n] = problem.df(xk)
            full = problem.dg(xk)
            problem.njev += 1
            dg[:mw, :n] = full[order, :]
            if callback is not None:
                callback(xk.copy())
        else:
            break

    return {
        "x": np.array(x[:n]),
        "fun": float(fval),
        "constr": np.array(g[:m]),
        "jac": np.array(df[:n]),
        "multipliers": np.array(u[:mnn2]),
        "active": np.array(act[:m], dtype=bool),
        "working_set": np.asarray(kwa[:mw], dtype=np.intp) - 1,
        "status": int(ifail),
        "nit": 0,
        "nqp": 0,
    }


def solve_nlpqlg(  # ruff:ignore[too-many-locals]
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
    n, m, me = problem.n, problem.m, problem.me
    nproc = int(opt["nproc"])
    dim = nlpqlg_sizes(n, m, nproc)
    nmax, mmax, mnn2 = dim["nmax"], dim["mmax"], dim["mnn2"]

    x = _empty((nmax, nproc))
    f = _empty((nproc,))
    g = _empty((mmax, nproc))
    df = _empty((nmax,))
    dg = _empty((mmax, nmax))
    u = _empty((mnn2,))
    c = _empty((nmax, nmax))
    d = _empty((nmax,))
    wa = _empty((dim["lwa"],))
    kwa = np.zeros(dim["lkwa"], dtype=np.int32)
    act = np.zeros(dim["lact"], dtype=np.int32)

    xlw = _empty((nmax,))
    xuw = _empty((nmax,))
    xlw[:n] = problem.xl
    xuw[:n] = problem.xu
    xlw[n:] = -1.0e30
    xuw[n:] = 1.0e30

    x[:n, 0] = problem.x0
    f[0] = problem.f(problem.x0)
    if m > 0:
        g[:m, 0] = problem.g(problem.x0)
    df[:n] = problem.df(problem.x0)
    if m > 0:
        dg[:m, :n] = problem.dg(problem.x0)
    problem.njev += 1

    ifail = 0
    while True:
        out = _nlpql.nlpqlgq(
            m,
            me,
            n,
            x,
            f,
            g,
            df,
            dg,
            u,
            xlw,
            xuw,
            c,
            d,
            opt["acc"],
            opt["accqp"],
            opt["stpmin"],
            opt["maxfun"],
            opt["maxiter"],
            opt["maxnm"],
            opt["rho"],
            opt["ncycle"],
            opt["iprint"],
            opt["mode"],
            opt["iout"],
            ifail,
            wa,
            kwa,
            act,
            opt["lql"],
        )
        x, f, g, df, dg, u, c, d, ifail, wa, kwa, act = out
        if ifail == NEED_FUNCTIONS:
            for k in range(nproc):
                xk = np.ascontiguousarray(x[:n, k])
                f[k] = problem.f(xk)
                if m > 0:
                    g[:m, k] = problem.g(xk)
        elif ifail == NEED_GRADIENTS:
            xk = np.ascontiguousarray(x[:n, 0])
            df[:n] = problem.df(xk)
            if m > 0:
                dg[:m, :n] = problem.dg(xk)
            problem.njev += 1
            if callback is not None:
                callback(xk.copy())
        else:
            break

    return {
        "x": np.array(x[:n, 0]),
        "fun": float(f[0]),
        "constr": np.array(g[:m, 0]) if m > 0 else np.zeros(0),
        "jac": np.array(df[:n]),
        "multipliers": np.array(u[:mnn2]),
        "active": np.array(act[:m], dtype=bool)
        if m > 0
        else np.zeros(0, dtype=bool),
        "status": int(ifail),
        "nit": int(kwa[2]),
        "nqp": int(kwa[3]),
    }


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


#: Only the feasibility constraints are requested by NLPQLF.
NEED_FEASIBILITY = -3


def solve_fit(  # ruff:ignore[too-many-locals]
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

    x = _empty((lnmax,))
    g = _empty((lmmax,))
    df = _empty((lnmax,))
    dg = _empty((lmmax, lnmax))
    u = _empty((dim["lmnn2"],))
    c = _empty((lnmax, lnmax))
    d = _empty((lnmax,))
    xl = _empty((lnmax,))
    xu = _empty((lnmax,))
    wa = _empty((dim["lwa"],))
    kwa = np.zeros(dim["lkwa"], dtype=np.int32)
    act = np.zeros(dim["lact"], dtype=np.int32)

    xl[:n] = problem.xl
    xu[:n] = problem.xu
    xl[n:] = -1.0e30
    xu[n:] = 1.0e30
    x[:n] = problem.x0
    if m > 0:
        g[:m] = problem.g(problem.x0)
        dg[:m, :n] = problem.dg(problem.x0)
    g[m : m + ell] = problem.fvec(problem.x0)
    dg[m : m + ell, :n] = problem.jvec(problem.x0)
    problem.njev += 1

    solver = getattr(_nlpql, method + "q")
    fval = 0.0
    ifail = 0
    while True:
        out = solver(
            ell,
            m,
            me,
            n,
            x,
            fval,
            g,
            df,
            dg,
            u,
            xl,
            xu,
            c,
            d,
            opt["acc"],
            opt["accqp"],
            opt["maxfun"],
            opt["maxiter"],
            opt["maxnm"],
            opt["rho"],
            opt["iprint"],
            opt["iout"],
            ifail,
            wa,
            kwa,
            act,
        )
        x, fval, g, df, dg, u, xl, xu, c, d, ifail, wa, kwa, act = out
        xk = np.ascontiguousarray(x[:n])
        if ifail == NEED_FUNCTIONS:
            if m > 0:
                g[:m] = problem.g(xk)
            g[m : m + ell] = problem.fvec(xk)
        elif ifail == NEED_GRADIENTS:
            if m > 0:
                dg[:m, :n] = problem.dg(xk)
            dg[m : m + ell, :n] = problem.jvec(xk)
            problem.njev += 1
            if callback is not None:
                callback(xk.copy())
        else:
            break

    xk = np.ascontiguousarray(x[:n])
    return {
        "x": np.array(xk),
        "fun": float(fval),
        "residuals": problem.fvec(xk),
        "constr": np.array(problem.g(xk)) if m > 0 else np.zeros(0),
        "jac": np.zeros(n),
        "multipliers": np.array(u),
        "active": np.zeros(0, dtype=bool),
        "status": int(ifail),
        "nit": int(kwa[2]),
        "nqp": int(kwa[3]),
    }


def solve_nlpjob(  # ruff:ignore[too-many-locals, too-many-statements]
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

    x = _empty((lnmax,))
    g = _empty((lmmax,))
    df = _empty((lnmax,))
    dg = _empty((lmmax, lnmax))
    u = _empty((dim["lmnn2"],))
    xl = _empty((lnmax,))
    xu = _empty((lnmax,))
    weights = _empty((ell,))
    fkvec = _empty((ell,))
    fwvec = _empty((ell,))
    wa = _empty((dim["lwa"],))
    kwa = np.zeros(dim["lkwa"], dtype=np.int32)
    logwa = np.zeros(dim["llogwa"], dtype=np.int32)

    if opt.get("weights") is not None:
        weights[:] = np.asarray(opt["weights"], dtype=np.float64).ravel()
    else:
        weights[:] = 1.0
    if opt.get("fk") is not None:
        fkvec[:] = np.asarray(opt["fk"], dtype=np.float64).ravel()
    else:
        fkvec[:] = 1.0
    xl[:n] = problem.xl
    xu[:n] = problem.xu
    xl[n:] = -1.0e30
    xu[n:] = 1.0e30
    x[:n] = problem.x0
    if m > 0:
        g[:m] = problem.g(problem.x0)
        dg[:m, :n] = problem.dg(problem.x0)
    g[m : m + ell] = problem.fvec(problem.x0)
    dg[m : m + ell, :n] = problem.jvec(problem.x0)
    problem.njev += 1

    fval = 0.0
    ifail = 0
    while True:
        out = _nlpql.nlpjob(
            ell,
            m,
            me,
            n,
            int(opt["model"]),
            int(opt["imin"]),
            x,
            fval,
            g,
            df,
            dg,
            u,
            xl,
            xu,
            weights,
            fkvec,
            fwvec,
            opt["acc"],
            opt["accqp"],
            opt["maxfun"],
            opt["maxiter"],
            opt["iprint"],
            opt["iout"],
            ifail,
            wa,
            kwa,
            logwa,
        )
        (x, fval, g, df, dg, u, xl, xu, fwvec, ifail, wa, kwa, logwa) = out
        xk = np.ascontiguousarray(x[:n])
        if ifail == NEED_FUNCTIONS:
            if m > 0:
                g[:m] = problem.g(xk)
            g[m : m + ell] = problem.fvec(xk)
        elif ifail == NEED_GRADIENTS:
            if m > 0:
                dg[:m, :n] = problem.dg(xk)
            dg[m : m + ell, :n] = problem.jvec(xk)
            problem.njev += 1
            if callback is not None:
                callback(xk.copy())
        else:
            break

    xk = np.ascontiguousarray(x[:n])
    return {
        "x": np.array(xk),
        "fun": float(fval),
        "objectives": np.array(fwvec),
        "constr": np.array(problem.g(xk)) if m > 0 else np.zeros(0),
        "jac": np.zeros(n),
        "multipliers": np.array(u),
        "active": np.zeros(0, dtype=bool),
        "status": int(ifail),
        "nit": int(kwa[2]),
        "nqp": int(kwa[3]),
    }


def solve_nlpqlf(  # ruff:ignore[too-many-locals, too-many-statements]
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

    x = _empty((nmax,))
    g = _empty((mmax,))
    df = _empty((nmax,))
    dg = _empty((mmax, nmax))
    u = _empty((dim["mnn2"],))
    c = _empty((nmax, nmax))
    d = _empty((nmax,))
    xl = _empty((nmax,))
    xu = _empty((nmax,))
    wa = _empty((dim["lwa"],))
    kwa = np.zeros(dim["lkwa"], dtype=np.int32)
    act = np.zeros(dim["lact"], dtype=np.int32)

    xl[:n] = problem.xl
    xu[:n] = problem.xu
    xl[n:] = -1.0e30
    xu[n:] = 1.0e30
    x[:n] = problem.x0
    fval = problem.f(problem.x0)
    if mf > 0:
        g[:mf] = problem.g(problem.x0)
        dg[:mf, :n] = problem.dg(problem.x0)
    if m > 0:
        g[mf : mf + m] = feas.g(problem.x0)
        dg[mf : mf + m, :n] = feas.dg(problem.x0)
    df[:n] = problem.df(problem.x0)
    problem.njev += 1

    ifail = 0
    while True:
        out = _nlpql.nlpqlfq(
            mf,
            mef,
            m,
            n,
            x,
            fval,
            g,
            df,
            dg,
            u,
            xl,
            xu,
            c,
            d,
            opt["acc"],
            opt["accf"],
            opt["accqp"],
            opt["maxfun"],
            opt["maxiter"],
            opt["maxnm"],
            opt["rho"],
            opt["iprint"],
            opt["iout"],
            ifail,
            wa,
            kwa,
            act,
        )
        x, fval, g, df, dg, u, c, d, ifail, wa, kwa, act = out
        xk = np.ascontiguousarray(x[:n])
        if ifail == NEED_FEASIBILITY:
            if m > 0:
                g[mf : mf + m] = feas.g(xk)
        elif ifail == NEED_FUNCTIONS:
            fval = problem.f(xk)
            if mf > 0:
                g[:mf] = problem.g(xk)
        elif ifail == NEED_GRADIENTS:
            df[:n] = problem.df(xk)
            if mf > 0:
                dg[:mf, :n] = problem.dg(xk)
            if m > 0:
                dg[mf : mf + m, :n] = feas.dg(xk)
            problem.njev += 1
            if callback is not None:
                callback(xk.copy())
        else:
            break

    return {
        "x": np.array(x[:n]),
        "fun": float(fval),
        "constr": np.array(g[:mf]) if mf > 0 else np.zeros(0),
        "feasibility": np.array(g[mf : mf + m]) if m > 0 else np.zeros(0),
        "jac": np.array(df[:n]),
        "multipliers": np.array(u),
        "active": np.array(act[: mf + m], dtype=bool)
        if mf + m > 0
        else np.zeros(0, dtype=bool),
        "status": int(ifail),
        "nit": int(kwa[2]),
        "nqp": int(kwa[3]),
    }
