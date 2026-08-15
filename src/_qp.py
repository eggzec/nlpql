"""Direct access to the convex quadratic programming solver QL."""

from __future__ import annotations

import numpy as np

from . import _nlpql as raw
from ._optimize import OptimizeResult
from ._status import IFAIL_MESSAGES


def solve_qp(  # ruff:ignore[too-many-arguments, too-many-locals, too-many-positional-arguments]
    c: object,
    d: object,
    a: object = None,
    b: object = None,
    me: int = 0,
    xl: object = None,
    xu: object = None,
    eps: float = 1.0e-12,
) -> OptimizeResult:
    """Solve a strictly convex quadratic program with ``QL``.

    The problem solved is

    .. code-block:: text

        min  1/2 x^T C x + d^T x
             a_j^T x + b_j  = 0 ,  j = 1, ..., me
             a_j^T x + b_j >= 0 ,  j = me+1, ..., m
             xl <= x <= xu

    Args:
        c: Symmetric positive definite objective matrix of shape
            ``(n, n)``.
        d: Linear part of the objective, length ``n``.
        a: Constraint matrix of shape ``(m, n)`` or ``None``.
        b: Right hand side of the constraints, length ``m``.
        me: Number of equality constraints, they occupy the first rows.
        xl: Lower bounds of the variables or ``None``.
        xu: Upper bounds of the variables or ``None``.
        eps: Final termination accuracy.

    Returns:
        An :class:`OptimizeResult` with the fields ``x``, ``fun``,
        ``multipliers``, ``status``, ``success`` and ``message``.

    Example:
        >>> import numpy as np
        >>> from nlpql import solve_qp
        >>> res = solve_qp(np.eye(2), np.array([-1.0, -1.0]))
        >>> np.allclose(res.x, [1.0, 1.0])
        True
    """
    cmat = np.asfortranarray(np.atleast_2d(np.asarray(c, dtype=np.float64)))
    n = cmat.shape[0]
    dvec = np.ascontiguousarray(np.asarray(d, dtype=np.float64).ravel())
    if a is None:
        amat = np.zeros((1, n), dtype=np.float64, order="F")
        bvec = np.zeros(1, dtype=np.float64)
        m = 0
    else:
        amat = np.asfortranarray(np.atleast_2d(np.asarray(a, dtype=np.float64)))
        bvec = np.ascontiguousarray(np.asarray(b, dtype=np.float64).ravel())
        m = amat.shape[0]
    big = 1.0e30
    low = np.full(n, -big) if xl is None else np.asarray(xl, dtype=np.float64)
    upp = np.full(n, big) if xu is None else np.asarray(xu, dtype=np.float64)
    mmax = max(m, 1)
    mnn = m + n + n
    lwar = (3 * n * n) // 2 + 10 * n + 2 * mmax + 2
    war = np.zeros(lwar, dtype=np.float64)
    iwar = np.zeros(n, dtype=np.int32)
    cwork = np.asfortranarray(cmat.copy())
    xsol = np.zeros(n, dtype=np.float64)
    umul = np.zeros(mnn, dtype=np.float64)
    amat = np.asfortranarray(np.resize(amat, (mmax, n)))
    _cw, x, u, ifail, _war, _iwar = raw.ql(
        m,
        me,
        n,
        cwork,
        dvec,
        amat,
        bvec,
        low,
        upp,
        xsol,
        umul,
        eps,
        1,
        6,
        0,
        0,
        war,
        iwar,
    )
    fval = 0.5 * float(x @ (cmat @ x)) + float(dvec @ x)
    return OptimizeResult(
        x=np.array(x),
        fun=fval,
        multipliers=np.array(u),
        status=int(ifail),
        success=int(ifail) == 0,
        message=IFAIL_MESSAGES.get(int(ifail), f"QL flag {int(ifail)}."),
    )
